"""Core schema.

Two tables, and the distinction between them is the one the whole pipeline turns on:

- **`abstract`** — every PubMed record we know about, whether or not we could read it.
  Where `sync` lands, and the only thing it writes.
- **`paper`** — a record we actually retrieved full text for. Created by the PMC fetch.

So `select(Paper)` **is** the working corpus: every consumer operates on papers it can read, with no
"has full text" filter anywhere, because being in `paper` is that filter. `abstract LEFT JOIN paper`
answers the question when the ingest side needs to ask it.

**Each table keeps raw + parsed of the source it owns.** `abstract` owns the PubMed citation record
(`pubmed_xml` → `pubmed_json`); `paper` owns the PMC article (`jats_xml` → `fulltext_jats`). The raw
form is always kept, so nothing the parser didn't think to extract is lost, and a parser change can be
caught up offline with no NCBI round-trip. The two halves do that differently: a `pubmed_xml` parser
change is replayed by `store.reparse_bibliographic`, which rewrites `pubmed_json`/`paper.citation`,
while `fulltext_jats` needs no replay for the change that actually recurs — section classification is
re-derived as it is read (`jats.effective_type`), so widening a heading pattern takes effect on the
stored corpus immediately. `jats_xml` is still kept against a deeper JATS parser change (one altering
which sections exist, or their text), which read-time classification could not cover.

The parsed JSON is why so few real columns remain: `pubmed_json` is a complete `asdict()` of the parse
— authors, MeSH, the abstract, every bibliographic locator — so a field is only a column when something
*queries* it. Display fields on `Paper` are properties reading through the relationship, not stored
twice.

**`corpus_config` is the third table and a different kind of thing** — not corpus content but a record
of what the corpus was built from. See its own docstring.
"""

from datetime import date, datetime
from typing import Any

from pubmedcorpus.db import Base
from sqlalchemy import (Boolean, Computed, Date, DateTime, Float, ForeignKey,
                        Integer, String, Text, func)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column, relationship


class Abstract(Base):
    """One PubMed record. Every synced record has exactly one of these.

    Named for what it holds beyond identity: the citation record and its abstract.
    It does **not** hold the paper's text — see `Paper`.
    """

    __tablename__ = "abstract"

    pmid: Mapped[str] = mapped_column(String(20), primary_key=True)

    # The verbatim <PubmedArticle> element. Source of truth: everything below and
    # everything on `paper` re-derives from it.
    pubmed_xml: Mapped[str] = mapped_column(Text)
    # `asdict(ParsedPaper)` minus the XML — title, journal, doi, the locators, the
    # ordered author list with ORCID and affiliation, mesh_terms, the abstract text.
    # Queryable (JSONB), which is why none of those need columns of their own.
    pubmed_json: Mapped[dict] = mapped_column(JSONB)

    # Columned because they are queried, not merely stored: `entry_date` sets sync's
    # resume window, `pub_year` groups the report's decade histogram and orders the
    # fetch queue, `pmc_id` is the fetch handle.
    entry_date: Mapped[date | None] = mapped_column(Date, index=True)
    pub_year: Mapped[int | None] = mapped_column(Integer, index=True)
    pmc_id: Mapped[str | None] = mapped_column(String(20), index=True)
    is_retracted: Mapped[bool] = mapped_column(Boolean, default=False, index=True)

    # Whether a full-text attempt was made and came back empty. Combined with the
    # `paper` row this gives three states with no enum: no paper row and not failed =
    # never attempted; failed = asked, no body; paper row = we have it. A retry keys on
    # `fetch_failed`, and `fetch_failed AND pmc_id IS NOT NULL` is the recheck set —
    # it had a PMCID but no open-access body, so an embargo may since have lifted.
    fetch_failed: Mapped[bool] = mapped_column(Boolean, default=False, index=True)

    # Title (weight A) + abstract (weight B) + MeSH terms (weight C), in `english` and `simple`.
    #
    # MeSH sits *below* the abstract deliberately: it is a curated topical label, so a hit there is
    # weaker evidence of relevance than the paper's own prose — ranking it above the abstract would
    # float topic-adjacent papers over ones that actually discuss the query.
    #
    # **Known wart, kept deliberately: this is a consumer concern living on a library model.** It
    # exists to serve a lexical retrieval path in the application that first needed it, not anything
    # this library does. It is generated, so it costs nothing to carry and cannot drift — but a
    # consumer that wants different weights or a different language configuration has to change the
    # library rather than its own code. Left as-is for v0.1 because moving it would mean either
    # dropping a column the one existing consumer depends on, or inventing a configurability hook
    # before a second consumer has said what shape it needs.
    #
    # Three properties, each load-bearing: **generated**, so it cannot drift from `pubmed_json`
    # and survives a bibliographic reparse with nothing to re-run; **deferred**, so a plain
    # `select(Abstract)` never drags a tsvector into Python, which nothing has any use for; and
    # **`Computed`**, so SQLAlchemy leaves the column out of INSERT and UPDATE — without it,
    # constructing a row with this field set would produce a statement Postgres rejects.
    #
    # The expression is spelled out here and again in the migration, deliberately: a migration is
    # a frozen historical record and must not import from a model that keeps changing.
    #
    # `pubmed_json->>'mesh_terms'` hands the whole array over as one string — `["Epilepsy",
    # "Death, Sudden"]` — whose brackets, quotes and commas tokenise to nothing. Done this way
    # because a generated column's expression must be immutable, which rules out the `unnest` that
    # per-term tokenising would need. **The cost: every MeSH term shares one positional sequence,
    # and punctuation does not break adjacency in a tsvector, so a phrase query (`<->`,
    # `phraseto_tsquery`) can match across the seam between two unrelated terms — "epilepsy death"
    # from the example above.** Harmless to the plain `websearch_to_tsquery` matching the one
    # consumer does, and weight C limits the ranking damage; a consumer that wants phrase search
    # over MeSH needs a normalised term table instead, not a fix to this column.
    search_tsv: Mapped[str | None] = mapped_column(
        TSVECTOR,
        Computed(
            "setweight(to_tsvector('english', coalesce(pubmed_json->>'title', '')), 'A')"
            " || setweight(to_tsvector('english', coalesce(pubmed_json->>'abstract_text', '')), 'B')"
            " || setweight(to_tsvector('english', coalesce(pubmed_json->>'mesh_terms', '')), 'C')"
            " || setweight(to_tsvector('simple',  coalesce(pubmed_json->>'title', '')), 'A')"
            " || setweight(to_tsvector('simple',  coalesce(pubmed_json->>'abstract_text', '')), 'B')"
            " || setweight(to_tsvector('simple',  coalesce(pubmed_json->>'mesh_terms', '')), 'C')",
            persisted=True,
        ),
        deferred=True,
    )

    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    paper: Mapped["Paper | None"] = relationship(
        back_populates="abstract", uselist=False, cascade="all, delete-orphan"
    )


class Paper(Base):
    """A paper we can actually read: PMC gave us a body.

    Existence is the signal. Nothing filters on a status column, because a row here
    *means* we have the full text — which is why every downstream queue and export
    just selects papers.
    """

    __tablename__ = "paper"

    pmid: Mapped[str] = mapped_column(
        ForeignKey("abstract.pmid", ondelete="CASCADE"), primary_key=True
    )

    # The verbatim <article> element from PMC. Kept for the same reason as
    # `pubmed_xml`: `parse_article` classifies sections and drops everything outside
    # <sec>/<p> — tables, figure captions, boxed text — which would otherwise be
    # unrecoverable without re-asking PMC.
    jats_xml: Mapped[str] = mapped_column(Text)
    # Section-structured full text, plus the article's own abstract and licence label.
    # Structured so a consumer can target Discussion/Conclusion, not the whole article.
    fulltext_jats: Mapped[dict] = mapped_column(JSONB)

    # The rendered Vancouver reference, frozen when the paper was promoted. Frozen
    # rather than computed on read: it is what a published bibliography renders from,
    # and the locator survives here even though volume/issue/pages are not columns
    # anywhere. A bibliographic reparse re-renders it.
    citation: Mapped[str] = mapped_column(Text)

    abstract: Mapped["Abstract"] = relationship(back_populates="paper")

    # --- Read-only views onto the record ------------------------------------
    # Properties, not columns: these are display fields, and storing them here would
    # duplicate what `pubmed_json` already holds. NOTE they are Python-only — they
    # cannot appear in a WHERE or ORDER BY. Query `Abstract` for that, and load the
    # relationship eagerly when rendering many papers at once.

    @property
    def title(self) -> str | None:
        return (self.abstract.pubmed_json or {}).get("title")

    @property
    def journal(self) -> str | None:
        return (self.abstract.pubmed_json or {}).get("journal")

    @property
    def pub_year(self) -> int | None:
        return self.abstract.pub_year

    @property
    def record(self) -> dict[str, Any]:
        """The whole parsed PubMed record, for the fields that have no property."""
        return self.abstract.pubmed_json or {}


class CorpusConfig(Base):
    """What this database was built from — one row per distinct ingest configuration.

    **The gap it fills is a real bug, not bookkeeping.** `sync`'s resume cursor is
    `max(abstract.entry_date)` over the whole table, with no awareness of the query that produced those
    rows. Change the query and the next incremental sync resumes from the *old* corpus's high-water
    mark under the *new* definition: the corpus silently becomes a mixture, and nothing records or
    reports it.

    **Recording only — deliberately not enforcement.** Records carry no `corpus_config_id`, and `sync`
    does not refuse when the live config differs from the last row. Read that honestly:

    - **What it buys:** a query change becomes visible instead of invisible. "What definitions has this
      database been built under, and when" becomes a query.
    - **What it does not buy:** the drift bug remains. Nothing prevents a mixed corpus, and nothing can
      say authoritatively which of two definitions produced a given paper.
    - **What is still inferable:** these rows carry `created_at` and `Abstract` carries `ingested_at`,
      so the definition active when a record arrived is recoverable by timestamp — not authoritative,
      since a sync can straddle a change, but usable.

    What would close it, in order of cost: a nullable `abstract.corpus_config_id` (one column, makes it
    per-record and authoritative), then a confirmation step that diffs the live config against the last
    row and refuses without acknowledgement — which would make a mixed corpus impossible rather than
    merely visible.

    **Identity is a hash, not a multi-column UNIQUE.** A `UNIQUE` across these columns would *not*
    enforce one-row-per-definition, because Postgres treats NULLs as distinct — two rows with
    `year_min IS NULL` would both be admitted. So the primary key is a sha256 over the canonical JSON
    of the recorded fields, which sidesteps NULL semantics entirely and gives each definition a stable
    id for that future `corpus_config_id` to point at.

    **Every `IngestConfig` field is recorded except the API key**, which is a credential and does not
    belong in a table meant to be read. The consequence is accepted rather than overlooked: only
    `query`/`year_*`/`max_records` determine which papers land, so **tuning `batch_size` writes a new
    row.** The table is therefore a full audit of what the library was invoked with, and a reader tells
    a throughput change from a corpus change by looking at the four fields that matter.
    """

    __tablename__ = "corpus_config"

    # sha256 hex of the canonical JSON of every column below except `created_at`. Written by the
    # library when it is asked to run under a configuration, with `on_conflict_do_nothing` — so the
    # first run under a definition dates it and re-runs are silent.
    config_hash: Mapped[str] = mapped_column(String(64), primary_key=True)

    # --- the corpus definition: these four decide which papers land ---------------------
    query: Mapped[str] = mapped_column(Text)
    year_min: Mapped[int | None] = mapped_column(Integer)
    year_max: Mapped[int | None] = mapped_column(Integer)
    max_records: Mapped[int | None] = mapped_column(Integer)

    # --- throughput and attribution: recorded for audit, not part of the definition -----
    batch_size: Mapped[int] = mapped_column(Integer)
    pmc_batch_size: Mapped[int] = mapped_column(Integer)
    ncbi_email: Mapped[str] = mapped_column(Text)
    ncbi_tool: Mapped[str] = mapped_column(Text)
    rate_limit: Mapped[float] = mapped_column(Float)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
