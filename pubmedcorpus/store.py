"""Persisting parsed records. Upsert, so re-running is always safe.

Sync writes `abstract` rows and nothing else. A `paper` row is created later, by
`pmc-fetch`, when PMC actually returns a body — see `pmc.py`.
"""

import dataclasses
import logging
from datetime import date

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from pubmedcorpus import citation
from pubmedcorpus.config import IngestConfig
from pubmedcorpus.models import Abstract, CorpusConfig, Paper
from pubmedcorpus.parse import ParsedPaper, parse_one

log = logging.getLogger(__name__)

# Columns refreshed on re-ingest. `fetch_failed` is deliberately absent: it is owned
# by the pmc-fetch job, and a PubMed re-sync must not reset what PMC already told us.
_ABSTRACT_UPDATE_COLS = (
    "pubmed_xml", "pubmed_json", "entry_date", "pub_year", "pmc_id", "is_retracted",
)


def record_corpus_config(session: Session, config: IngestConfig) -> str:
    """Note that a sync ran under this definition. Returns the `config_hash`.

    **Recording, not enforcement** — it does not compare against the last row or refuse a changed
    definition. See `CorpusConfig`'s docstring for the honest accounting of what that leaves open (a
    mixed corpus stays possible, just no longer invisible) and what would close it.

    `on_conflict_do_nothing` is what makes `created_at` mean *first seen under this definition*: the
    first sync dates it, later ones are silent. Both halves of the row come from `config.recorded()`, so
    the stored columns and the hash that keys them cannot disagree.

    Called from `sync` alone. The PMC fetch also runs under a config, but `sync` is what decides which
    papers exist, and recording in both would date a definition to whichever happened to run first.
    """
    recorded = config.recorded()
    config_hash = config.config_hash()
    session.execute(
        insert(CorpusConfig)
        .values(config_hash=config_hash, **recorded)
        .on_conflict_do_nothing(index_elements=["config_hash"])
    )
    log.info("corpus definition %s: %s", config_hash[:12], recorded["query"])
    return config_hash


def _blank_to_none(v: str | None) -> str | None:
    """Empty or whitespace-only text → NULL, so `IS NOT NULL` means "has content".

    The PubMed parser already returns None for empties, but enforcing it here — the
    single write point for every ingest source — guarantees the invariant every
    `isnot(None)` check relies on, now and for any future source. Content is preserved
    verbatim; only the blank case changes.
    """
    return None if v is None or not v.strip() else v


def as_pubmed_json(parsed: ParsedPaper) -> dict:
    """The parsed PubMed record as a JSON-serialisable dict — everything but the XML.

    The XML is left out because it is already stored verbatim in its own column, and
    nesting it here would be a second copy of the largest field in the corpus.

    Everything else goes in whole. This is what lets the schema carry so few real
    columns: the ordered author list (with ORCID, affiliation and blocking key),
    mesh_terms, the abstract, and every bibliographic locator live here and are
    queryable, so each earns a column only if something actually filters or sorts on it.
    """
    row = dataclasses.asdict(parsed)
    row.pop("raw_xml", None)
    # JSONB has no date type — same rule the corpus snapshot follows.
    for key in ("pub_date", "entry_date"):
        value = row.get(key)
        if isinstance(value, date):
            row[key] = value.isoformat()
    return row


def abstract_values(parsed: ParsedPaper) -> dict:
    """Column values for one `abstract` row. Split out from the upsert so it is
    testable without a database — `pubmed_json` fails silently if it regresses (a
    non-serialisable field only errors against a live connection)."""
    return {
        "pmid": parsed.pmid,
        "pubmed_xml": parsed.raw_xml,
        "pubmed_json": as_pubmed_json(parsed),
        "entry_date": parsed.entry_date,
        "pub_year": parsed.pub_year,
        "pmc_id": _blank_to_none(parsed.pmc_id),
        "is_retracted": parsed.is_retracted,
    }


def paper_values(parsed: ParsedPaper) -> dict:
    """The derived half of a `paper` row: the frozen reference.

    Separate from the JATS half because `pmc-fetch` writes them together but a re-sync
    refreshes only this one. Note the citation keeps its full locator
    (`2013;12(10):966-977`) even though volume/issue/pages are not columns anywhere —
    it is rendered here, from the parse, and stored as text.
    """
    return {"citation": citation.format_reference(parsed, parsed.authors)}


def upsert_abstract(session: Session, parsed: ParsedPaper) -> None:
    """Write one PubMed record, and refresh a promoted paper's citation if it has one."""
    stmt = insert(Abstract).values(**abstract_values(parsed))
    stmt = stmt.on_conflict_do_update(
        index_elements=[Abstract.pmid],
        set_={c: getattr(stmt.excluded, c) for c in _ABSTRACT_UPDATE_COLS},
    )
    session.execute(stmt)

    # A promoted paper's citation was rendered from the XML we may have just replaced.
    # Refresh it so re-sync keeps updating metadata the way it did when the record and
    # the full text shared one table. `jats_xml` and `fulltext_jats` are never touched
    # here — they are PMC's, and sync has nothing to say about them.
    session.execute(
        update(Paper).where(Paper.pmid == parsed.pmid).values(**paper_values(parsed))
    )


def upsert_batch(session: Session, papers: list[ParsedPaper]) -> int:
    for p in papers:
        upsert_abstract(session, p)
    return len(papers)


def reparse_bibliographic(session: Session, batch_size: int = 200) -> int:
    """Rebuild the derived columns from stored `pubmed_xml`. No PubMed round-trip.

    This is what makes it safe to store derivations of the source: change the parser or
    the citation format, run this, and the corpus catches up offline. It rewrites
    `abstract.pubmed_json` (plus the columns cut from the parse) and `paper.citation`,
    and touches nothing else — PMC full text and extractions are untouched, so it is
    safe on a live corpus. Returns the number of records updated.
    """
    pmids = [row[0] for row in session.execute(select(Abstract.pmid)).all()]
    n = 0
    for i, pmid in enumerate(pmids, start=1):
        raw = session.execute(
            select(Abstract.pubmed_xml).where(Abstract.pmid == pmid)
        ).scalar_one()
        parsed = parse_one(raw)
        if parsed is None:
            continue
        values = abstract_values(parsed)
        values.pop("pmid")
        values.pop("pubmed_xml")  # unchanged by definition — it is what we just parsed
        session.execute(update(Abstract).where(Abstract.pmid == pmid).values(**values))
        session.execute(
            update(Paper).where(Paper.pmid == pmid).values(**paper_values(parsed))
        )
        n += 1
        if i % batch_size == 0:
            session.commit()
            log.info("reparse: %d/%d", i, len(pmids))
    session.commit()
    return n
