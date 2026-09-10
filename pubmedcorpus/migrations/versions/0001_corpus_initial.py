"""the corpus schema: abstract, paper

Revision ID: pubmedcorpus_0001
Revises:
Create Date: 2026-09-08

**The library's own migration branch starts here.** `abstract` and `paper` used to be created by the
application's `0001_initial`, and their DDL was *moved* — not copied — so each table still has exactly
one history. The application's `0001` now declares `depends_on` on this revision.

**Why the library ships migrations at all.** An earlier draft had it ship models only, on the reasoning
that the one existing consumer already had these tables migrated. That is true and it is also the whole
problem: a library that cannot create its own schema is not installable by a *second* consumer, which
is the point of extracting it. Django is the precedent — every app owns `migrations/`, cross-app
ordering is declared explicitly, and one shared namespace keeps foreign keys resolvable.

**Everything here is guarded on `has_table("abstract")`, and that guard is the point.** Alembic has no
`--fake-initial`, so it is hand-rolled: on the database this library was extracted from, these tables
already exist and this revision must be a no-op that simply records itself as applied. On an empty
database it does the real work. Both paths end with the same schema, which is what makes the branch
safe to introduce to a live database.

The `search_tsv` column and its GIN index came from the application's `0014_lexical_search`, folded in
here rather than kept as a separate library revision: this branch has no history to preserve yet, so
its first revision is simply the current state of the two tables.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "pubmedcorpus_0001"
down_revision: Union[str, None] = None
# The label the application's `alembic.ini` and its migrations refer to this branch by.
branch_labels: Union[str, Sequence[str], None] = ("pubmedcorpus",)
depends_on: Union[str, Sequence[str], None] = None


# Title (weight A) + abstract (weight B), in `english` and `simple`. Spelled out here and again in
# `pubmedcorpus.models.Abstract.search_tsv`, deliberately: a migration is a frozen historical record
# and must not import from a model that keeps changing. Keep them identical if either moves.
_ABSTRACT_TSV = """
    setweight(to_tsvector('english', coalesce(pubmed_json->>'title', '')), 'A')
 || setweight(to_tsvector('english', coalesce(pubmed_json->>'abstract_text', '')), 'B')
 || setweight(to_tsvector('simple',  coalesce(pubmed_json->>'title', '')), 'A')
 || setweight(to_tsvector('simple',  coalesce(pubmed_json->>'abstract_text', '')), 'B')
"""


def upgrade() -> None:
    # The hand-rolled `--fake-initial`. On the database this library came out of, `abstract` and
    # `paper` were created by the application's `0001` long ago; re-creating them would fail, and
    # skipping the revision entirely would leave the branch unrecorded. Detecting them is what lets
    # one revision serve both an existing database and an empty one.
    if sa.inspect(op.get_bind()).has_table("abstract"):
        return

    # Every PubMed record we know about. `sync` writes here and nowhere else.
    #
    # Very few real columns on purpose: `pubmed_json` is a complete `asdict()` of the
    # parse — title, journal, doi, volume/issue/pages, the ordered author list with
    # ORCID and affiliation, mesh_terms, the abstract — so a field earns a column only
    # when something queries it. Authorship in particular has no table: it is in the
    # JSON, and is rendered once into `paper.citation` at promotion.
    op.create_table(
        "abstract",
        sa.Column("pmid", sa.String(20), primary_key=True),
        sa.Column("pubmed_xml", sa.Text(), nullable=False),
        sa.Column("pubmed_json", postgresql.JSONB(), nullable=False),
        sa.Column("entry_date", sa.Date()),
        sa.Column("pub_year", sa.Integer()),
        sa.Column("pmc_id", sa.String(20)),
        sa.Column("is_retracted", sa.Boolean(), nullable=False, server_default=sa.false()),
        # With the `paper` row, this gives three states and needs no enum: no paper row
        # and not failed = never attempted; failed = asked, no body; paper row = have it.
        sa.Column("fetch_failed", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("ingested_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_abstract_pub_year", "abstract", ["pub_year"])
    op.create_index("ix_abstract_entry_date", "abstract", ["entry_date"])
    op.create_index("ix_abstract_pmc_id", "abstract", ["pmc_id"])
    op.create_index("ix_abstract_is_retracted", "abstract", ["is_retracted"])
    op.create_index("ix_abstract_fetch_failed", "abstract", ["fetch_failed"])

    # A record we could actually read. Created by the PMC fetch, never by sync.
    #
    # No indexes and no status column: nothing filters this table, because *being in
    # it* is the filter every consumer wants. Each column is NOT NULL because a row
    # is only ever written when the fetch and the parse both succeeded — a
    # half-promoted paper should not exist.
    op.create_table(
        "paper",
        sa.Column(
            "pmid",
            sa.String(20),
            sa.ForeignKey("abstract.pmid", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("jats_xml", sa.Text(), nullable=False),
        sa.Column("fulltext_jats", postgresql.JSONB(), nullable=False),
        sa.Column("citation", sa.Text(), nullable=False),
    )

    op.execute(
        f"ALTER TABLE abstract ADD COLUMN search_tsv tsvector "
        f"GENERATED ALWAYS AS ({_ABSTRACT_TSV}) STORED"
    )
    # GIN, not GiST: a corpus is read far more than written (one ingest pass, then queries), and GIN
    # is the faster of the two for lookups at the cost of slower updates.
    op.execute("CREATE INDEX ix_abstract_search_tsv ON abstract USING GIN (search_tsv)")


def downgrade() -> None:
    # Unguarded, unlike `upgrade`. Downgrading this revision means "remove the corpus schema", and on
    # the database this came from that is exactly as destructive as it sounds — every consumer table
    # with `ForeignKey("paper.pmid", ondelete="CASCADE")` loses its rows. The application's branch
    # declares `depends_on` on this revision, so Alembic will refuse to run this while those tables
    # still exist, which is the protection that matters.
    op.execute("DROP INDEX IF EXISTS ix_abstract_search_tsv")
    op.drop_table("paper")
    op.drop_table("abstract")
