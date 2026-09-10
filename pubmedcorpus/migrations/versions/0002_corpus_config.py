"""corpus_config: what this database was built from

Revision ID: pubmedcorpus_0002
Revises: pubmedcorpus_0001
Create Date: 2026-09-08

**The only revision on this branch that does work on an existing database** — `0001` detects the two
tables it would create and returns.

The gap this fills is a real bug rather than bookkeeping: `sync`'s resume cursor is
`max(abstract.entry_date)` over the whole table, with no awareness of the query that produced those
rows. Change the query and the next incremental sync resumes from the old corpus's high-water mark
under the new definition — the corpus silently becomes a mixture. This makes that visible.

**Recording, not enforcement.** No `corpus_config_id` on `abstract`, and nothing refuses a sync whose
config differs from the last row. See `pubmedcorpus.models.CorpusConfig` for the honest accounting of
what that does and does not buy, and what would close it.

**The primary key is a hash, and that is not a stylistic choice.** A `UNIQUE` across these columns
would not enforce one-row-per-definition, because Postgres treats NULLs as distinct — two rows with
`year_min IS NULL` would both be admitted, which is the common case. A sha256 over the canonical JSON
of the recorded fields sidesteps NULL semantics entirely.

Nothing writes rows yet; the library starts recording when it is given an `IngestConfig` to hash.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "pubmedcorpus_0002"
down_revision: Union[str, None] = "pubmedcorpus_0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "corpus_config",
        # sha256 hex of the canonical JSON of every column below except `created_at`. Inserted with
        # `on_conflict_do_nothing`, so the first run under a definition dates it and re-runs are
        # silent — which is what makes `created_at` mean "first seen".
        sa.Column("config_hash", sa.String(64), primary_key=True),
        # The corpus definition proper: these four decide which papers land.
        sa.Column("query", sa.Text(), nullable=False),
        sa.Column("year_min", sa.Integer()),
        sa.Column("year_max", sa.Integer()),
        sa.Column("max_records", sa.Integer()),
        # Throughput and attribution. Recorded for a full audit of what the library was invoked with,
        # accepting the consequence: tuning `batch_size` writes a new row. A reader tells a throughput
        # change from a corpus change by looking at the four fields above.
        sa.Column("batch_size", sa.Integer(), nullable=False),
        sa.Column("pmc_batch_size", sa.Integer(), nullable=False),
        sa.Column("ncbi_email", sa.Text(), nullable=False),
        sa.Column("ncbi_tool", sa.Text(), nullable=False),
        sa.Column("rate_limit", sa.Float(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
    )
    # No index beyond the primary key. The table gets one row per distinct configuration — single
    # digits over a corpus's life — so any query over it is a sequential scan of a handful of rows.


def downgrade() -> None:
    op.drop_table("corpus_config")
