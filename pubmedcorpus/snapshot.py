"""Corpus snapshot — export and re-import the downloaded data.

The corpus (PubMed records + PMC full text) is the expensive asset: rate-limited to fetch, slow to
rebuild, and dependent on NCBI's goodwill. Everything a consumer derives from it is regenerable
locally. This snapshots the corpus to a portable file so it can outlive any one database, and so
analysis can be redone against a fixed input.

**Library half only.** `export` and `import_` take a session and a stream; they neither open a
database nor parse arguments. The CLI that wraps them belongs to the consuming application, because
deciding which environment variables must be present before touching a database is the
application's call, not the library's — see `sudep/corpus.py` for this project's.

What a snapshot is NOT:

- **Not derived output.** Whatever a consumer computes from the corpus is regenerable and is
  deliberately excluded — a snapshot is the *input* to that work, so it can be wiped and redone
  without re-crawling. Import writes only `abstract` and `paper`; a re-import over a live database
  refreshes records and leaves everything downstream in place.
- **Not a database dump.** That is the whole database, moved between machines. This is corpus-only
  and round-trips either direction.

Format: one JSON object per line (JSONL), one PubMed record, with its full text nested under a
`"paper"` key — **absent when we could not read it**, which is the same signal the schema uses.
Authorship rides inside `pubmed_json` rather than as a nested list, since there is no author table to
restore into. Import is idempotent — it upserts, matching `store.py` — and additive: records absent
from the file are left alone, never deleted. There is no version field; a pre-split snapshot is
detected by shape and rejected (`SnapshotFormatError`) rather than imported with holes in it.

There is no separate "restore marker" to write. A later `sync` on the restored database resumes
incrementally for free: its cursor is `max(abstract.entry_date)` over whatever is actually stored, and
a restore populates exactly that column. See `sync.py` for the cursor itself.
"""

import json
from datetime import date

from pubmedcorpus.models import Abstract, Paper
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session, selectinload

# Content columns only. `ingested_at`/`updated_at` are DB bookkeeping and are
# regenerated on import (you did just re-ingest the row).
#
# `fetch_failed` DOES travel, unlike in store.py's upsert which leaves it to pmc-fetch:
# here the snapshot is the authority, so a restore must reproduce what PMC already told
# us rather than re-asking for thousands of records.
ABSTRACT_COLUMNS = (
    "pmid", "pubmed_xml", "pubmed_json",
    "entry_date", "pub_year", "pmc_id", "is_retracted", "fetch_failed",
)

# The nested block. Present only for records we could read — its absence *is* the
# has-full-text signal on the wire, matching the table it restores into.
#
# `citation` is derived from the PubMed XML but travels rather than being recomputed:
# a restore must reproduce the corpus it was taken from, and recomputing would silently
# apply the *importing* build's parser to the *exported* corpus. `ingest reparse` is
# the deliberate way to refresh it.
PAPER_COLUMNS = ("pmid", "jats_xml", "fulltext_jats", "citation")

_DATE_COLUMNS = frozenset({"entry_date"})


class SnapshotFormatError(Exception):
    """A snapshot this build cannot import."""


def encode_record(record: Abstract) -> dict:
    """One `abstract` row (+ its paper, if any) → a JSON-serialisable dict."""
    row: dict = {}
    for col in ABSTRACT_COLUMNS:
        value = getattr(record, col)
        row[col] = value.isoformat() if isinstance(value, date) else value
    if record.paper is not None:
        row["paper"] = {col: getattr(record.paper, col) for col in PAPER_COLUMNS}
    return row


def decode_record(rec: dict) -> tuple[dict, dict | None]:
    """A snapshot dict → (abstract values, paper values or None).

    Rejects pre-split snapshots rather than letting them through. Reads are otherwise
    tolerant (`rec.get`), so an old file would import with every derived column NULL —
    which the NOT NULL constraints would then reject with a message that says nothing
    about what to do. Fail here instead, where we can name the fix.
    """
    if "raw_xml" in rec or "abstract_text" in rec or "fulltext_status" in rec or "authors" in rec:
        raise SnapshotFormatError(
            f"snapshot predates the abstract/paper split (pmid {rec.get('pmid')!r}): it "
            "carries one flat row per paper instead of a record with its full text nested "
            "under 'paper'. Re-export the corpus from a current build — this file cannot "
            "be upgraded in place."
        )
    values: dict = {}
    for col in ABSTRACT_COLUMNS:
        value = rec.get(col)
        if col in _DATE_COLUMNS and value is not None:
            value = date.fromisoformat(value)
        values[col] = value

    paper = rec.get("paper")
    if paper is None:
        return values, None
    return values, {col: paper.get(col) for col in PAPER_COLUMNS}


def export(session: Session, out) -> int:
    """Write the whole corpus as JSONL. Ordered by pmid so snapshots diff.

    **Streams: `scalars()` is a live cursor, not a materialised list.** So the session must stay open
    for the whole write — closing it before `out` is fully consumed would truncate the snapshot
    silently, producing a file that looks like a smaller corpus rather than a failure. That used to be
    guaranteed by an internal `session_scope`; it is now the caller's to hold, which is the trade
    injection makes and the reason it is worth stating here.
    """
    n = 0
    records = session.execute(
        select(Abstract).options(selectinload(Abstract.paper)).order_by(Abstract.pmid)
    ).scalars()
    for record in records:
        out.write(json.dumps(encode_record(record), ensure_ascii=False))
        out.write("\n")
        n += 1
    return n


def _flush(session, batch: list[dict]) -> None:
    for rec in batch:
        values, paper = decode_record(rec)
        stmt = insert(Abstract).values(**values)
        stmt = stmt.on_conflict_do_update(
            index_elements=[Abstract.pmid],
            set_={c: getattr(stmt.excluded, c) for c in ABSTRACT_COLUMNS if c != "pmid"},
        )
        session.execute(stmt)

        # The paper is written only when the snapshot has one. A record that lost its
        # full text between snapshots is not un-promoted here: import is additive, and
        # deleting a paper would take everything keyed to it along (FK cascade).
        if paper is None:
            continue
        stmt = insert(Paper).values(**paper)
        stmt = stmt.on_conflict_do_update(
            index_elements=[Paper.pmid],
            set_={c: getattr(stmt.excluded, c) for c in PAPER_COLUMNS if c != "pmid"},
        )
        session.execute(stmt)


def import_(session: Session, lines, batch_size: int = 200) -> int:
    """Upsert records (and their full text) from a JSONL stream. Idempotent, additive.

    **The caller owns the session's lifetime, not its transaction.** Committing per batch is what makes
    a half-bad file leave the good batches behind — the CLI's error path documents relying on exactly
    that — so an outer transaction around this call cannot roll it back.
    """
    n = 0
    batch: list[dict] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        batch.append(json.loads(line))
        if len(batch) >= batch_size:
            _flush(session, batch)
            session.commit()
            n += len(batch)
            batch.clear()
    if batch:
        _flush(session, batch)
        session.commit()
        n += len(batch)
    return n
