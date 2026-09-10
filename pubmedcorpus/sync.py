"""The one ingestion job: pull whatever is new, add it, touch nothing else.

"New" is relative to what the database already holds. On an empty database
everything is new, so the first run *is* the full-corpus pull — there is no
separate backfill command. After a clean first pull, runs are scoped by PubMed
entry date (edat), fetching records indexed since the newest `entry_date` we
already hold *minus an overlap window* (upserts are idempotent, and re-seeing a
record is much cheaper than missing one).

**The cursor is `max(abstract.entry_date)`, not a run log.** There is no
`ingest_run` table — history lives in the application log, which already
carries start, progress, outcome and error for every run. Deriving the cursor
from the data means it can never drift from what was actually stored, and a
restored snapshot resumes correctly for free: the restored records carry their
own entry dates, so the very next sync reopens from the right place with no
marker to write.

**The corresponding cost: a crashed run no longer self-heals.** The old design
kept a `status='failed'` row that marked the corpus incomplete, so the next run
went full and recovered on its own. A crash now leaves a partial corpus whose
`max(entry_date)` looks perfectly healthy, and the next run resumes from it —
silently never re-fetching whatever the crash missed earlier in its window.
Recovery is a human decision: check the log for a failure, then run with
`--full`. Acceptable here because a full pull is minutes (~2,500 records, one
history-server search plus a dozen paged efetch calls), but it will not
announce itself — say so before it needs saying.
"""

import logging
from datetime import date, timedelta

from sqlalchemy import func, select

from pubmedcorpus.models import Abstract
from pubmedcorpus.ncbi import NCBIClient
from pubmedcorpus.parse import parse_batch
from pubmedcorpus.store import record_corpus_config, upsert_batch
from sqlalchemy.orm import Session

log = logging.getLogger(__name__)


def _cursor(session) -> date | None:
    """The newest entry date we hold, or None on an empty corpus.

    Records whose entry date failed to parse are NULL and `max()` skips them, so
    a corpus of nothing-but-unparseable-dates reads as empty and the next sync
    goes full — the safe direction, not a bug.
    """
    newest = session.execute(select(func.max(Abstract.entry_date))).scalar()
    return newest


def sync(
    session: Session,
    client: NCBIClient,
    since: date | None = None,
    days_back: int = 3,
    batch_size: int | None = None,
    full: bool = False,
) -> int:
    """Pull whatever is new and upsert it.

    The mode is chosen, not commanded: `full=True` forces a full-corpus pull
    (and ignores `since`), an explicit `since` scopes by entry date from that
    day, and otherwise the run goes full if the corpus is empty, incremental
    from `max(entry_date) - days_back` if it isn't.

    **The corpus definition arrives as one object, `client.config`** — the query, the year bounds and
    the record cap. It used to be three separate arguments here (and before that, read-only properties
    derived from a dev/prod switch). Both are gone: a caller stating the corpus twice is exactly the
    drift `corpus_config` exists to detect, so there is now one statement of it per run and this
    function reads rather than accepts it. A CLI flag narrowing a single run does so by constructing a
    narrower config, which is then what gets recorded.

    **The year bounds reach the full branch only**, and that is not an oversight: they become a `[dp]`
    publication-date filter, and applying one to an incremental `[edat]` pull would re-exclude old
    papers that were indexed recently — which is the whole reason the incremental branch scopes by
    entry date. A config carrying year bounds on an incremental run is therefore accepted and ignored.

    **The caller owns the session's lifetime, not its transaction.** This commits after every batch, and
    the recovery story documented above depends on it: a crashed run leaves the batches it finished
    durably stored. So wrapping this call in an outer transaction and rolling back will not undo it.
    """
    config = client.config
    batch_size = batch_size or config.batch_size

    # Before anything is fetched, so a run that dies mid-pull still leaves a record of what it was
    # trying to build. `on_conflict_do_nothing`, so re-running under the same definition is silent.
    record_corpus_config(session, config)

    if not full and since is None:
        newest = _cursor(session)
        if newest is not None:
            since = newest - timedelta(days=days_back)
        else:
            full = True

    if full:
        # The full corpus definition, year bounds included. Uses [dp] (publication date) — edat
        # here would drop old papers that were indexed into PubMed recently.
        term = client.build_query()
        log.info("Sync (full corpus): %s", term)
    else:
        # **Composed by hand rather than through `build_query`, and that is the point of this
        # branch**: it scopes by [edat] instead of [dp]. It used to read a global query here while
        # the branch above went through `build_query`, so parameterising `build_query` alone would
        # have left this one still reading configuration — the reason threading the query had to
        # reach both branches rather than just the composer.
        term = (
            f"({config.query}) AND "
            f"{since:%Y/%m/%d}:{date.today():%Y/%m/%d}[edat]"
        )
        log.info("Sync (new since %s): %s", since, term)

    processed = 0
    try:
        search = client.search_history(term)
        log.info("Found %s records", search.count)

        total = search.count
        # Cap full pulls only. Capping an incremental run would silently
        # drop the newest records and then scope past them forever.
        if full and config.max_records is not None:
            total = min(total, config.max_records)
            log.info("Capped to %s records (--max-records)", total)

        offset = 0
        while offset < total:
            take = min(batch_size, total - offset)
            log.info("Fetching %s-%s of %s", offset, offset + take, total)
            xml = client.fetch_batch(search.webenv, search.query_key, offset, take)
            papers = parse_batch(xml)
            upsert_batch(session, papers)
            offset += take
            processed += len(papers)
            session.commit()
    except Exception:
        log.exception("Sync failed after %s records — the corpus is now partial. "
                       "max(entry_date) will look current, so the next run resumes "
                       "as if this one succeeded. Run with --full to recover.", processed)
        raise

    log.info("Sync complete: %s records", processed)
    return processed
