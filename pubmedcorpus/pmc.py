"""PMC full-text retrieval.

Reviews, editorials and commentary — where researchers actually *state* beliefs
— often have thin or absent abstracts, so full text is not optional for this
project. Only PMC's open-access subset is retrievable this way; the rest is
recorded as such and left alone. Publisher sites are never scraped.

Separate from `sync` on purpose. Full text is a different resource with a
different failure mode (most records simply don't have it), and a PubMed
re-sync must never wipe text that was already fetched — hence `store.py`
excludes these columns from its upsert.

**This job is what creates `paper` rows.** An `abstract` row is a record we know of; a
`paper` row is one we can read, so a successful fetch *promotes* the record: parse its
stored PubMed XML for the citation, store the article beside it. A failure sets
`abstract.fetch_failed` and writes no paper.
"""

import logging

from sqlalchemy import and_, exists, nulls_last, or_, select
from sqlalchemy.dialects.postgresql import insert

from pubmedcorpus.jats import parse_pmc_batch
from pubmedcorpus.models import Abstract, Paper
from pubmedcorpus.ncbi import NCBIClient
from pubmedcorpus.parse import parse_one
from pubmedcorpus.store import paper_values
from sqlalchemy.orm import Session

log = logging.getLogger(__name__)

# Labels for the run summary only — nothing stores them. The database records an
# outcome in two bits: whether a `paper` row appeared, and `abstract.fetch_failed`.
# The finer taxonomy these once formed is recoverable where it matters —
# `fetch_failed AND pmc_id IS NULL` is the old `no_pmcid`, and `fetch_failed AND
# pmc_id IS NOT NULL` is the old `not_oa`, which is exactly the `--recheck` set
# (it had a PMC entry but no open body, so an embargo may since have lifted).
OUTCOME_OK = "ok"
OUTCOME_NO_PMCID = "no_pmcid"
OUTCOME_NOT_OA = "not_oa"
OUTCOME_FAILED = "failed"


_HAS_PAPER = exists().where(Paper.pmid == Abstract.pmid)


def _candidates(session, pmids: list[str] | None, retry: bool, recheck: bool, limit: int | None):
    """Records still owed a full-text attempt.

    Anything without a `paper` row and without a recorded failure always qualifies;
    failures are opt-in, so a routine re-run doesn't re-request thousands of records
    PMC has already said no to.

    An explicit `--pmid` bypasses all of it, including the already-promoted check —
    naming a paper by hand is how you deliberately re-fetch one.
    """
    stmt = select(Abstract)
    if pmids:
        stmt = stmt.where(Abstract.pmid.in_(pmids))
    else:
        wanted = [Abstract.fetch_failed.is_(False)]
        if retry:
            wanted.append(Abstract.fetch_failed.is_(True))
        if recheck:
            # Had a PMC entry but no open body — an embargo may have lifted since.
            # Records with no PMCID at all are never worth re-asking about.
            wanted.append(and_(Abstract.fetch_failed.is_(True), Abstract.pmc_id.isnot(None)))
        # Never re-request something we already hold: a `paper` row is the outcome.
        stmt = stmt.where(~_HAS_PAPER).where(or_(*wanted))

    # Newest first: recent records have far better OA coverage, so a capped dev
    # run actually exercises the parser instead of collecting `no_pmcid`.
    stmt = stmt.order_by(nulls_last(Abstract.pub_year.desc()), Abstract.pmid)
    if limit is not None:
        stmt = stmt.limit(limit)
    return list(session.execute(stmt).scalars())


def _apply(session, record: Abstract, parsed) -> str:
    """Promote one record to a `paper`, or mark the attempt failed.

    PMC returns front matter with no body for anything outside the open-access subset,
    so "not open access" arrives here as a parse outcome rather than an error.

    Promotion re-parses the record's own stored PubMed XML: the citation is rendered
    now and frozen, which is what lets `paper` carry no bibliographic columns.
    """
    if not parsed.has_body:
        record.fetch_failed = True
        return OUTCOME_NOT_OA

    payload = parsed.as_dict()
    # PMC occasionally omits permissions. The licence now rides inside this payload,
    # which a refetch replaces wholesale, so carry forward what we already recorded —
    # blanking it would lose the only record of what reuse is permitted.
    if not payload.get("license") and record.paper is not None:
        payload["license"] = (record.paper.fulltext_jats or {}).get("license")

    values = {
        "pmid": record.pmid,
        "jats_xml": parsed.raw_xml,
        "fulltext_jats": payload,
    }
    pubmed = parse_one(record.pubmed_xml)
    if pubmed is None:
        # The XML we stored at sync no longer parses — a real defect, not a PMC
        # outcome. Fail the attempt rather than writing a paper with no citation.
        log.warning("%s: stored pubmed_xml did not parse; not promoting", record.pmid)
        record.fetch_failed = True
        return OUTCOME_FAILED
    values.update(paper_values(pubmed))

    stmt = insert(Paper).values(**values)
    stmt = stmt.on_conflict_do_update(
        index_elements=[Paper.pmid],
        set_={c: getattr(stmt.excluded, c) for c in ("jats_xml", "fulltext_jats", "citation")},
    )
    session.execute(stmt)
    record.fetch_failed = False
    return OUTCOME_OK


def fetch_fulltext(
    session: Session,
    client: NCBIClient,
    pmids: list[str] | None = None,
    limit: int | None = None,
    batch_size: int | None = None,
    retry: bool = False,
    recheck: bool = False,
    relink: bool = False,
) -> dict[str, int]:
    """Fetch JATS full text for papers that don't have it yet.

    Idempotent and resumable: each paper's outcome is committed with its batch,
    so an interrupted run resumes simply by being run again — the rows it
    already settled no longer qualify as candidates.

    **Writes no `corpus_config` row**, unlike `sync`. This job reads whatever `abstract` already holds
    and asks PMC for bodies; it does not decide which papers are in the corpus, so recording a
    definition here would date it to a fetch rather than to the sync that defined it.

    **The caller owns the session's lifetime, not its transaction.** The resumability above *is* the
    per-batch commit, so wrapping this call in an outer transaction and rolling back will not undo the
    batches it finished.
    """
    batch_size = batch_size or client.config.pmc_batch_size
    # `limit` caps papers attempted this run and stays a plain argument rather than moving into the
    # config: it is a resumability knob, not part of the corpus definition — the papers it defers are
    # picked up by the next run. `config.max_records`, by contrast, changes which papers exist at all.

    counts: dict[str, int] = {}

    def tally(status: str) -> None:
        counts[status] = counts.get(status, 0) + 1

    records = _candidates(session, pmids, retry, recheck, limit)
    log.info("%s records to attempt", len(records))
    if not records:
        return counts

    processed = 0
    try:
        # Resolve PMCIDs first. PubMed's own ArticleIdList already carries
        # one for essentially every record that has a PMC entry, so this
        # normally costs no requests at all.
        pending: list[Abstract] = []
        for record in records:
            if not record.pmc_id and relink:
                record.pmc_id = client.elink_pmcid(record.pmid)
            if record.pmc_id:
                pending.append(record)
            else:
                record.fetch_failed = True
                tally(OUTCOME_NO_PMCID)
        session.commit()

        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            log.info("Fetching %s-%s of %s", start, start + len(batch), len(pending))

            try:
                xml = client.fetch_pmc([p.pmc_id for p in batch])
                articles = parse_pmc_batch(xml)
            except Exception as exc:
                # One bad batch shouldn't end the run — mark and continue,
                # since `--retry` exists precisely for these.
                log.warning("Batch failed, marking retryable: %s", exc)
                for record in batch:
                    record.fetch_failed = True
                    tally(OUTCOME_FAILED)
                session.commit()
                continue

            # PMC returns articles in whatever order it likes, and may omit
            # ones it won't serve, so match on identity rather than position.
            # Case-normalised: a mismatch here would look exactly like a
            # record PMC declined to serve, and be recorded as not_oa.
            by_pmcid = {a.pmcid.upper(): a for a in articles if a.pmcid}
            by_pmid = {a.pmid: a for a in articles if a.pmid}

            for record in batch:
                parsed = by_pmcid.get((record.pmc_id or "").upper()) or by_pmid.get(record.pmid)
                if parsed is None:
                    # Requested but not returned: PMC has no open copy.
                    record.fetch_failed = True
                    tally(OUTCOME_NOT_OA)
                    continue
                tally(_apply(session, record, parsed))

            processed += len(batch)
            session.commit()

    except Exception:
        log.exception(
            "PMC fetch failed after %s records (outcomes so far: %s)",
            processed, ", ".join(f"{k}={v}" for k, v in sorted(counts.items())),
        )
        session.commit()
        raise

    log.info("PMC fetch complete: %s", ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    return counts
