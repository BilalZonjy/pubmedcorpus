"""E-utilities access.

Everything goes through Bio.Entrez rather than hand-rolled HTTP: it already handles the history server,
the rate limit (3/sec, or 10/sec with an API key), and the URL construction.

**`NCBIClient` holds an `IngestConfig` so nothing here reads global configuration.** It is the one
object a caller constructs; `sync` and `fetch_fulltext` take the client and read `client.config` rather
than accepting their own copy of the configuration, so a run cannot end up with a client built from one
definition and a batch size from another.

**Two pieces of state stay module-level, for opposite reasons — neither is an oversight.**

*The throttle* (`_rate_lock`, `_last_call`) is global because **NCBI's limit is global**: it applies per
caller, not per object. Two clients each holding their own lock would each observe 1/sec and together
exceed it. Sharing the gate is the correct behaviour, and the lock is held across the sleep on purpose
so callers serialise rather than each sleeping independently and then firing at once.

*Biopython's identity* (`Entrez.email`, `Entrez.tool`, `Entrez.api_key`) is global because Biopython
made it module state, so `_configure` cannot be per-instance however much it looks like it should be.
**The consequence, documented rather than worked around: two `NCBIClient`s with different credentials
in one process overwrite each other's identity, last call wins.** Harmless when a process uses one
configuration — which is the documented pattern — and not worth wrapping Biopython to avoid.
"""

import logging
import threading
import time
from dataclasses import dataclass

from Bio import Entrez
from pubmedcorpus.config import IngestConfig
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

log = logging.getLogger(__name__)

BASE_DB = "pubmed"
PMC_DB = "pmc"

# --- Rate limiting ----------------------------------------------------------
# Biopython already enforces NCBI's ceiling (3/sec, or 10/sec with a key). This gate sits on top and is
# the binding constraint whenever `config.rate_limit` is set lower, which it is by default (1/sec).
# Setting it higher than NCBI's ceiling has no effect — Biopython's own limiter still applies underneath.
#
# Module-level, deliberately: see the note in the module docstring on why per-instance locks would be
# wrong rather than merely different.
_rate_lock = threading.Lock()
_last_call: float = 0.0


def _throttle(rate: float) -> None:
    """Wait until the next request is allowed. `rate` comes from the calling client's config."""
    global _last_call
    if rate <= 0:
        return
    interval = 1.0 / rate
    with _rate_lock:
        now = time.monotonic()
        wait = _last_call + interval - now
        if wait > 0:
            log.debug("Rate limit: sleeping %.2fs", wait)
            time.sleep(wait)
            now = time.monotonic()
        _last_call = now


@dataclass
class SearchResult:
    count: int
    webenv: str
    query_key: str
    query: str


def strip_pmc_prefix(pmcid: str) -> str:
    """efetch wants the bare accession number; we store the "PMC" form.

    Module-level rather than a method: pure string handling, nothing about it depends on a config.
    """
    return pmcid[3:] if pmcid.upper().startswith("PMC") else pmcid


_retry = retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    retry=retry_if_exception_type((OSError, IOError)),
    reraise=True,
)


class NCBIClient:
    """E-utilities calls under one `IngestConfig`.

    Construct one per process and pass it down. Every method calls `_configure` then `_throttle` before
    reaching the network, exactly as the free functions did — the ordering matters, because Biopython
    reads its module globals at call time.
    """

    def __init__(self, config: IngestConfig) -> None:
        self.config = config

    # --- setup ------------------------------------------------------------------------

    def _configure(self) -> None:
        """Push this client's identity into Biopython's module globals.

        Called before every request rather than once at construction, because another client in the same
        process may have overwritten them since — see the module docstring. Cheap: three assignments.
        """
        Entrez.email = self.config.ncbi_email
        Entrez.tool = self.config.ncbi_tool
        if self.config.ncbi_api_key:
            Entrez.api_key = self.config.ncbi_api_key
        ceiling = 10.0 if self.config.ncbi_api_key else 3.0
        if self.config.rate_limit > ceiling:
            log.warning(
                "rate_limit=%.1f/sec exceeds NCBI's ceiling of %.0f/sec — "
                "Biopython will throttle to the ceiling regardless",
                self.config.rate_limit,
                ceiling,
            )

    def _before_request(self) -> None:
        self._configure()
        _throttle(self.config.rate_limit)

    # --- query composition ------------------------------------------------------------

    def build_query(self, year_min: int | None = None, year_max: int | None = None) -> str:
        """Compose the search term, applying the year filter as a term filter.

        The base query is parenthesised deliberately. PubMed evaluates booleans strictly left to right
        rather than binding AND tighter than OR, so `A OR B AND C` silently means `(A OR B) AND C` here
        — correct by accident. Parentheses make it correct on purpose.

        Year scoping uses [dp] (publication date). Incremental syncs use edat (entry date) instead:
        scoping a full pull by edat would drop old papers that were indexed into PubMed recently.

        The bounds default to the config's own, so a caller that has already stated them once in its
        corpus definition does not restate them here. Passing them explicitly overrides, which is what
        a one-off narrower pull needs.
        """
        lo = year_min if year_min is not None else self.config.year_min
        hi = year_max if year_max is not None else self.config.year_max

        term = f"({self.config.query})"
        if lo is not None or hi is not None:
            term += f" AND {lo or 1800}:{hi or 3000}[dp]"
        return term

    # --- requests ---------------------------------------------------------------------

    @_retry
    def search_history(self, term: str) -> SearchResult:
        """esearch with usehistory=y — results stay server-side for paged efetch."""
        self._before_request()
        with Entrez.esearch(db=BASE_DB, term=term, usehistory="y", retmax=0) as handle:
            result = Entrez.read(handle)
        return SearchResult(
            count=int(result["Count"]),
            webenv=result["WebEnv"],
            query_key=result["QueryKey"],
            query=term,
        )

    @_retry
    def count(self, term: str) -> int:
        self._before_request()
        with Entrez.esearch(db=BASE_DB, term=term, retmax=0) as handle:
            return int(Entrez.read(handle)["Count"])

    @_retry
    def fetch_batch(self, webenv: str, query_key: str, retstart: int, retmax: int) -> bytes:
        """efetch a page off the history server. Returns raw XML."""
        self._before_request()
        with Entrez.efetch(
            db=BASE_DB,
            query_key=query_key,
            WebEnv=webenv,
            retstart=retstart,
            retmax=retmax,
            retmode="xml",
        ) as handle:
            return handle.read()

    @_retry
    def fetch_pmids(self, pmids: list[str]) -> bytes:
        """efetch specific PMIDs. Used for fixtures and targeted re-fetch."""
        self._before_request()
        with Entrez.efetch(db=BASE_DB, id=",".join(pmids), retmode="xml") as handle:
            return handle.read()

    @_retry
    def fetch_pmc(self, pmcids: list[str]) -> bytes:
        """efetch JATS full text from PMC. Returns raw XML.

        Only the open-access subset has a retrievable body. For anything else PMC returns front matter
        alone (or an error element) rather than failing, so "not open access" is detected while parsing,
        not here.
        """
        self._before_request()
        ids = ",".join(strip_pmc_prefix(p) for p in pmcids)
        with Entrez.efetch(db=PMC_DB, id=ids, retmode="xml") as handle:
            return handle.read()

    @_retry
    def elink_pmcid(self, pmid: str) -> str | None:
        """PMID → PMCID via elink, one record at a time.

        A fallback only. PubMed's own ArticleIdList already carries the PMCID for essentially every
        record that has one, and `parse.py` stores it — so the normal path costs no request at all.
        elink merges results when given many ids at once, which loses the per-PMID mapping, hence one
        call per record; that is why this is opt-in rather than the default.
        """
        self._before_request()
        with Entrez.elink(dbfrom=BASE_DB, db=PMC_DB, id=pmid, linkname="pubmed_pmc") as handle:
            result = Entrez.read(handle)

        for linkset in result or []:
            for db in linkset.get("LinkSetDb", []):
                for link in db.get("Link", []):
                    value = str(link.get("Id", "")).strip()
                    if value:
                        return f"PMC{value}"
        return None
