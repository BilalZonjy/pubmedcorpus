"""Sync's resume cursor and the term each mode composes. Offline — a fake session stands in for the DB.

There is no run-log table. The cursor is `max(abstract.entry_date)`, derived from
whatever the corpus actually holds — a restored snapshot resumes correctly for free,
because its records carry their own entry dates. This is what used to be protected by
a synthetic "import" run row; the guarantee now lives in the data.

**The query tests stopped stubbing `build_query` and now exercise it.** They used to monkeypatch it to
return a marker string, plus fake `settings.pubmed_query` as a property, because both were module-level
globals. `_FakeClient` subclasses the real `NCBIClient` and overrides only `search_history`, so the term
these tests assert on is the one the real composer produced from a real `IngestConfig` — the same change
that removed the globals made the tests able to check more.
"""

from datetime import date, timedelta
from types import SimpleNamespace

import pytest
from pubmedcorpus import sync as sync_mod
from pubmedcorpus.config import IngestConfig
from pubmedcorpus.ncbi import NCBIClient
from pubmedcorpus.sync import _cursor


class _FakeResult:
    def __init__(self, value):
        self._value = value

    def scalar(self):
        return self._value


class _FakeSession:
    """Returns a canned scalar for any statement.

    `_cursor` issues one; `record_corpus_config` issues an INSERT whose result it ignores.
    """

    def __init__(self, value):
        self._value = value

    def execute(self, _stmt):
        return _FakeResult(self._value)

    def commit(self):
        pass


class TestCursor:
    def test_reads_the_newest_entry_date(self):
        assert _cursor(_FakeSession(date(2025, 9, 20))) == date(2025, 9, 20)

    def test_none_on_an_empty_corpus(self):
        # Sync reads this as "go full" — the safe direction on a fresh database, and
        # indistinguishable from a corpus whose every entry_date failed to parse.
        assert _cursor(_FakeSession(None)) is None


def config(query: str = "configured", **kw) -> IngestConfig:
    return IngestConfig(query=query, ncbi_email="who@example.org", ncbi_tool="test", **kw)


class _FakeClient(NCBIClient):
    """The real client with the network removed.

    Only `search_history` is overridden — the first thing the composed term reaches — so recording it
    there and returning a zero count means the fetch loop never runs and no batch is ever parsed or
    upserted. `build_query` is the genuine implementation.
    """

    def __init__(self, cfg: IngestConfig, seen: dict):
        super().__init__(cfg)
        self.seen = seen

    def search_history(self, term):
        self.seen["term"] = term
        return SimpleNamespace(count=0, webenv=None, query_key=None)


@pytest.fixture
def run():
    """Run `sync` far enough to compose its term, then stop. Returns (call, seen).

    **No `session_scope` to monkeypatch any more.** `sync` takes the session, so the fake is simply
    passed in — the fixture stopped needing to patch a module attribute to intercept a scope it opened
    for itself.
    """
    seen: dict = {}

    def call(cfg: IngestConfig | None = None, newest: date | None = None, **kwargs):
        return sync_mod.sync(_FakeSession(newest), _FakeClient(cfg or config(), seen), **kwargs)

    return call, seen


class TestSyncQuery:
    """Which term each mode composes: `[dp]` scoping on a full pull, `[edat]` on an incremental one."""

    def test_a_full_pull_composes_from_the_configs_query(self, run):
        call, seen = run
        call(full=True)
        assert seen["term"] == "(configured)"

    def test_an_incremental_pull_also_uses_the_configs_query(self, run):
        """**The blocker this pins, now fixed.**

        The incremental branch bypasses `build_query` and composes its own string, because it scopes by
        `[edat]` rather than `[dp]`. It used to read `settings.pubmed_query` there, so parameterising
        `build_query` alone would have left this branch still reading a global — which is why the query
        had to be threaded into both branches rather than just the composer.
        """
        call, seen = run
        call(since=date(2025, 1, 1))
        assert seen["term"].startswith("(configured) AND 2025/01/01:")

    def test_the_incremental_branch_scopes_by_edat(self, run):
        call, seen = run
        call(since=date(2025, 1, 1))
        assert "[edat]" in seen["term"]

    def test_neither_branch_scopes_by_year_unless_the_config_says_so(self, run):
        """**The asymmetry this test was written to pin is gone, and the fix was a deletion.**

        The full branch used to apply `settings.ingest_year_min` as a `[dp]` filter while the
        incremental branch applied none, so the two modes described different corpora. Both derived
        from `IS_DEV_ENVIRONMENT`; removing that switch removed the difference, because neither branch
        has a configured fallback to disagree about any more.
        """
        call, seen = run
        call(full=True)
        assert "[dp]" not in seen["term"]

        call(since=date(2025, 1, 1))
        assert "[dp]" not in seen["term"]

    def test_year_bounds_reach_the_full_branch_from_the_config(self, run):
        call, seen = run
        call(config(year_min=2018), full=True)
        assert "2018:3000[dp]" in seen["term"]

    def test_year_bounds_are_accepted_and_ignored_on_an_incremental_pull(self, run):
        """Not an oversight — a `[dp]` floor on an `[edat]` pull would re-exclude old papers
        indexed recently, which is the whole reason the incremental branch scopes by entry date."""
        call, seen = run
        call(config(year_min=2018), since=date(2025, 1, 1))
        assert "[dp]" not in seen["term"]

    def test_max_records_caps_a_full_pull(self, run):
        """It was `settings.ingest_max_records`, then an argument, now part of the definition."""
        call, _ = run
        assert call(config(max_records=5), full=True) == 0

    def test_an_empty_corpus_goes_full_even_without_the_flag(self, run):
        """`_cursor` returning None is the "go full" signal — the safe direction on a fresh database."""
        call, seen = run
        assert call() == 0
        assert seen["term"] == "(configured)"

    def test_full_ignores_an_explicit_since(self, run):
        call, seen = run
        call(since=date(2025, 1, 1), full=True)
        assert seen["term"] == "(configured)"

    def test_days_back_widens_the_window_behind_the_cursor(self, run):
        """The overlap window: re-seeing a record is far cheaper than missing one."""
        call, seen = run
        newest = date(2025, 6, 10)
        call(newest=newest, days_back=3)
        expected = (newest - timedelta(days=3)).strftime("%Y/%m/%d")
        assert f"AND {expected}:" in seen["term"]


class TestCorpusConfigRecording:
    def test_a_sync_records_the_definition_it_ran_under(self, run, monkeypatch):
        """Before anything is fetched, so a run that dies mid-pull still says what it was building."""
        recorded = []
        monkeypatch.setattr(sync_mod, "record_corpus_config",
                            lambda _session, cfg: recorded.append(cfg))
        call, _ = run
        cfg = config(query="recorded OR terms")
        call(cfg, full=True)
        assert recorded == [cfg]
