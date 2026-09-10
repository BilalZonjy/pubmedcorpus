"""`NCBIClient.build_query` — the PubMed search term, and the only place it is composed.

**Written before the extraction touched it, because it had no tests at all.** The year filter was half
of a full-vs-incremental asymmetry, since resolved by deleting the dev/prod switch both sides derived
from; the query itself was a global these tests now pin as a parameter.

**The mechanism got simpler when the config arrived, which is the point.** Pinning the default base used
to need `monkeypatch.setattr(type(client.settings), "pubmed_query", property(...))` — faking a read-only
property on a settings singleton to observe a global fallback. There is no fallback now: the term is
composed from `config.query`, so the test just builds a config. A test that gets shorter after a
refactor is usually evidence the refactor removed something real.

Offline: composing a term touches no network and no database. `_client` never calls a request method.
"""

from pubmedcorpus.config import IngestConfig
from pubmedcorpus.ncbi import NCBIClient, strip_pmc_prefix


def _client(query: str = "a", **kw) -> NCBIClient:
    """A client with the three required fields filled in and nothing else assumed."""
    return NCBIClient(IngestConfig(
        query=query, ncbi_email="who@example.org", ncbi_tool="test", **kw
    ))


class TestBuildQuery:
    def test_the_base_is_parenthesised(self):
        """**Correct on purpose rather than by accident**, per the method's own docstring.

        PubMed evaluates booleans strictly left to right rather than binding AND tighter than OR, so
        an unparenthesised `A OR B AND C` silently means `(A OR B) AND C`. That happens to be what is
        wanted here, which is exactly why it must not be left implicit.
        """
        assert _client("a OR b").build_query() == "(a OR b)"

    def test_no_bounds_means_no_year_filter(self):
        assert "[dp]" not in _client().build_query()

    def test_both_bounds_apply_a_dp_range(self):
        assert _client().build_query(year_min=2018, year_max=2020) == "(a) AND 2018:2020[dp]"

    def test_a_lower_bound_alone_runs_to_an_open_upper(self):
        # 3000 is the "no upper bound" sentinel, not a real year.
        assert _client().build_query(year_min=2018) == "(a) AND 2018:3000[dp]"

    def test_an_upper_bound_alone_runs_from_an_open_lower(self):
        assert _client().build_query(year_max=2020) == "(a) AND 1800:2020[dp]"

    def test_dp_not_edat(self):
        """Publication date, deliberately — the docstring says why.

        Scoping a *full* pull by `edat` would drop old papers that were indexed into PubMed
        recently. The incremental path uses `edat` for the opposite reason; see
        `test_sync.TestSyncQuery`.
        """
        term = _client().build_query(year_min=2018)
        assert "[dp]" in term and "[edat]" not in term

    def test_the_term_comes_from_the_config(self):
        """What replaced the global fallback.

        `IngestConfig.query` has no default at all — a library cannot ship a sensible one, and anything
        generic would be *silently* wrong, producing a corpus nobody asked for rather than an error.
        """
        assert _client("configured OR terms").build_query() == "(configured OR terms)"

    def test_bounds_default_to_the_configs_own(self):
        """A caller that stated the bounds once in its corpus definition need not restate them here."""
        assert _client(year_min=1990, year_max=2000).build_query() == "(a) AND 1990:2000[dp]"

    def test_explicit_bounds_override_the_config(self):
        """What a one-off narrower pull needs, without editing the checked-in definition."""
        assert _client(year_min=1990).build_query(year_min=2015) == "(a) AND 2015:3000[dp]"

    def test_the_year_bounds_have_no_environment_fallback(self):
        """**They used to, and that was the blocker.**

        `year_min`/`year_max` fell back to `settings.ingest_year_min`/`_max`, read-only properties
        derived from `IS_DEV_ENVIRONMENT` — so scoping was implied by an environment variable and a
        caller could not turn it off. Both the properties and the switch are gone: unbounded unless the
        config or the call says otherwise, which is the shape a library can ship.
        """
        assert _client().build_query() == "(a)"


class TestStripPmcPrefix:
    """Kept module-level rather than made a method: pure string handling, no config involved."""

    def test_strips_the_stored_form(self):
        assert strip_pmc_prefix("PMC12345") == "12345"

    def test_is_case_insensitive(self):
        assert strip_pmc_prefix("pmc12345") == "12345"

    def test_leaves_a_bare_accession_alone(self):
        assert strip_pmc_prefix("12345") == "12345"
