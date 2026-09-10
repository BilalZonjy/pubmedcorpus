"""Promotion decisions for full-text fetching.

Offline: these exercise the rules applied to one record, not the database query that
selects them. `_apply` is where a record becomes a paper — it writes the `paper` row —
so the session is faked and the emitted INSERT values are read back.
"""

from sqlalchemy.dialects import postgresql

from pubmedcorpus.jats import ParsedFullText, Section
from pubmedcorpus.models import Abstract, Paper
from pubmedcorpus.pmc import (OUTCOME_FAILED, OUTCOME_NO_PMCID, OUTCOME_NOT_OA,
                              OUTCOME_OK, _apply)

# A minimal but real PubMed record: promotion re-parses this to render the citation,
# so a stub string would fail the way a corrupt row would.
_PUBMED_XML = """<PubmedArticle><MedlineCitation><PMID>1</PMID><Article>
<ArticleTitle>Risk markers for SUDEP</ArticleTitle>
<Journal><Title>Lancet</Title><JournalIssue><Volume>406</Volume>
<PubDate><Year>2025</Year></PubDate></JournalIssue></Journal>
<AuthorList><Author><LastName>Lhatoo</LastName><Initials>SD</Initials></Author></AuthorList>
</Article></MedlineCitation></PubmedArticle>"""


class _FakeSession:
    """Captures the statements `_apply` executes. No DB, no ORM flush."""

    def __init__(self):
        self.statements = []

    def execute(self, statement):
        self.statements.append(statement)


def record(**kwargs) -> Abstract:
    kwargs.setdefault("pubmed_xml", _PUBMED_XML)
    kwargs.setdefault("paper", None)
    return Abstract(pmid="1", **kwargs)


def with_body(**kwargs) -> ParsedFullText:
    return ParsedFullText(
        pmcid="PMC1",
        raw_xml="<article><body/></article>",
        sections=[Section(type="discussion", title="Discussion", text="We argue X.")],
        **kwargs,
    )


def promote(rec: Abstract, parsed: ParsedFullText) -> tuple[str, dict]:
    """Run `_apply` and return (outcome, the values it tried to insert)."""
    session = _FakeSession()
    outcome = _apply(session, rec, parsed)
    if not session.statements:
        return outcome, {}
    # ON CONFLICT is postgresql-specific, so the dialect is not optional here.
    compiled = session.statements[0].compile(dialect=postgresql.dialect())
    return outcome, dict(compiled.params)


# `TestStripPmcPrefix` moved to `pubmedcorpus/tests/test_ncbi.py` with the function itself — it is
# E-utilities' accession format, not part of the fetch job's behaviour.


class TestPromotion:
    def test_body_creates_the_paper_row(self):
        rec = record()
        outcome, values = promote(rec, with_body())
        assert outcome == OUTCOME_OK
        assert values["pmid"] == "1"
        assert [s["type"] for s in values["fulltext_jats"]["sections"]] == ["discussion"]
        assert rec.fetch_failed is False

    def test_raw_article_xml_is_kept(self):
        # parse_article drops everything outside <sec>/<p>; without this column that
        # content is unrecoverable without re-asking PMC.
        _outcome, values = promote(record(), with_body())
        assert values["jats_xml"] == "<article><body/></article>"

    def test_citation_is_rendered_from_the_stored_record(self):
        # Promotion is a call site for the duck-typed formatter: if its field names
        # drift from ParsedAuthor the author segment vanishes with no error.
        _outcome, values = promote(record(), with_body())
        assert values["citation"].startswith("Lhatoo SD.")
        assert "Risk markers for SUDEP" in values["citation"]

    def test_licence_recorded(self):
        _outcome, values = promote(record(), with_body(license="CC BY-NC"))
        assert values["fulltext_jats"]["license"] == "CC BY-NC"

    def test_existing_licence_survives_a_licenceless_refetch(self):
        # PMC occasionally omits permissions. The licence rides inside the payload,
        # which a refetch replaces wholesale — blanking it would lose the only record
        # of what reuse is permitted.
        already = Paper(pmid="1", fulltext_jats={"license": "CC BY", "sections": []})
        _outcome, values = promote(record(paper=already), with_body(license=None))
        assert values["fulltext_jats"]["license"] == "CC BY"

    def test_unparseable_stored_record_fails_rather_than_promoting(self):
        # A paper with no citation must not exist. This is a defect in our own stored
        # XML, not a PMC outcome, so it fails the attempt and writes nothing.
        rec = record(pubmed_xml="not xml at all")
        outcome, values = promote(rec, with_body())
        assert outcome == OUTCOME_FAILED
        assert values == {}
        assert rec.fetch_failed is True


class TestNoBody:
    def test_front_matter_only_is_not_oa(self):
        # PMC serves metadata for non-OA records rather than failing. Expected, not an
        # error — and deliberately not stored as an empty paper.
        rec = record()
        outcome, values = promote(rec, ParsedFullText(pmcid="PMC1"))
        assert outcome == OUTCOME_NOT_OA
        assert values == {}
        assert rec.fetch_failed is True

    def test_empty_sections_do_not_count_as_full_text(self):
        # A section list whose entries are all empty is not a retrieved body; a paper
        # row for it would look extractable while having no text.
        empty = ParsedFullText(pmcid="PMC1", sections=[Section("other", None, "")])
        rec = record()
        outcome, values = promote(rec, empty)
        assert outcome == OUTCOME_NOT_OA
        assert values == {}


# --- The outer loop -----------------------------------------------------------------
#
# **Untested until session injection made it reachable.** `fetch_fulltext` used to open its own
# `session_scope`, so nothing could call it offline; the tests above cover `_apply`, the one piece
# that already took a session. What is checked here is what the *loop* decides — which outcome each
# record gets and, critically, when it commits — because the resumability the docstring promises is
# the per-batch commit and nothing else.


class _LoopSession(_FakeSession):
    """`_FakeSession` plus the two things the loop needs: query results and a commit count.

    Commits are counted rather than ignored because "committed with its batch" is the whole
    resumability claim; a version that committed once at the end would pass every assertion about
    outcomes and still lose an interrupted run's work.
    """

    def __init__(self, records):
        super().__init__()
        self._records = records
        self.commits = 0

    def execute(self, statement):
        self.statements.append(statement)
        return _Scalars(self._records)

    def commit(self):
        self.commits += 1


class _Scalars:
    def __init__(self, rows):
        self._rows = rows

    def scalars(self):
        return self._rows


class _FakeClient:
    """Stands in for `NCBIClient`: canned PMC responses, no network.

    `fetch_pmc` returns bytes the real one would, but `parse_pmc_batch` is patched alongside it —
    building real JATS XML here would test the parser, which `test_jats.py` already does.
    """

    def __init__(self, config, pmc=b"<x/>", elink=None, raises=False):
        self.config = config
        self._pmc, self._elink, self._raises = pmc, elink, raises
        self.pmc_calls = 0

    def fetch_pmc(self, pmcids):
        self.pmc_calls += 1
        if self._raises:
            raise RuntimeError("PMC said no")
        return self._pmc

    def elink_pmcid(self, pmid):
        return self._elink


def _config(**kw):
    from pubmedcorpus.config import IngestConfig

    return IngestConfig(query="a", ncbi_email="e@x.org", ncbi_tool="t", **kw)


def _run(records, monkeypatch, articles=None, config=None, kwargs=None, **client_kw):
    """Run `fetch_fulltext` against fakes. Returns (counts, session, client).

    `config` goes to the `IngestConfig`, `kwargs` to `fetch_fulltext`, and anything else to the fake
    client — spelled out as named parameters rather than sifted out of one `**kw` bag, which is easy to
    get wrong in the order things are popped.
    """
    from pubmedcorpus import pmc as pmc_mod

    monkeypatch.setattr(pmc_mod, "parse_pmc_batch", lambda _xml: articles or [])
    session = _LoopSession(records)
    client = _FakeClient(_config(**(config or {})), **client_kw)
    counts = pmc_mod.fetch_fulltext(session, client, **(kwargs or {}))
    return counts, session, client


class TestFetchFulltext:
    def test_no_candidates_returns_empty_and_asks_pmc_nothing(self, monkeypatch):
        """The early return. A no-op run must not cost a request."""
        counts, session, client = _run([], monkeypatch)
        assert counts == {}
        assert client.pmc_calls == 0
        assert session.commits == 0

    def test_a_record_without_a_pmcid_is_marked_failed_not_requested(self, monkeypatch):
        """No PMCID means there is nothing to ask PMC for — recorded, not retried forever."""
        rec = record(pmc_id=None)
        counts, _, client = _run([rec], monkeypatch)
        assert counts == {OUTCOME_NO_PMCID: 1}
        assert rec.fetch_failed is True
        assert client.pmc_calls == 0

    def test_relink_resolves_a_missing_pmcid(self, monkeypatch):
        """Opt-in because elink costs one request per record — see `elink_pmcid`'s docstring."""
        rec = record(pmc_id=None)
        counts, _, _ = _run(
            [rec], monkeypatch, articles=[with_body(pmid="1")],
            elink="PMC1", kwargs={"relink": True},
        )
        assert rec.pmc_id == "PMC1"
        assert counts == {OUTCOME_OK: 1}

    def test_a_requested_record_pmc_does_not_return_is_not_oa(self, monkeypatch):
        rec = record(pmc_id="PMC1")
        counts, _, _ = _run([rec], monkeypatch, articles=[])
        assert counts == {OUTCOME_NOT_OA: 1}
        assert rec.fetch_failed is True

    def test_a_failing_batch_is_marked_retryable_and_does_not_end_the_run(self, monkeypatch):
        """One bad batch must not abort the rest — `--retry` exists precisely for these."""
        rec = record(pmc_id="PMC1")
        counts, session, _ = _run([rec], monkeypatch, raises=True)
        assert counts == {OUTCOME_FAILED: 1}
        assert rec.fetch_failed is True
        assert session.commits >= 1

    def test_it_commits_once_per_batch_not_once_at_the_end(self, monkeypatch):
        """**The resumability guarantee, stated as a number.**

        Three records at `pmc_batch_size=1` is three batches. With the PMCID-resolution commit that
        precedes them, a correct run commits four times; a version that saved everything at the end
        would commit once and silently lose an interrupted run's completed work.
        """
        recs = [record(pmc_id=f"PMC{i}") for i in range(3)]
        for i, r in enumerate(recs):
            r.pmid = str(i)
        counts, session, client = _run(
            recs, monkeypatch, articles=[], config={"pmc_batch_size": 1},
        )
        assert client.pmc_calls == 3
        assert session.commits == 4
        assert counts == {OUTCOME_NOT_OA: 3}

    def test_batch_size_defaults_to_the_configs_value(self, monkeypatch):
        """One batch, not three — the config supplies it when the caller does not."""
        recs = [record(pmc_id=f"PMC{i}") for i in range(3)]
        for i, r in enumerate(recs):
            r.pmid = str(i)
        _, _, client = _run(recs, monkeypatch, articles=[], config={"pmc_batch_size": 20})
        assert client.pmc_calls == 1
