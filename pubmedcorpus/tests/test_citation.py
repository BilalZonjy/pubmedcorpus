"""Citation renderer — pure formatting, no DB. Duck-typed stubs stand in for
ParsedPaper / ParsedAuthor so the formatter is tested in isolation.

The stub field names must match `ParsedAuthor` exactly (parse.py): the formatter
reads them with `getattr(..., None)` defaults, so a name that drifts here drops the
author list silently instead of failing. `test_store.py` guards the real call site.
"""

from dataclasses import dataclass

from pubmedcorpus import citation


@dataclass
class _A:
    position: int
    last_name: str | None = None
    fore_name: str | None = None
    initials: str | None = None
    collective_name: str | None = None


@dataclass
class _P:
    pmid: str = "111"
    title: str | None = None
    journal: str | None = None
    volume: str | None = None
    issue: str | None = None
    pages: str | None = None
    pub_year: int | None = None
    doi: str | None = None


class TestFormatReference:
    def test_full_vancouver(self):
        authors = [_A(1, "Ryvlin", "Philippe"), _A(2, "Nashef", "Lina")]
        p = _P(title="The MORTEMUS study", journal="Lancet Neurol", volume="12",
               issue="10", pages="966-977", pub_year=2013, doi="10.1016/x")
        assert citation.format_reference(p, authors) == (
            "Ryvlin P, Nashef L. The MORTEMUS study. Lancet Neurol. "
            "2013;12(10):966-977. doi:10.1016/x"
        )

    def test_et_al_after_six(self):
        authors = [_A(i, f"Last{i}", "Fore") for i in range(1, 8)]
        ref = citation.format_reference(_P(title="T", journal="J", pub_year=2020), authors)
        assert "et al." in ref
        assert "Last7" not in ref  # seventh author folded into "et al"

    def test_collective_author(self):
        ref = citation.format_reference(
            _P(title="T", pub_year=2019), [_A(1, collective_name="SUDEP Group")]
        )
        assert ref.startswith("SUDEP Group.")

    def test_pmid_fallback_when_no_doi(self):
        ref = citation.format_reference(_P(pmid="999", title="T", pub_year=2001), [])
        assert ref.endswith("PMID: 999")

    def test_year_only_locator(self):
        ref = citation.format_reference(_P(title="T", journal="J", pub_year=2005), [])
        assert "2005." in ref

    def test_initials_stripped_of_periods(self):
        ref = citation.format_reference(
            _P(title="T", pub_year=2000), [_A(1, "Smith", initials="J.R.")]
        )
        assert ref.startswith("Smith JR.")


class TestReferenceUrl:
    def test_pubmed_from_pmid(self):
        assert citation.reference_url(_P(pmid="123")) == "https://pubmed.ncbi.nlm.nih.gov/123/"

    def test_doi_fallback_when_no_pmid(self):
        assert citation.reference_url(_P(pmid=None, doi="10.1/x")) == "https://doi.org/10.1/x"

    def test_none_when_neither(self):
        assert citation.reference_url(_P(pmid=None, doi=None)) is None
