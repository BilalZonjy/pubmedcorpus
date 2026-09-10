"""Ingest write-point invariants. Pure — no database.

Covers the two dicts ingest writes, because both fail *silently* if they regress:
`pubmed_json` only errors against a live connection if it holds something
JSON-incompatible, and the citation formatter is duck-typed, so a field-name drift
drops the author list without raising anywhere.
"""

import json
from datetime import date

from pubmedcorpus.parse import ParsedAuthor, ParsedPaper
from pubmedcorpus.store import (
    _ABSTRACT_UPDATE_COLS,
    _blank_to_none,
    abstract_values,
    as_pubmed_json,
    paper_values,
)


def a_parsed_paper() -> ParsedPaper:
    return ParsedPaper(
        pmid="40975113",
        raw_xml="<PubmedArticle/>",
        title="Risk markers for SUDEP",
        journal="Lancet",
        volume="406",
        issue="10510",
        pages="1234-1245",
        doi="10.1016/S0140-6736(25)01636-8",
        pub_date=date(2025, 10, 4),
        pub_year=2025,
        entry_date=date(2025, 9, 20),
        abstract_text="Background: ...",
        mesh_terms=["Epilepsy", "Death, Sudden"],
        pmc_id="PMC12707170",
        authors=[
            ParsedAuthor(position=1, last_name="Ochoa-Urrea", fore_name="Manuela",
                         initials="M", canonical_key="ochoa-urrea|m"),
            ParsedAuthor(position=2, last_name="Lhatoo", fore_name="Samden D", initials="SD",
                         is_last=True, affiliation="McGovern Medical School",
                         orcid="0000-0001-2345-6789", canonical_key="lhatoo|s"),
        ],
    )


class TestBlankToNone:
    def test_none_stays_none(self):
        assert _blank_to_none(None) is None

    def test_empty_becomes_none(self):
        assert _blank_to_none("") is None

    def test_whitespace_only_becomes_none(self):
        # The case that would otherwise pass `IS NOT NULL` with nothing to read.
        assert _blank_to_none("   \n\t ") is None

    def test_real_content_is_preserved_verbatim(self):
        # Only the blank case changes; surrounding whitespace on real text stays.
        assert _blank_to_none("  an abstract  ") == "  an abstract  "


class TestFrozenCitation:
    """The guard on the formatter's duck typing.

    `citation._author_name` reads `last_name` / `fore_name` / `initials` /
    `collective_name` with `getattr(..., None)` defaults. If those names drift from
    `ParsedAuthor`, every lookup returns None, `format_authors` returns "", and every
    promoted paper gets a citation that silently begins at the title — with no
    exception anywhere. These assertions are what makes that loud.
    """

    def test_citation_leads_with_the_authors(self):
        assert paper_values(a_parsed_paper())["citation"].startswith("Ochoa-Urrea M, Lhatoo SD.")

    def test_citation_keeps_the_locator_that_has_no_column(self):
        # volume/issue/pages are not stored anywhere as columns — they survive only
        # because the reference is rendered here and frozen as text.
        assert paper_values(a_parsed_paper())["citation"] == (
            "Ochoa-Urrea M, Lhatoo SD. Risk markers for SUDEP. Lancet. "
            "2025;406(10510):1234-1245. doi:10.1016/S0140-6736(25)01636-8"
        )

    def test_never_empty_even_with_no_metadata(self):
        # The column is NOT NULL, and the PMID fallback in `_identifier` is what
        # guarantees a bare record still renders something citeable.
        assert paper_values(ParsedPaper(pmid="999", raw_xml="<x/>"))["citation"] == "PMID: 999"


class TestAbstractValues:
    def test_columns_are_the_queried_ones(self):
        assert set(abstract_values(a_parsed_paper())) == {
            "pmid", "pubmed_xml", "pubmed_json",
            "entry_date", "pub_year", "pmc_id", "is_retracted",
        }

    def test_pubmed_xml_is_the_verbatim_record(self):
        assert abstract_values(a_parsed_paper())["pubmed_xml"] == "<PubmedArticle/>"

    def test_fetch_failed_is_not_refreshed_by_a_resync(self):
        # It belongs to pmc-fetch. If sync reset it, every re-sync would re-open
        # thousands of records PMC has already declined.
        assert "fetch_failed" not in _ABSTRACT_UPDATE_COLS
        assert "fetch_failed" not in abstract_values(a_parsed_paper())


class TestPubmedJson:
    """Everything with no column of its own has to be reachable here."""

    def test_carries_the_author_detail_that_has_no_table(self):
        authors = as_pubmed_json(a_parsed_paper())["authors"]
        assert [a["position"] for a in authors] == [1, 2]
        assert authors[1]["orcid"] == "0000-0001-2345-6789"
        assert authors[1]["affiliation"] == "McGovern Medical School"
        assert authors[1]["canonical_key"] == "lhatoo|s"
        assert authors[1]["is_last"] is True

    def test_carries_the_fields_that_have_no_column(self):
        row = as_pubmed_json(a_parsed_paper())
        assert row["title"] == "Risk markers for SUDEP"
        assert row["journal"] == "Lancet"
        assert row["volume"] == "406"
        assert row["doi"] == "10.1016/S0140-6736(25)01636-8"
        assert row["mesh_terms"] == ["Epilepsy", "Death, Sudden"]

    def test_carries_the_abstract(self):
        # The report's coverage rows count on this being queryable.
        assert as_pubmed_json(a_parsed_paper())["abstract_text"] == "Background: ..."

    def test_excludes_the_xml(self):
        # It has its own column; nesting it here would be a second copy of the
        # largest field in the corpus.
        assert "raw_xml" not in as_pubmed_json(a_parsed_paper())

    def test_dates_are_iso_strings(self):
        # JSONB has no date type — a raw date would fail on the way to Postgres.
        row = as_pubmed_json(a_parsed_paper())
        assert row["pub_date"] == "2025-10-04"
        assert row["entry_date"] == "2025-09-20"

    def test_is_json_serialisable(self):
        # The real failure this catches: a dataclass field json can't encode would
        # only blow up against a live connection.
        assert json.loads(json.dumps(as_pubmed_json(a_parsed_paper())))["pmid"] == "40975113"

    def test_null_dates_stay_null(self):
        assert as_pubmed_json(ParsedPaper(pmid="1", raw_xml="<x/>"))["pub_date"] is None
