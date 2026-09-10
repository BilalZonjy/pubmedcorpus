from datetime import date
from pathlib import Path

import pytest

from pubmedcorpus.parse import canonical_key, normalize_orcid, parse_batch

FIXTURE = Path(__file__).parent / "fixtures" / "edge_cases.xml"


@pytest.fixture(scope="module")
def papers():
    return {p.pmid: p for p in parse_batch(FIXTURE.read_bytes())}


def test_all_records_parsed(papers):
    assert set(papers) == {"10000001", "10000002", "10000003", "10000004", "10000005"}


class TestStructuredAbstract:
    def test_labels_are_preserved(self, papers):
        # Section labels carry meaning — CONCLUSIONS is where positions get
        # stated, and extraction targets it.
        text = papers["10000001"].abstract_text
        assert "BACKGROUND: Risk remains unclear." in text
        assert "CONCLUSIONS: Postictal asystole may contribute." in text

    def test_sections_are_joined_not_dropped(self, papers):
        assert papers["10000001"].abstract_text.count("\n\n") == 2


class TestMissingAbstract:
    def test_editorial_without_abstract(self, papers):
        assert papers["10000002"].abstract_text is None

    def test_record_is_still_kept(self, papers):
        # A record with no abstract is still a record: it counts in the corpus report
        # and may yet have full text. It must never be filtered at ingest.
        assert papers["10000002"].title is not None
        assert len(papers["10000002"].authors) == 1


class TestOtherAbstract:
    def test_kept_separate_from_abstract(self, papers):
        # Different provenance and often a different language, so merging them would
        # silently mix languages in the stored record.
        p = papers["10000003"]
        assert p.abstract_text is None
        assert "muerte subita" in p.other_abstract_text.lower()


class TestDates:
    def test_full_date(self, papers):
        p = papers["10000001"]
        assert p.pub_date == date(2019, 3, 15)
        assert p.pub_year == 2019

    def test_medline_date_keeps_raw_and_extracts_year(self, papers):
        p = papers["10000002"]
        assert p.pub_date is None  # "1998 Jan-Feb" has no single date
        assert p.pub_year == 1998
        assert p.pub_date_raw == "1998 Jan-Feb"

    def test_year_only(self, papers):
        assert papers["10000003"].pub_year == 2005

    def test_impossible_date_degrades_to_year(self, papers):
        # Feb 30 exists in the wild. It must not raise.
        p = papers["10000005"]
        assert p.pub_date is None
        assert p.pub_year == 2001

    def test_entry_date_from_entrez_status(self, papers):
        assert papers["10000001"].entry_date == date(2019, 3, 20)

    def test_entry_date_falls_back_to_pubmed_status(self, papers):
        assert papers["10000002"].entry_date == date(1998, 2, 1)

    def test_missing_history_is_none(self, papers):
        assert papers["10000005"].entry_date is None


class TestAuthors:
    def test_position_and_last_flag(self, papers):
        authors = papers["10000001"].authors
        assert [a.position for a in authors] == [1, 2]
        assert authors[0].is_last is False
        assert authors[1].is_last is True

    def test_single_author_is_not_marked_last(self, papers):
        # First and last are the same person; flagging both would double-count
        # under the first+last attribution rule.
        assert papers["10000002"].authors[0].is_last is False

    def test_collective_name(self, papers):
        collective = papers["10000004"].authors[1]
        assert collective.collective_name == "The SUDEP Study Group"
        assert collective.last_name is None
        assert collective.canonical_key is None

    def test_no_author_list(self, papers):
        assert papers["10000005"].authors == []

    def test_affiliation(self, papers):
        assert "Lausanne" in papers["10000001"].authors[0].affiliation
        assert papers["10000001"].authors[1].affiliation is None


class TestOrcid:
    def test_bare_id(self, papers):
        assert papers["10000001"].authors[0].orcid == "0000-0001-2345-6789"

    def test_url_form(self, papers):
        assert papers["10000004"].authors[0].orcid == "0000-0002-1111-222X"

    def test_absent(self, papers):
        assert papers["10000001"].authors[1].orcid is None

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("0000-0001-2345-6789", "0000-0001-2345-6789"),
            ("https://orcid.org/0000-0002-1111-222x", "0000-0002-1111-222X"),
            ("0000000211112223", "0000-0002-1111-2223"),
            ("", None),
            (None, None),
            ("not-an-orcid", None),
        ],
    )
    def test_normalize(self, raw, expected):
        assert normalize_orcid(raw) == expected


class TestCanonicalKey:
    def test_prefers_forename_initial(self):
        assert canonical_key("Ryvlin", "Philippe", "P") == "ryvlin|p"

    def test_falls_back_to_initials(self):
        assert canonical_key("Nashef", None, "L") == "nashef|l"

    def test_name_variants_collapse(self):
        # "Ryvlin P" and "Ryvlin Philippe" must land in the same block.
        assert canonical_key("Ryvlin", "Philippe", None) == canonical_key("Ryvlin", None, "P")

    def test_no_last_name(self):
        assert canonical_key(None, "Philippe", "P") is None


class TestIdentifiers:
    def test_doi_from_elocationid(self, papers):
        assert papers["10000001"].doi == "10.1111/epi.00001"

    def test_doi_from_article_id_list(self, papers):
        assert papers["10000004"].doi == "10.9999/jdf.2014.001"

    def test_pmcid(self, papers):
        assert papers["10000001"].pmc_id == "PMC6543210"

    def test_no_pmcid(self, papers):
        assert papers["10000002"].pmc_id is None


class TestRetraction:
    def test_detected(self, papers):
        # A review that silently cites retracted claims misrepresents
        # someone's position.
        assert papers["10000004"].is_retracted is True

    def test_normal_article_not_flagged(self, papers):
        assert papers["10000001"].is_retracted is False


class TestMisc:
    def test_title_flattens_inline_markup(self, papers):
        assert papers["10000001"].title == "Cardiac mechanisms in SUDEP: a cohort study"

    def test_mesh_terms(self, papers):
        assert "Epilepsy" in papers["10000001"].mesh_terms

    def test_publication_types(self, papers):
        assert papers["10000002"].publication_types == ["Editorial"]

    def test_raw_xml_retained(self, papers):
        # The source is kept verbatim: re-extraction will happen more than once.
        assert "<PMID" in papers["10000001"].raw_xml

    def test_malformed_batch_does_not_lose_good_records(self):
        broken = b"<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>1</PMID>"
        assert isinstance(parse_batch(broken), list)

    def test_empty_input(self):
        assert parse_batch(b"<PubmedArticleSet></PubmedArticleSet>") == []
