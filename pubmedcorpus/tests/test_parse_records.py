"""Parser tests against real, committed PubMed records.

Each fixture is one `efetch` response saved verbatim under fixtures/<pmid>.xml,
so these run offline and deterministically — no NCBI calls. Real records catch
what hand-written XML doesn't: NLM's actual entity usage, full author lists,
and correction metadata that only shows up in the wild.

To add a record:

    dc exec api python -c \
      "from sudep.config import corpus_config; from pubmedcorpus.ncbi import NCBIClient; \
       import sys; \
       sys.stdout.buffer.write(NCBIClient(corpus_config()).fetch_pmids(['PMID']))" \
      > backend/pubmedcorpus/tests/fixtures/PMID.xml

then add a class here asserting what makes that record worth keeping.
"""

from datetime import date
from pathlib import Path

import pytest

from pubmedcorpus.parse import parse_batch

FIXTURES = Path(__file__).parent / "fixtures"


def load(pmid: str):
    """Parse one committed record and return it."""
    papers = parse_batch((FIXTURES / f"{pmid}.xml").read_bytes())
    assert papers, f"fixture {pmid}.xml parsed to nothing"
    return papers[0]


@pytest.fixture(scope="module")
def lancet():
    """Lhatoo et al 2025, Lancet — structured abstract, 33 authors, erratum."""
    return load("40975113")


class TestIdentifiers:
    def test_pmid(self, lancet):
        assert lancet.pmid == "40975113"

    def test_doi_from_elocationid(self, lancet):
        assert lancet.doi == "10.1016/S0140-6736(25)01636-8"

    def test_pmcid(self, lancet):
        assert lancet.pmc_id == "PMC12707170"

    def test_journal(self, lancet):
        assert lancet.journal == "Lancet (London, England)"

    def test_raw_xml_retained(self, lancet):
        # Re-extraction will happen more than once; the source is kept verbatim.
        assert "<PMID" in lancet.raw_xml


class TestDates:
    def test_publication_date(self, lancet):
        assert lancet.pub_date == date(2025, 10, 4)
        assert lancet.pub_year == 2025

    def test_entry_date_comes_from_entrez_not_publication(self, lancet):
        # Indexed two weeks *before* the print issue date. This gap is exactly
        # why incremental sync scopes by edat rather than pdat.
        assert lancet.entry_date == date(2025, 9, 20)
        assert lancet.entry_date < lancet.pub_date


class TestStructuredAbstract:
    def test_all_five_sections_joined(self, lancet):
        assert lancet.abstract_text.count("\n\n") == 4

    def test_labels_preserved(self, lancet):
        text = lancet.abstract_text
        assert text.startswith("BACKGROUND: ")
        # INTERPRETATION is where the position is stated — extraction targets it.
        assert "INTERPRETATION: This study shows an association" in text

    def test_no_other_abstract(self, lancet):
        assert lancet.other_abstract_text is None

    def test_numeric_entities_decoded(self, lancet):
        # NLM writes the middle dot in Lancet-style numbers as &#183;. If entity
        # decoding regressed, "1·54%" would reach the LLM as "1&#183;54%".
        assert "1·54%" in lancet.abstract_text


class TestAuthors:
    def test_full_list_kept(self, lancet):
        assert len(lancet.authors) == 33

    def test_first_and_last(self, lancet):
        assert lancet.authors[0].last_name == "Ochoa-Urrea"
        assert lancet.authors[-1].last_name == "Lhatoo"

    def test_only_final_author_is_flagged_last(self, lancet):
        assert [a.position for a in lancet.authors if a.is_last] == [33]

    def test_canonical_key(self, lancet):
        assert lancet.authors[0].canonical_key == "ochoa-urrea|m"

    def test_distinct_people_share_a_blocking_key(self, lancet):
        # Johnson P Hampson and Jaison S Hampson are different people on the
        # same paper. The key is deliberately coarse — it groups candidates,
        # Phase 5 splits them. If this ever passes as two keys, disambiguation
        # silently lost a collision it was built to handle.
        hampsons = [a for a in lancet.authors if a.last_name == "Hampson"]
        assert len(hampsons) == 2
        assert hampsons[0].canonical_key == hampsons[1].canonical_key == "hampson|j"

    def test_affiliation_captured(self, lancet):
        assert "McGovern Medical School" in lancet.authors[0].affiliation

    def test_no_orcids_in_this_record(self, lancet):
        # Typical of the corpus: ORCID coverage is thin, so nothing downstream
        # may assume it exists.
        assert all(a.orcid is None for a in lancet.authors)


class TestClassification:
    def test_publication_types(self, lancet):
        assert lancet.publication_types == [
            "Journal Article",
            "Multicenter Study",
            "Observational Study",
        ]

    def test_erratum_is_not_a_retraction(self, lancet):
        # This record carries <CommentsCorrections RefType="ErratumIn">. An
        # erratum corrects a paper; a retraction withdraws it. Conflating them
        # would drop a valid paper out of the corpus entirely.
        assert "ErratumIn" in lancet.raw_xml
        assert lancet.is_retracted is False

    def test_mesh_terms(self, lancet):
        assert "Sudden Unexpected Death in Epilepsy" in lancet.mesh_terms
        assert "Epilepsy" in lancet.mesh_terms
