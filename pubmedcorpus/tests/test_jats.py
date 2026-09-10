from pathlib import Path

import pytest

from pubmedcorpus.jats import (
    BACK_MATTER,
    CONCLUSIONS,
    DISCUSSION,
    INTRO,
    METHODS,
    OTHER,
    RESULTS,
    classify_section,
    effective_type,
    parse_pmc_batch,
)

FIXTURE = Path(__file__).parent / "fixtures" / "pmc_articles.xml"


@pytest.fixture(scope="module")
def articles():
    return {a.pmid: a for a in parse_pmc_batch(FIXTURE.read_bytes())}


def test_all_articles_parsed(articles):
    assert set(articles) == {"20000001", "20000002", "20000003", "20000004"}


class TestIdentifiers:
    def test_pmcid_gets_prefix(self, articles):
        # PMC returns the bare accession; we store the prefixed form so it
        # matches what PubMed's ArticleIdList gave us.
        assert articles["20000001"].pmcid == "PMC7000001"

    def test_pmid_captured(self, articles):
        # This is what maps a returned article back to its paper row — PMC may
        # reorder or omit articles, so position can't be trusted.
        assert articles["20000001"].pmid == "20000001"

    def test_title(self, articles):
        assert articles["20000001"].title == "Cardiac mechanisms in SUDEP"

    def test_abstract(self, articles):
        # Extraction reads the abstract from here — the article's own front matter —
        # rather than from PubMed's separately-indexed copy, so this is load-bearing
        # for the prompt, not just metadata.
        assert articles["20000001"].abstract == "We reviewed cardiac mechanisms in SUDEP."


class TestRawArticleKept:
    """Section parsing is lossy: anything outside <sec>/<p> — tables, figure captions,
    boxed text — never reaches `sections`. The raw article is kept so that content is
    recoverable without re-asking PMC."""

    def test_raw_xml_is_the_article_element(self, articles):
        raw = articles["20000001"].raw_xml
        assert raw.lstrip().startswith("<article")
        assert "Cardiac mechanisms in SUDEP" in raw

    def test_each_article_gets_its_own(self, articles):
        # Parsed from a batch response; slicing the wrong element would give every
        # paper the whole <pmc-articleset>.
        assert "20000002" not in articles["20000001"].raw_xml

    def test_as_dict_excludes_it(self, articles):
        # It has its own column. Nesting it in the JSONB payload would be a second
        # copy of the largest field on the row.
        assert "raw_xml" not in articles["20000001"].as_dict()


class TestLicence:
    def test_creative_commons_from_href(self, articles):
        assert articles["20000001"].license == "CC BY"

    def test_restrictive_variant_is_distinguished(self, articles):
        # BY-NC-ND is not BY. Collapsing them would misreport what may be reused.
        assert articles["20000002"].license == "CC BY-NC-ND"

    def test_falls_back_to_license_type(self, articles):
        assert articles["20000004"].license == "publisher-standard"

    def test_absent(self, articles):
        assert articles["20000003"].license is None


class TestSections:
    def test_types_from_sec_type(self, articles):
        assert [s.type for s in articles["20000001"].sections] == [
            INTRO,
            METHODS,
            RESULTS,
            DISCUSSION,
            CONCLUSIONS,
        ]

    def test_types_from_title_when_sec_type_absent(self, articles):
        # A large share of publishers omit @sec-type entirely.
        assert [s.type for s in articles["20000002"].sections] == [INTRO, DISCUSSION]

    def test_subsection_folded_into_parent(self, articles):
        # "Discussion > Limitations" is still discussion. Emitting it separately
        # would make section-targeted extraction miss half the text.
        discussion = articles["20000001"].sections[3]
        assert "Postictal asystole may contribute." in discussion.text
        assert "Our search was English only." in discussion.text

    def test_unsectioned_body_is_kept(self, articles):
        # Editorials and letters often have no <sec> at all, and they are
        # exactly the material that states positions most directly.
        sections = articles["20000004"].sections
        assert [s.type for s in sections] == [OTHER]
        assert "not settled on a mechanism" in sections[0].text

    def test_paragraphs_are_separated(self, articles):
        assert articles["20000004"].sections[0].text.count("\n\n") == 1


class TestOpenAccess:
    def test_body_present(self, articles):
        assert articles["20000001"].has_body is True

    def test_front_matter_only_is_not_an_error(self, articles):
        # PMC serves metadata for non-OA records rather than failing, so this
        # is how "not open access" is actually detected.
        paywalled = articles["20000003"]
        assert paywalled.sections == []
        assert paywalled.has_body is False


class TestPositionText:
    def test_gathers_discussion_and_conclusions(self, articles):
        text = articles["20000001"].section_text(DISCUSSION, CONCLUSIONS)
        assert "Postictal asystole" in text
        assert "remain unproven" in text
        # Methods and Results are noise for the belief schema and must not leak in.
        assert "searched PubMed" not in text
        assert "forty studies" not in text

    def test_none_when_absent(self, articles):
        assert articles["20000003"].section_text(DISCUSSION) is None


class TestSerialisation:
    def test_as_dict_shape(self, articles):
        payload = articles["20000002"].as_dict()
        assert payload["pmcid"] == "PMC7000002"
        assert payload["license"] == "CC BY-NC-ND"
        assert [s["type"] for s in payload["sections"]] == [INTRO, DISCUSSION]
        assert set(payload["sections"][0]) == {"type", "title", "text"}


class TestClassifySection:
    @pytest.mark.parametrize(
        "sec_type,title,expected",
        [
            ("intro", None, INTRO),
            ("materials|methods", None, METHODS),
            (None, "Materials and Methods", METHODS),
            (None, "Patients and Methods", METHODS),
            (None, "Statistical analysis", METHODS),
            (None, "Results", RESULTS),
            (None, "Discussion", DISCUSSION),
            (None, "Conclusions", CONCLUSIONS),
            (None, "Summary", CONCLUSIONS),
            (None, "Concluding remarks", CONCLUSIONS),
            (None, "Final Considerations", CONCLUSIONS),
            (None, "Background", INTRO),
            (None, "Acknowledgements", BACK_MATTER),
            (None, "Conflicts of interest", BACK_MATTER),
            (None, "References", BACK_MATTER),
            (None, "EPILEPSY: GENERAL ASPECTS", OTHER),
            (None, "SUDEP in Patients with Refractory Epilepsy", OTHER),
            (None, None, OTHER),
        ],
    )
    def test_classification(self, sec_type, title, expected):
        assert classify_section(sec_type, title) == expected

    def test_combined_heading_prefers_discussion(self):
        # "Results and Discussion" is one section carrying both. Discussion is
        # the part with positions in it, so that has to win.
        assert classify_section(None, "Results and Discussion") == DISCUSSION

    def test_sec_type_wins_over_a_misleading_title(self):
        assert classify_section("discussion", "General remarks") == DISCUSSION

    def test_back_matter_is_checked_before_discussion(self):
        # "Conflicts of interest and acknowledgements" must not fall through to a
        # content pattern just because BACK_MATTER isn't checked first.
        assert classify_section(None, "Conflicts of Interest and Acknowledgements") == BACK_MATTER

    def test_a_bare_patient_mention_is_not_methods(self):
        # The old pattern matched bare "patient"/"subject"/"participant", so a
        # topically organised review with "Patients" in a heading misclassified as
        # methods and was excluded from position text. "Patients and Methods"
        # (above) still matches — on "methods", not "patients".
        assert classify_section(None, "SUDEP in Patients with Refractory Epilepsy") != METHODS


class TestEffectiveType:
    """Re-classifying a STORED section as it is read, so a widened pattern reaches the
    existing corpus with no re-parse. The asymmetry is the point: recompute only what
    the parser gave up on."""

    def test_a_stored_type_wins_over_the_title(self):
        # The load-bearing case. `@sec-type` is not kept in the JSON, so the stored
        # `discussion` is the ONLY remaining trace that the publisher labelled this —
        # recomputing from "General remarks" alone would demote it to `other` and
        # silently change what a currently-working paper is extracted from.
        assert effective_type({"type": DISCUSSION, "title": "General remarks"}) == DISCUSSION

    def test_other_is_reclassified_from_the_title(self):
        # The whole point: a corpus row parsed under the old patterns.
        assert effective_type({"type": OTHER, "title": "FINAL CONSIDERATIONS"}) == CONCLUSIONS
        assert effective_type({"type": OTHER, "title": "Acknowledgements"}) == BACK_MATTER

    def test_an_unclassifiable_title_stays_other(self):
        assert effective_type({"type": OTHER, "title": "EPILEPSY: GENERAL ASPECTS"}) == OTHER

    def test_a_titleless_section_stays_other(self):
        # An unsectioned editorial: no heading to read, and none was stored.
        assert effective_type({"type": OTHER, "title": None, "text": "..."}) == OTHER

    def test_a_missing_type_is_classified_not_crashed(self):
        assert effective_type({"title": "Discussion"}) == DISCUSSION
        assert effective_type({}) == OTHER

    def test_it_never_moves_a_section_into_other(self):
        # Monotonic by construction — this is what makes the upgrade safe to apply on
        # read without a version bump. Nothing classified can be un-classified.
        for stored in (INTRO, METHODS, RESULTS, DISCUSSION, CONCLUSIONS, BACK_MATTER):
            assert effective_type({"type": stored, "title": "nonsense heading"}) == stored
