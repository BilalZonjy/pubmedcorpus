"""Parser extension: volume / issue / pages, via `parse_one` on inline XML."""

from pubmedcorpus.parse import parse_one

_WITH_MEDLINE_PGN = """
<PubmedArticle><MedlineCitation><PMID>1</PMID><Article>
<Journal><JournalIssue><Volume>12</Volume><Issue>10</Issue></JournalIssue>
<Title>Lancet Neurol</Title></Journal>
<ArticleTitle>Test article</ArticleTitle>
<Pagination><MedlinePgn>966-77</MedlinePgn></Pagination>
</Article></MedlineCitation></PubmedArticle>
"""

_WITH_START_END = """
<PubmedArticle><MedlineCitation><PMID>2</PMID><Article>
<Journal><JournalIssue><Volume>5</Volume></JournalIssue><Title>J</Title></Journal>
<ArticleTitle>T</ArticleTitle>
<Pagination><StartPage>100</StartPage><EndPage>110</EndPage></Pagination>
</Article></MedlineCitation></PubmedArticle>
"""


def test_volume_issue_and_medline_pgn():
    p = parse_one(_WITH_MEDLINE_PGN)
    assert p is not None
    assert p.volume == "12"
    assert p.issue == "10"
    assert p.pages == "966-77"


def test_start_end_page_fallback():
    p = parse_one(_WITH_START_END)
    assert p is not None
    assert p.volume == "5"
    assert p.issue is None
    assert p.pages == "100-110"
