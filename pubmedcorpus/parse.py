"""PubMed XML → structured records.

Parsed with lxml rather than Entrez.read: we need precise control over the
awkward parts (structured abstracts, OtherAbstract, collective authorship,
MedlineDate) and we keep the original XML verbatim alongside the parsed fields.
"""

import re
from dataclasses import dataclass, field
from datetime import date

from lxml import etree

MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

# The retracted article itself carries this type. ("Retraction of Publication"
# is the separate notice article and must not be confused with it.)
RETRACTED_TYPE = "retracted publication"

_ORCID_RE = re.compile(r"(\d{4}-\d{4}-\d{4}-\d{3}[\dXx])")


@dataclass
class ParsedAuthor:
    position: int
    is_last: bool = False
    last_name: str | None = None
    fore_name: str | None = None
    initials: str | None = None
    collective_name: str | None = None
    affiliation: str | None = None
    orcid: str | None = None
    canonical_key: str | None = None


@dataclass
class ParsedPaper:
    pmid: str
    raw_xml: str
    title: str | None = None
    journal: str | None = None
    volume: str | None = None
    issue: str | None = None
    pages: str | None = None
    doi: str | None = None
    pmc_id: str | None = None
    pub_date: date | None = None
    pub_year: int | None = None
    pub_date_raw: str | None = None
    entry_date: date | None = None
    abstract_text: str | None = None
    other_abstract_text: str | None = None
    publication_types: list[str] = field(default_factory=list)
    mesh_terms: list[str] = field(default_factory=list)
    is_retracted: bool = False
    authors: list[ParsedAuthor] = field(default_factory=list)


def _text(el) -> str | None:
    """Flatten an element's text including inline markup (<i>, <sup>, ...)."""
    if el is None:
        return None
    s = "".join(el.itertext()).strip()
    return s or None


def normalize_orcid(value: str | None) -> str | None:
    """ORCID arrives bare, as a URL, or occasionally without hyphens."""
    if not value:
        return None
    m = _ORCID_RE.search(value.replace(" ", ""))
    if m:
        return m.group(1).upper()
    digits = re.sub(r"[^0-9Xx]", "", value)
    if len(digits) == 16:
        return f"{digits[0:4]}-{digits[4:8]}-{digits[8:12]}-{digits[12:16]}".upper()
    return None


def canonical_key(last_name: str | None, fore_name: str | None, initials: str | None) -> str | None:
    """Blocking key: lastname + first initial, lowercased.

    Deliberately coarse — it groups candidates, it does not decide identity.
    "Wang J" collapses many people into one key on purpose.

    Nothing consumes this any more: it existed for the author-disambiguation step that
    fed the belief-trajectory view, and the project writes reviews instead. It is still
    computed because it costs nothing and rides along in `abstract.pubmed_json`, so an
    author-level question asked later starts from a key rather than from raw names.
    """
    if not last_name:
        return None
    initial = ""
    if fore_name:
        initial = fore_name.strip()[:1]
    elif initials:
        initial = initials.strip()[:1]
    return f"{last_name.strip().lower()}|{initial.lower()}"


def _parse_pub_date(article) -> tuple[date | None, int | None, str | None]:
    pubdate = article.find("./Journal/JournalIssue/PubDate")
    if pubdate is None:
        return None, None, None

    medline = _text(pubdate.find("MedlineDate"))
    if medline:
        # e.g. "1998 Jan-Feb", "1999-2000" — no reliable single date, so keep
        # the string and take the leading year only.
        m = re.search(r"(\d{4})", medline)
        year = int(m.group(1)) if m else None
        return None, year, medline

    y = _text(pubdate.find("Year"))
    mo = _text(pubdate.find("Month"))
    d = _text(pubdate.find("Day"))
    raw = " ".join(x for x in (y, mo, d) if x) or None
    if not y:
        return None, None, raw

    year = int(y)
    month = 1
    if mo:
        month = MONTHS.get(mo[:3].lower(), 0) or (int(mo) if mo.isdigit() else 1)
    day = int(d) if d and d.isdigit() else 1
    try:
        return date(year, month, day), year, raw
    except ValueError:
        return None, year, raw


def _parse_pages(article) -> str | None:
    """Page range for a citation. Prefer MedlinePgn (e.g. "966-77"), else start/end."""
    pagination = article.find("Pagination")
    if pagination is None:
        return None
    pgn = _text(pagination.find("MedlinePgn"))
    if pgn:
        return pgn
    start = _text(pagination.find("StartPage"))
    end = _text(pagination.find("EndPage"))
    if start and end:
        return f"{start}-{end}"
    return start or None


def _parse_entry_date(pubmed_data) -> date | None:
    if pubmed_data is None:
        return None
    # "entrez" is when the record entered PubMed; fall back to "pubmed".
    for status in ("entrez", "pubmed"):
        el = pubmed_data.find(f"./History/PubMedPubDate[@PubStatus='{status}']")
        if el is None:
            continue
        y, m, d = _text(el.find("Year")), _text(el.find("Month")), _text(el.find("Day"))
        if not y:
            continue
        try:
            return date(int(y), int(m or 1), int(d or 1))
        except ValueError:
            continue
    return None


def _join_abstract(parent) -> str | None:
    """Assemble AbstractText children.

    Structured abstracts arrive as several labelled children, not one blob.
    Labels are preserved because they carry meaning ("CONCLUSIONS:" is where
    positions get stated).
    """
    if parent is None:
        return None
    parts: list[str] = []
    for at in parent.findall("AbstractText"):
        body = _text(at)
        if not body:
            continue
        label = at.get("Label") or at.get("NlmCategory")
        parts.append(f"{label.strip()}: {body}" if label else body)
    return "\n\n".join(parts) or None


def _parse_authors(article) -> list[ParsedAuthor]:
    author_list = article.find("AuthorList")
    if author_list is None:
        return []
    els = author_list.findall("Author")
    out: list[ParsedAuthor] = []
    for i, el in enumerate(els, start=1):
        last = _text(el.find("LastName"))
        fore = _text(el.find("ForeName"))
        initials = _text(el.find("Initials"))
        collective = _text(el.find("CollectiveName"))

        orcid = None
        for ident in el.findall("Identifier"):
            if (ident.get("Source") or "").upper() == "ORCID":
                orcid = normalize_orcid(_text(ident))
                if orcid:
                    break

        aff_el = el.find("./AffiliationInfo/Affiliation")
        affiliation = _text(aff_el) if aff_el is not None else None

        out.append(
            ParsedAuthor(
                position=i,
                is_last=(i == len(els) and len(els) > 1),
                last_name=last,
                fore_name=fore,
                initials=initials,
                collective_name=collective,
                affiliation=affiliation,
                orcid=orcid,
                canonical_key=canonical_key(last, fore, initials),
            )
        )
    return out


def parse_article(el) -> ParsedPaper | None:
    """Parse one <PubmedArticle>."""
    citation = el.find("MedlineCitation")
    if citation is None:
        return None
    pmid = _text(citation.find("PMID"))
    if not pmid:
        return None

    article = citation.find("Article")
    pubmed_data = el.find("PubmedData")

    raw_xml = etree.tostring(el, encoding="unicode")
    paper = ParsedPaper(pmid=pmid, raw_xml=raw_xml)

    if article is not None:
        paper.title = _text(article.find("ArticleTitle"))
        paper.journal = _text(article.find("./Journal/Title")) or _text(
            citation.find("./MedlineJournalInfo/MedlineTA")
        )
        issue_el = article.find("./Journal/JournalIssue")
        if issue_el is not None:
            paper.volume = _text(issue_el.find("Volume"))
            paper.issue = _text(issue_el.find("Issue"))
        paper.pages = _parse_pages(article)
        paper.pub_date, paper.pub_year, paper.pub_date_raw = _parse_pub_date(article)
        paper.abstract_text = _join_abstract(article.find("Abstract"))
        paper.authors = _parse_authors(article)

        for eloc in article.findall("ELocationID"):
            if (eloc.get("EIdType") or "").lower() == "doi":
                paper.doi = _text(eloc)
                break

        paper.publication_types = [
            t for t in (_text(pt) for pt in article.findall("./PublicationTypeList/PublicationType")) if t
        ]

    # OtherAbstract sits outside <Abstract> — non-English or publisher-supplied.
    # Kept separate rather than merged: different provenance, often different
    # language, and merging would silently mix them in extraction.
    others = [_join_abstract(oa) for oa in citation.findall("OtherAbstract")]
    paper.other_abstract_text = "\n\n".join(o for o in others if o) or None

    paper.mesh_terms = [
        t
        for t in (
            _text(dn) for dn in citation.findall("./MeshHeadingList/MeshHeading/DescriptorName")
        )
        if t
    ]

    paper.is_retracted = any(pt.strip().lower() == RETRACTED_TYPE for pt in paper.publication_types)

    if pubmed_data is not None:
        paper.entry_date = _parse_entry_date(pubmed_data)
        for aid in pubmed_data.findall("./ArticleIdList/ArticleId"):
            id_type = (aid.get("IdType") or "").lower()
            value = _text(aid)
            if id_type == "doi" and not paper.doi:
                paper.doi = value
            elif id_type == "pmc" and value:
                paper.pmc_id = value if value.upper().startswith("PMC") else f"PMC{value}"

    return paper


def parse_one(xml: bytes | str) -> ParsedPaper | None:
    """Parse a single stored `<PubmedArticle>` string back into a ParsedPaper.

    The inverse of what ingest kept in `Abstract.pubmed_xml` (which is
    `etree.tostring` of one PubmedArticle element). Used by `ingest reparse` to
    backfill from the verbatim source with no PubMed round-trip, and by `pmc-fetch`
    when it promotes a record and needs the citation.
    """
    if isinstance(xml, str):
        xml = xml.encode("utf-8")
    parser = etree.XMLParser(recover=True, huge_tree=True)
    el = etree.fromstring(xml, parser=parser)
    if el is None:
        return None
    return parse_article(el)


def parse_batch(xml: bytes | str) -> list[ParsedPaper]:
    """Parse a <PubmedArticleSet> response into records."""
    if isinstance(xml, str):
        xml = xml.encode("utf-8")
    # recover=True: a single malformed record in a 200-record batch should not
    # lose the other 199.
    parser = etree.XMLParser(recover=True, huge_tree=True)
    root = etree.fromstring(xml, parser=parser)
    if root is None:
        return []
    papers = []
    for el in root.findall(".//PubmedArticle"):
        parsed = parse_article(el)
        if parsed:
            papers.append(parsed)
    return papers
