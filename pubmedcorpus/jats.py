"""PMC JATS XML → section-structured full text.

Structure is preserved rather than flattened, because *where* a claim appears
decides how much it is worth: Discussion and Conclusion are where authors state
positions, while Methods and Results are largely noise for the belief schema.
Extraction targets sections, so storage has to keep them.

Only the open-access subset has a retrievable body. PMC does not fail for the
rest — it returns front matter with no body — so "not open access" is a parse
outcome here, not a network error.
"""

import re
from dataclasses import dataclass, field

from lxml import etree

XLINK = "http://www.w3.org/1999/xlink"

# Canonical section types. Deliberately coarse: the only distinction that
# changes downstream behaviour is "did the author state a position here".
INTRO = "intro"
METHODS = "methods"
RESULTS = "results"
DISCUSSION = "discussion"
CONCLUSIONS = "conclusions"
BACK_MATTER = "back_matter"
OTHER = "other"

# The sections extraction reads (the abstract is deliberately excluded — see
# prompt.select_source). Order matters — conclusions first, being the most
# concentrated statement of position.
POSITION_SECTIONS = (CONCLUSIONS, DISCUSSION)

# Matched against JATS @sec-type and, failing that, the section <title>. Order
# matters: BACK_MATTER must come before DISCUSSION/CONCLUSIONS so "Conflicts of
# interest" doesn't fall through to a content pattern; "materials and methods"
# must hit METHODS before "results" can match a trailing word, so the more
# specific patterns come first.
_SECTION_PATTERNS: tuple[tuple[str, str], ...] = (
    (BACK_MATTER, r"acknowledg|reference|bibliograph|funding|financial support|"
                  r"conflict|competing interest|disclosure|ethic|consent|"
                  r"abbreviation|author contribution|data availab|supplementary"),
    (DISCUSSION, r"discussion"),
    # "conclud" (not just "conclusion") catches "concluding remarks"; "final
    # (consideration|remark|comment)" catches the non-IMRaD heading a topically
    # organised review closes with instead of "Conclusions".
    (CONCLUSIONS, r"conclu|summary|closing|final (consideration|remark|comment)"),
    # No longer matches bare "patient"/"subject"/"participant" — that let a
    # topical heading like "SUDEP in Patients with Refractory Epilepsy"
    # misclassify as methods. "Patients and Methods" still matches on "method".
    (METHODS, r"method|material|procedure|design|statistic"),
    (RESULTS, r"result|finding|outcome"),
    (INTRO, r"intro|background|objective|rationale"),
)


@dataclass
class Section:
    type: str
    title: str | None
    text: str

    def as_dict(self) -> dict:
        return {"type": self.type, "title": self.title, "text": self.text}


@dataclass
class ParsedFullText:
    pmcid: str | None = None
    pmid: str | None = None
    license: str | None = None
    title: str | None = None
    abstract: str | None = None
    sections: list[Section] = field(default_factory=list)
    # The verbatim <article> element, stored as `paper.jats_xml`. Section parsing is
    # lossy by design — anything outside <sec>/<p> (tables, figure captions, boxed
    # text) never reaches `sections` — so without this the only way back to it is
    # asking PMC again. Same raw-beside-parsed rule the PubMed record follows.
    raw_xml: str = ""

    @property
    def has_body(self) -> bool:
        return any(s.text for s in self.sections)

    def as_dict(self) -> dict:
        """The JSONB payload stored on paper.fulltext_jats.

        Deliberately excludes `raw_xml`: it has its own column, and nesting it would
        be a second copy of the largest field on the row.
        """
        return {
            "pmcid": self.pmcid,
            "license": self.license,
            "title": self.title,
            "abstract": self.abstract,
            "sections": [s.as_dict() for s in self.sections],
        }

    def section_text(self, *types: str) -> str | None:
        """Concatenate sections of the given types, in document order."""
        wanted = set(types)
        parts = [s.text for s in self.sections if s.type in wanted and s.text]
        return "\n\n".join(parts) or None


def classify_section(sec_type: str | None, title: str | None) -> str:
    """Map a JATS section to one of the canonical types.

    @sec-type is authoritative where publishers set it, but a large share of
    articles omit it entirely, so the title is the fallback rather than a
    tie-breaker.
    """
    for source in (sec_type, title):
        if not source:
            continue
        haystack = source.lower()
        for canonical, pattern in _SECTION_PATTERNS:
            if re.search(pattern, haystack):
                return canonical
    return OTHER


def effective_type(section: dict) -> str:
    """The type a **stored** section (a `fulltext_jats["sections"]` entry) should be
    read as — the current classifier applied to data parsed under older patterns.

    Section classification is deliberately NOT persisted-and-replayed: `title` is stored
    alongside `type`, so widening a pattern here takes effect on the existing corpus the
    moment this code changes, with no re-parse command to run and no way for the stored
    data to silently lag the classifier.

    The stored value wins whenever it is not `OTHER`, and that asymmetry is the whole
    trick. `parse_article` classifies from `@sec-type` first and the title second, but
    only the *result* survives into the JSON — the `@sec-type` itself is gone. So
    recomputing from the title alone would demote a section the publisher had labelled
    (`@sec-type="discussion"` under a title like "General remarks" reads as `discussion`
    today, but as `OTHER` from the title). Recomputing **only** the `OTHER` case cannot
    hit that: a section is `OTHER` precisely because neither `@sec-type` nor the title
    matched, so there is no lost `@sec-type` to preserve. The upgrade is monotonic — it
    can move a section out of `OTHER`, never into it — so no paper that reads correctly
    today can regress.

    What this cannot fix is a change to which sections exist or what text they hold;
    that needs the raw `paper.jats_xml`, which is still stored against the day one is
    wanted.
    """
    stored = section.get("type")
    if stored and stored != OTHER:
        return stored
    return classify_section(None, section.get("title"))


def _text(el) -> str | None:
    if el is None:
        return None
    s = " ".join("".join(el.itertext()).split())
    return s or None


def _paragraphs(sec) -> str:
    """All prose under a section, including nested subsections.

    Subsections are folded into their parent rather than emitted separately: a
    "Discussion > Limitations" subsection is still discussion, and splitting it
    out would make section-targeted extraction miss half the text.
    """
    parts: list[str] = []
    for p in sec.iter("p"):
        body = _text(p)
        if body:
            parts.append(body)
    return "\n\n".join(parts)


def _license_label(article_meta) -> str | None:
    """A short, comparable licence label.

    Stored because OA terms differ and some are non-commercial — a licence a
    downstream reader must be able to check without re-fetching the article.
    """
    if article_meta is None:
        return None
    lic = article_meta.find(".//permissions/license")
    if lic is None:
        return None

    href = lic.get(f"{{{XLINK}}}href") or ""
    m = re.search(r"creativecommons\.org/licenses/([a-z\-]+)", href, re.I)
    if m:
        return f"CC {m.group(1).upper()}"
    if "creativecommons.org/publicdomain/zero" in href.lower():
        return "CC0"

    declared = lic.get("license-type")
    if declared:
        return declared.strip()[:100]

    body = _text(lic)
    return body[:100] if body else None


def _article_id(article_meta, id_type: str) -> str | None:
    if article_meta is None:
        return None
    el = article_meta.find(f'.//article-id[@pub-id-type="{id_type}"]')
    return _text(el)


def parse_article(article) -> ParsedFullText:
    """Parse one <article> element."""
    meta = article.find(".//front/article-meta")

    pmcid = _article_id(meta, "pmc")
    out = ParsedFullText(
        pmcid=f"PMC{pmcid}" if pmcid and not pmcid.upper().startswith("PMC") else pmcid,
        pmid=_article_id(meta, "pmid"),
        license=_license_label(meta),
        title=_text(meta.find(".//title-group/article-title")) if meta is not None else None,
        raw_xml=etree.tostring(article, encoding="unicode"),
    )

    if meta is not None:
        # PMC's abstract can differ from PubMed's (publisher-supplied vs
        # indexed). Kept alongside rather than replacing it.
        out.abstract = _text(meta.find(".//abstract"))

    body = article.find("body")
    if body is None:
        return out

    sections = body.findall("sec")
    if not sections:
        # Some articles — especially editorials and letters, which matter here —
        # have unsectioned prose. Losing them would drop exactly the commentary
        # extraction needs.
        text = _paragraphs(body)
        if text:
            out.sections.append(Section(type=OTHER, title=None, text=text))
        return out

    for sec in sections:
        title = _text(sec.find("title"))
        out.sections.append(
            Section(
                type=classify_section(sec.get("sec-type"), title),
                title=title,
                text=_paragraphs(sec),
            )
        )
    return out


def parse_pmc_batch(xml: bytes | str) -> list[ParsedFullText]:
    """Parse a <pmc-articleset> response.

    recover=True for the same reason as PubMed parsing: one malformed article
    in a batch must not cost the rest.
    """
    if isinstance(xml, str):
        xml = xml.encode("utf-8")
    parser = etree.XMLParser(recover=True, huge_tree=True)
    root = etree.fromstring(xml, parser=parser)
    if root is None:
        return []
    articles = root.iter("article") if root.tag != "article" else [root]
    return [parse_article(a) for a in articles]
