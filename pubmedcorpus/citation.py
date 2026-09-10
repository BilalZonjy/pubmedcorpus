"""Render a Vancouver/AMA-style reference from parsed metadata.

Pure and offline: it formats data the parser already got out of the paper's own
XML, never fetching anything. This is the whole "citation library" — a small
deterministic formatter — and it is what lets the generative layer cite by key
while we render the real bibliography ourselves, so a model can never invent a
reference.

**Called once per paper, when PMC full text promotes it**, and the result is frozen
into `paper.citation`; readers use that column rather than re-formatting. `ingest
reparse` re-runs it over the stored `abstract.pubmed_xml` when the format changes.

`format_reference` takes any object exposing the `ParsedPaper` fields (title,
journal, volume, issue, pages, pub_year, doi, pmid) and an iterable of
`ParsedAuthor`-like objects (position, last_name, fore_name, initials,
collective_name). Duck-typed on attributes so tests can pass lightweight stubs —
which also means a renamed field degrades to a silently missing author list rather
than an error; `test_store.py` guards the real call site against exactly that.
"""

# Vancouver lists up to six authors, then "et al".
_MAX_NAMES = 6


def _author_name(a) -> str | None:
    """One author as "Lastname II" (initials, no periods), or a collective name."""
    collective = getattr(a, "collective_name", None)
    if collective and collective.strip():
        return collective.strip()

    last = (getattr(a, "last_name", None) or "").strip()
    if not last:
        return None

    initials = (getattr(a, "initials", None) or "").strip()
    if not initials:
        fore = (getattr(a, "fore_name", None) or "").strip()
        initials = "".join(part[0] for part in fore.split() if part)
    initials = initials.replace(".", "").replace(" ", "")

    return f"{last} {initials}".strip() if initials else last


def format_authors(authors, max_names: int = _MAX_NAMES) -> str:
    """Author list in citation order, truncated to `max_names` then "et al"."""
    ordered = sorted(authors, key=lambda a: getattr(a, "position", 0))
    names = [n for a in ordered if (n := _author_name(a))]
    if not names:
        return ""
    if len(names) > max_names:
        return ", ".join(names[:max_names]) + ", et al"
    return ", ".join(names)


def _ensure_period(text: str) -> str:
    text = text.strip()
    return text if text[-1:] in ".?!" else text + "."


def _locator(paper) -> str:
    """Year;Volume(Issue):Pages — omitting whatever's missing."""
    year = str(paper.pub_year) if getattr(paper, "pub_year", None) else ""
    volume = (getattr(paper, "volume", None) or "").strip()
    issue = (getattr(paper, "issue", None) or "").strip()
    pages = (getattr(paper, "pages", None) or "").strip()

    s = year
    if volume:
        s += (";" if s else "") + volume
    if issue:
        s += f"({issue})"
    if pages:
        s += (":" if (volume or issue) else (";" if year else "")) + pages
    return s


def _identifier(paper) -> str:
    doi = getattr(paper, "doi", None)
    if doi and doi.strip():
        return f"doi:{doi.strip()}"
    pmid = getattr(paper, "pmid", None)
    if pmid:
        return f"PMID: {pmid}"
    return ""


def pubmed_url(pmid: str) -> str:
    """The PubMed page for a pmid. **The one definition of that template.**

    Three outputs carry this link — the markdown export, the JSON export and the HTML pages —
    and each used to build it from its own f-string. A URL that reads three ways is a URL that
    can disagree with itself; there is nothing to decide here, so there is one function.

    Takes the pmid rather than a paper object, unlike `reference_url` below: the inventory holds
    pmid strings and has no paper to pass.
    """
    return f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"


def reference_url(paper) -> str | None:
    """A link readers can follow to confirm the reference is real.

    Prefers the PubMed page (every corpus paper has a PMID), falling back to the
    DOI resolver. None only if the paper somehow has neither.
    """
    pmid = getattr(paper, "pmid", None)
    if pmid:
        return pubmed_url(pmid)
    doi = getattr(paper, "doi", None)
    if doi and doi.strip():
        return f"https://doi.org/{doi.strip()}"
    return None


def format_reference(paper, authors=()) -> str:
    """A single reference string: Authors. Title. Journal. Year;Vol(Issue):Pages. id."""
    parts: list[str] = []

    author_str = format_authors(authors)
    if author_str:
        parts.append(_ensure_period(author_str))

    title = (getattr(paper, "title", None) or "").strip()
    if title:
        parts.append(_ensure_period(title))

    journal = (getattr(paper, "journal", None) or "").strip().rstrip(".")
    if journal:
        parts.append(f"{journal}.")

    locator = _locator(paper)
    if locator:
        parts.append(f"{locator}.")

    ident = _identifier(paper)
    if ident:
        parts.append(ident)

    return " ".join(parts).strip()
