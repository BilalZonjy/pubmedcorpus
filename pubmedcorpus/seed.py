"""Hand-supplied full text, for papers the open-access fetch cannot reach.

The PMC fetch retrieves only PMC's open-access subset, and publisher sites are never scraped. That
leaves a real gap: a field's most important paper is often paywalled, so no review built from an
open-access corpus can cite it. This module closes that gap one paper at a time, from text the operator
obtained under their own access.

**A seeded paper is not equivalent to a fetched one**, and the difference is recorded rather than hidden:

- `fulltext_jats["license"]` says so, which puts it in any licence tally over the corpus.
- `jats_xml` holds a provenance marker instead of JATS, because there is none.
- Seeding logs at WARNING.

The honest cost, which a consumer should write into its own documentation: a corpus built this way
cannot be reproduced from a clean checkout, and the paper in it was chosen by a human.

Two jobs, kept separate so the text conversion can be tested with no database: `parse_supplied_text`
turns raw text into the stored JSON shape, `seed_paper` writes it.

**What `SeedProfile` is for.** This module was written for one PDF from one journal, and roughly half of
it was values describing *that* PDF — a hand-derived ligature map, a heading-length cutoff tuned to its
column width, the name that journal gives its abstract. Those are now a parameter. The parsing structure
(`clean_text`, `_unwrap`, `_is_heading`, the heading/body block split) is general; the values are not,
and pretending otherwise is how a library ends up silently correct for exactly one caller.
"""

import json
import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass, field

from pubmedcorpus.jats import OTHER, classify_section
from pubmedcorpus.models import Abstract, Paper
from pubmedcorpus.parse import parse_one
from pubmedcorpus.store import paper_values
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

log = logging.getLogger(__name__)

# Superscript reference markers flattened into the prose: "pathophysiology.1,3",
# "interpretation.4–7", "years.1".
#
# **General, despite where it came from**, which is why it is the shipped default rather than something
# every caller must supply. Requiring **three or more lowercase letters** before the period is what keeps
# it away from numbers: "in 2013.5 units" cannot match, because `2013` is digits. The journal this was
# written for also writes decimals with a middle dot ("5·1", "2·6–9·2"), which is a second line of
# defence and the one its original comment credited — but the letter guard is the load-bearing one and it
# holds for any publisher. A citation left in place is only noise in an embedding; a mangled statistic
# would be a false fact.
DEFAULT_REF_MARKER = re.compile(r"([a-z]{3,})\.\d+(?:[,–-]\d+)*")

# A heading is short. 30 characters suits an ordinary single-column export; text hard-wrapped into
# narrow columns needs it lower, because a great many wrapped fragments are short enough to pass.
DEFAULT_MAX_HEADING_CHARS = 30

# Both names are common enough to default. `"summary"` matters more than it looks: `classify_section`
# maps it to CONCLUSIONS (its pattern covers "Summary and conclusions"), so a journal that heads its
# abstract "Summary" would otherwise get an abstract stored as a body section and extracted from as
# though it were the paper's argument.
DEFAULT_ABSTRACT_HEADINGS = frozenset({"abstract", "summary"})

# What `fulltext_jats["license"]` carries for a seeded paper. Not a licence — the point is that there
# isn't one. **Deliberately says nothing about any particular project's documentation**: this string is
# written into the database, so a default naming a doc path would leave every other consumer's rows
# citing a file they do not have. Override it via `SeedProfile.seeded_license` to point at your own.
DEFAULT_SEEDED_LICENSE = "hand-seeded; NOT open access, publisher copyright"

# `jats_xml` is NOT NULL and there is no JATS for a seeded paper. A marker rather than an empty string,
# so anyone reading the row sees at once that it did not come from PMC.
DEFAULT_SEEDED_JATS = "<!-- hand-seeded from operator-supplied text; no JATS source -->"


@dataclass(frozen=True)
class SeedProfile:
    """How to read one publisher's text export, and what to record about the result.

    **Defaults are the neutral case, not an example.** An empty ligature map means "this text needs no
    repairs", which is right for anything that was not extracted from a PDF — so unlike
    `IngestConfig.query`, a default here is genuinely usable rather than silently wrong. There is no such
    thing as a neutral corpus definition; there is such a thing as neutral text handling.

    `frozen=True` and no `slots=True`, matching `IngestConfig`: on a frozen dataclass, slots adds nothing
    for correctness and turns a mistyped field name into an opaque `TypeError` about `super()` instead of
    a `FrozenInstanceError` that names the field.
    """

    # PDF extraction drops ff/fi/ffi/fl ligatures and leaves a space where the glyph was.
    #
    # **An explicit map, not a regex, and this is load-bearing.** The obvious rule — join a token ending
    # in `ff`/`fi` to the next token — corrupts correct English: "staff regarding" and "staff compared"
    # would fuse into "staffregarding". There is no way to tell those from "off er" by shape alone, so
    # the broken forms are enumerated instead.
    #
    # To derive one for a new paper:
    #   grep -oE "\b[A-Za-z]*(ff|fi|ffi|fl) [a-z]+" <paper>.txt | sort -u
    # then read the list and add only the genuine splits.
    #
    # Order does not matter: where one key is a prefix of another ("eff ect" / "eff ective"), replacing
    # the shorter first still yields the longer word, because the suffix is left untouched.
    ligatures: Mapping[str, str] = field(default_factory=dict)

    # A line is a section heading only if it is short, starts with a capital, is unpunctuated, and
    # classifies to a real section type. **All four guards earn their place.** The capital-letter one is
    # the easiest to think unnecessary: without it a wrapped line like "but this finding" scores as a
    # heading, because `classify_section`'s RESULTS pattern includes "finding" — and the Discussion
    # silently splits in two, its tail becoming a bogus `results` section.
    max_heading_chars: int = DEFAULT_MAX_HEADING_CHARS
    abstract_headings: frozenset[str] = DEFAULT_ABSTRACT_HEADINGS
    ref_marker: re.Pattern[str] = DEFAULT_REF_MARKER

    # Written into the database — see the constants above on why neither default names a doc path.
    seeded_license: str = DEFAULT_SEEDED_LICENSE
    seeded_jats: str = DEFAULT_SEEDED_JATS


# The neutral profile, named so callers and tests can be explicit about using it.
DEFAULT_PROFILE = SeedProfile()


def clean_text(raw: str, profile: SeedProfile = DEFAULT_PROFILE) -> str:
    """Undo the extraction damage: ligature splits and flattened reference markers."""
    for broken, fixed in profile.ligatures.items():
        raw = raw.replace(broken, fixed)
    return profile.ref_marker.sub(r"\1.", raw)


def _unwrap(lines: list[str]) -> str:
    """Join hard-wrapped lines into paragraphs, breaking on blank lines.

    A PDF export wraps mid-sentence. Left as-is the model reads a column of fragments and any embedding
    is computed over one, so this is not cosmetic. No profile: joining on blank lines is universal.
    """
    paragraphs: list[str] = []
    current: list[str] = []
    for line in lines:
        if line.strip():
            current.append(line.strip())
        elif current:
            paragraphs.append(" ".join(current))
            current = []
    if current:
        paragraphs.append(" ".join(current))
    return "\n\n".join(paragraphs)


def _is_heading(line: str, profile: SeedProfile = DEFAULT_PROFILE) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > profile.max_heading_chars:
        return False
    if not stripped[0].isupper():
        return False
    if stripped[-1] in ".,;:":
        return False
    if stripped.lower() in profile.abstract_headings:
        return True
    return classify_section(None, stripped) != OTHER


def parse_supplied_text(
    raw: str, profile: SeedProfile = DEFAULT_PROFILE, *, title: str | None = None
) -> dict:
    """Operator-supplied article text → the `paper.fulltext_jats` payload.

    Returns exactly the shape `jats.ParsedFullText.as_dict()` produces, because that is what a consumer's
    source-selection reads. In particular the section `type` values must be `jats.py`'s constants: a
    Discussion tagged anything else drops out of `POSITION_SECTIONS` and silently demotes the paper to a
    whole-article fallback meant for articles with no identifiable position section at all.

    Layout expected — the shape a PDF text export takes:

        <title>                     first non-blank line, unless `title` is given
        <authors, any number of lines>
        Summary                     (or Abstract) → the `abstract` field, NOT a section
        ...
        Introduction                → a section, classified by `jats.classify_section`
        ...
        Discussion
        ...

    Everything before the first heading is dropped: it is the title and author block, and authorship is
    already on the `Abstract` row's `pubmed_json`.
    """
    lines = clean_text(raw, profile).splitlines()

    doc_title = title
    if doc_title is None:
        doc_title = next((ln.strip() for ln in lines if ln.strip()), None)

    # Split into (heading, body-lines) blocks. The preamble before the first heading is
    # the title/author block and is discarded.
    blocks: list[tuple[str, list[str]]] = []
    for line in lines:
        if _is_heading(line, profile):
            blocks.append((line.strip(), []))
        elif blocks:
            blocks[-1][1].append(line)

    abstract: str | None = None
    sections: list[dict] = []
    for heading, body in blocks:
        text = _unwrap(body)
        if not text:
            continue
        if abstract is None and heading.lower() in profile.abstract_headings:
            abstract = text
            continue
        kind = classify_section(None, heading)
        # A later "Summary"/"Conclusions" is a real closing section, not the abstract.
        sections.append({"type": kind, "title": heading, "text": text})

    return {
        "pmcid": None,
        "license": profile.seeded_license,
        "title": doc_title,
        "abstract": abstract,
        "sections": sections,
    }


class SeedError(RuntimeError):
    """The paper cannot be seeded. Never partially applied."""


def seed_paper(
    session: Session, pmid: str, payload: dict, profile: SeedProfile = DEFAULT_PROFILE
) -> Paper:
    """Write a hand-supplied `fulltext_jats` payload as a `Paper`. Idempotent.

    **Requires the `Abstract` row to exist already.** `Paper.pmid` is a foreign key onto it, and the
    citation is rendered from its stored `pubmed_xml` (via `store.paper_values`) so a seeded paper's
    bibliography entry is byte-identical to a fetched one's. Sync first; a pmid the corpus query never
    matched cannot be seeded, and that is a refusal rather than something to work around — a paper
    outside the query is outside the corpus by definition.
    """
    record = session.get(Abstract, pmid)
    if record is None:
        raise SeedError(
            f"pmid {pmid} has no abstract row: run a sync first. If sync has run, "
            f"this paper does not match the corpus query and should not be seeded."
        )
    if not payload.get("sections"):
        raise SeedError(f"pmid {pmid}: payload has no sections — nothing to extract from.")

    pubmed = parse_one(record.pubmed_xml)
    if pubmed is None:
        raise SeedError(f"pmid {pmid}: stored pubmed_xml did not parse; cannot render a citation.")

    values = {"pmid": pmid, "jats_xml": profile.seeded_jats, "fulltext_jats": payload}
    values.update(paper_values(pubmed))

    stmt = insert(Paper).values(**values)
    stmt = stmt.on_conflict_do_update(
        index_elements=[Paper.pmid],
        set_={k: v for k, v in values.items() if k != "pmid"},
    )
    session.execute(stmt)
    # The record may have been marked failed by an earlier fetch attempt; it is no longer
    # true that we have no text for it.
    record.fetch_failed = False

    log.warning(
        "pmid %s: HAND-SEEDED from supplied text (%d section(s), %s) — not fetched from PMC.",
        pmid, len(payload["sections"]),
        ", ".join(s["type"] for s in payload["sections"]),
    )
    return session.get(Paper, pmid)


def load_payload(path: str) -> dict:
    """Read a converted payload from disk, failing on the shape rather than at insert."""
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    missing = {"title", "abstract", "sections", "license"} - set(payload)
    if missing:
        raise SeedError(f"{path}: payload is missing {', '.join(sorted(missing))}")
    return payload
