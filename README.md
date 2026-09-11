# pubmedcorpus

Build and maintain a corpus of biomedical papers from PubMed and PMC.

- **Sync from PubMed** with a cursor derived from the data rather than from a run log — so a restored
  database, a fresh one and an interrupted run all resume correctly without bookkeeping to keep in step.
- **Fetch open-access full text** from PMC, resumably: each paper's outcome commits with its batch, so
  an interrupted run resumes by being run again.
- **Parse JATS** into sections whose type is re-derived at read time, not frozen at write time.
- **Snapshot** the whole corpus to JSONL and restore it, because the corpus is the expensive asset.
- **Seed** paywalled text by hand, recorded as such rather than passed off as fetched.

Extracted from a working SUDEP literature-review pipeline, where it ran against ~2,500 papers.

**Postgres only, by construction.** `on_conflict_do_update`, `JSONB`, `TSVECTOR` and a persisted
generated column are all load-bearing. There is no build of this that works on another database.

## Status

**v0.1.1, and honestly 0.x.** One consumer so far, which is the reason the public surface is
deliberately small (see *What it does not do*): a library extracted before it has a second user tends
to abstract the wrong joints, so this promises as little as it can until one exists. Expect the API to
move before 1.0. Pin a tag.

## Versions

### 0.1.1

**`abstract.search_tsv` now indexes MeSH terms**, at weight C — below title (A) and abstract (B),
because a MeSH hit is a curated topical label and weaker evidence of relevance than the paper's own
prose. A query that matches a paper's indexed subject headings now finds it even when the abstract
phrases things differently.

**One caveat if you use phrase search.** The terms are read as one string
(`pubmed_json->>'mesh_terms'`), because a generated column's expression must be immutable and
per-term tokenising would need `unnest`. Punctuation does not break adjacency in a `tsvector`, so
all terms share one positional sequence and `phraseto_tsquery`/`<->` can match across the seam
between two unrelated headings. Plain and `websearch_to_tsquery` matching is unaffected.

### 0.1.0

Initial extraction from the SUDEP review pipeline: sync, PMC fetch, JATS parsing, seeding,
snapshots, and the library's own Alembic branch.

## Installing

```bash
pip install pubmedcorpus
```

If you also intend to run the migrations that create its tables, take the extra — see *Migrations*:

```bash
pip install "pubmedcorpus[migrations]"
```

Or straight from git, pinned, if you want something newer than the last release. **A tag or a commit,
never a branch** — `@main` reinstalls something different every time the branch moves, which is the
exact failure a pinned dependency exists to prevent. Installing this way needs `git` present, which a
slim Docker base image usually lacks:

```
pubmedcorpus @ git+https://github.com/BilalZonjy/pubmedcorpus.git@v0.1.1
```

**Do not also vendor a copy.** If the package exists both in `site-packages` and in your project's own
source tree, the tree wins `sys.path` and the pinned version is decoration — you would be running
something no requirements file describes. The check, in the environment where the code actually runs:

```bash
python -c "import pubmedcorpus; print(pubmedcorpus.__file__)"
```

A path under `site-packages/` is the installed copy. Anything else means something in your tree is
shadowing it.

## What it does not do

Deliberately, because these are the caller's:

- **No CLI.** Argument parsing and deciding which environment variables must be present before a
  command may touch a database are application decisions.
- **No sessions.** Every function that touches the database takes a `Session` you opened. The library
  never opens, closes or wraps one.
- **No environment reading.** The corpus definition arrives as an object you construct, in your source,
  version-controlled. There is no `PUBMED_QUERY` this reads behind your back.
- **No re-exports** from `pubmedcorpus/__init__.py`. Import submodules: `from pubmedcorpus import jats`.
  The public surface stays small until a second consumer shows which joints are real.

## The corpus definition is a constant, not configuration

```python
from pubmedcorpus.config import IngestConfig

CORPUS = IngestConfig(
    query='("sudden unexpected death in epilepsy"[tiab] OR "SUDEP"[tiab])',
    ncbi_email="you@example.org",     # NCBI requires a contact address
    ncbi_tool="my-review-pipeline",
    year_min=1990,
    ncbi_api_key=None,                # raises the rate limit from 3/s to 10/s
)
```

`query` has no default and never will. Every other field does, because there is such a thing as a
neutral batch size and no such thing as a neutral corpus: a default query would be silently wrong for
everyone. Keep it in source rather than in the environment — changing it changes what your corpus *is*,
which is a versioned decision, not a deployment knob.

`sync` records each definition it runs under in a `corpus_config` row, keyed by a hash of the
definition. That is **recording, not enforcement**: it does not refuse a changed query. A corpus mixed
from two definitions stays possible — it just stops being invisible.

## Quickstart

```python
from pubmedcorpus.ncbi import NCBIClient
from pubmedcorpus.sync import sync
from pubmedcorpus.pmc import fetch_fulltext

client = NCBIClient(CORPUS)

with your_session_factory() as session:          # you own the session
    n = sync(session, client)                    # incremental if the corpus has rows, full if empty
    session.commit()

with your_session_factory() as session:
    outcomes = fetch_fulltext(session, client, limit=100)
    # {"ok": 63, "not_oa": 28, "no_pmcid": 7, "failed": 2}
```

`sync(full=True)` forces a full pull, an explicit `since=date(...)` scopes by entry date, and the
default chooses: full when the corpus is empty, incremental from `max(entry_date) - days_back` when it
is not. The cursor is a query over what is actually stored, so a restored snapshot resumes correctly
with no marker to write.

## Schema

Two tables, and **the split is the point**:

| Table | Row means |
|---|---|
| `abstract` | This paper is in the corpus. PubMed XML, its parse, entry date, PMC id, retraction flag. |
| `paper` | We have its full text. JATS, the parsed sections, a rendered citation. |

A `paper` row's **existence is the has-full-text filter** — there is no `fulltext_status` column to
fall out of step with reality. `Abstract.paper` is `None` or it is a body you can read.

Both are on the library's own declarative `Base` (`pubmedcorpus.db`), which is separate from yours so
the two packages cannot collide over a class name.

## Migrations

The library **owns its schema** and ships an Alembic *branch* — not a second history root — labelled
`pubmedcorpus`, with `pubmedcorpus_0001` creating the two tables and `pubmedcorpus_0002` adding
`corpus_config`.

Running them needs Alembic itself: `pip install "pubmedcorpus[migrations]"`. It is an extra rather
than a required dependency because nothing in this package imports Alembic — only the migration
scripts do, and those are run by the `alembic` command, which brings its own copy.

Point your Alembic at the branch alongside your own versions. The second path is where pip put the
package:

```ini
# alembic.ini
version_locations = migrations/versions /path/to/site-packages/pubmedcorpus/migrations/versions
```

Print the correct value rather than guessing at the layout, which moves with the Python version:

```bash
python -c "import pubmedcorpus, pathlib; print(pathlib.Path(pubmedcorpus.__file__).parent / 'migrations' / 'versions')"
```

**It has to be in the ini file, not computed in `env.py`.** Alembic builds its `ScriptDirectory` from
the ini *before* `env.py` runs, so a `config.set_main_option("version_locations", …)` there is read too
late — the branch is silently absent and `alembic heads` quietly shows one head instead of two.

Then `alembic upgrade heads` — **`heads`, not `head`**: with two branches, `head` is ambiguous and
errors. `alembic current` prints two rows, one per branch.

Your models and the library's must **share one `MetaData`** if any of your foreign keys point at
`paper.pmid` or `abstract.pmid`. A string FK target is looked up in the owning table's own metadata, so
two `MetaData` objects means the target never resolves:

```python
# your db.py
from pubmedcorpus.db import Base as CorpusBase

class Base(DeclarativeBase):
    metadata = CorpusBase.metadata      # borrowed: separate registry, shared metadata
```

`pubmedcorpus_0001` detects tables that already exist and no-ops, so it can be applied to a database
that predates the library without dropping anything.

## Full text you cannot fetch

The most important paper in a field is often paywalled, and a corpus that silently omits it produces
reviews with a hole in them. `seed.py` converts operator-supplied text — obtained under your own
access — into the same stored shape a fetched paper has, and records that it is not equivalent:
`fulltext_jats["license"]` says so, `jats_xml` holds a provenance marker instead of JATS, and seeding
logs at WARNING.

PDF text extraction damages text in ways specific to the publisher, so those values are a parameter,
not built in:

```python
from pubmedcorpus.seed import SeedProfile, parse_supplied_text, seed_paper

PROFILE = SeedProfile(
    ligatures={"eff ective": "effective", "signifi cant": "significant"},
    max_heading_chars=30,
    abstract_headings=frozenset({"abstract", "summary"}),
)

payload = parse_supplied_text(open("paper.txt").read(), PROFILE)
with your_session_factory() as session:
    paper = seed_paper(session, "24012372", payload, PROFILE)
```

`SeedProfile()` with no arguments is the neutral case — an empty ligature map means "this text needs no
repairs", which is right for anything not extracted from a PDF.

**Deriving a ligature map.** PDF extraction drops `ff`/`fi`/`ffi`/`fl` ligatures and leaves a space
where the glyph was. List the candidates:

```bash
grep -oE "\b[A-Za-z]*(ff|fi|ffi|fl) [a-z]+" paper.txt | sort -u
```

Then **read the list and add only the genuine splits**. This step cannot be automated, and that is the
whole reason the map is enumerated rather than matched by pattern: `staff regarding` and `staff
compared` are correct English, and nothing about their shape distinguishes them from `off er`. A regex
that joins a token ending in `ff` to the next one produces `staffregarding`.

Seeding requires the `abstract` row to exist already — sync first. A pmid your query never matched is
outside your corpus by definition, and `seed_paper` refuses it rather than working around it.

## Snapshots

```python
from pubmedcorpus.snapshot import export, import_

with your_session_factory() as session:
    export(session, sys.stdout)          # JSONL, one record per line, ordered by pmid

with your_session_factory() as session:
    import_(session, sys.stdin)          # idempotent, additive, commits per batch
```

Import never deletes: records absent from the file are left alone, and a paper that lost its full text
is not un-promoted. `export` **streams from a live cursor**, so the session must stay open until the
output is fully written — closing it early truncates the snapshot silently, producing a file that looks
like a smaller corpus rather than an error.

## Rate limits

NCBI's ceiling is 3 requests/second without an API key and 10 with one. `IngestConfig.rate_limit`
defaults to **1.0/second** — well under either — and `NCBIClient` enforces it itself rather than
trusting callers to sleep, retrying with backoff on the failures NCBI actually returns under load.
Raising it is opt-in and yours to justify; the default is chosen so an unattended nightly job never
becomes the reason an institution gets blocked.

Set `ncbi_email` to a real address: it is how NCBI reaches you before blocking you, and their terms
require it. `ncbi_api_key` is optional, and is excluded from what `corpus_config` records — a secret
does not belong in a provenance row.

## Development

```bash
git clone https://github.com/BilalZonjy/pubmedcorpus.git
cd pubmedcorpus
pip install -e ".[dev]"
pytest
```

**The suite is fully offline** — no database, no network, no fixtures that reach NCBI. Parser tests run
against committed PubMed and PMC XML; the database-touching functions are exercised through fake
sessions that record the statements they were handed. That is deliberate: the failure modes worth
catching here are a mis-classified section or a wrong upsert target, and neither needs Postgres to
show up.

`pytest` also works without installing anything (`pythonpath` is set in `pyproject.toml`), which is the
point — a contributor should be testing the checkout, not an installed copy of it.

## License

MIT. See `LICENSE`.
