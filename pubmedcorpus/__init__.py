"""Building a corpus of biomedical papers from PubMed and PMC.

**Being extracted from the SUDEP review pipeline it was written for**, one step at a time, and it
lives inside `backend/` while that happens: `PYTHONPATH` is already `backend/`, so `import
pubmedcorpus` resolves with no install step, no editable install and no bind mount. When the tests
are green and the application consumes it, `git subtree split` moves the whole folder — history
included — into its own repository.

What it is for: PubMed sync with a cursor derived from the data rather than a run log, PMC
open-access full-text fetch, JATS parsing whose section classification is re-derived at read time,
and a corpus schema where a row's existence *is* the has-full-text filter.

**No re-exports here, deliberately.** Callers import submodules — `from pubmedcorpus import jats` —
so the public surface stays small enough that the next project to use this can reshape it without a
deprecation cycle. A library extracted before it has a second consumer tends to abstract the wrong
joints; the mitigation is to promise as little as possible until one exists.

**Postgres-only, by construction**, not by accident: `on_conflict_do_update`, `JSONB`, `TSVECTOR` and
a persisted generated column are all load-bearing.
"""
