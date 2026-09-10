"""The library's declarative base — and nothing else.

**`Base` only: no engine, no `sessionmaker`, no `session_scope`.** Every write path in this library
already takes a `Session` as a parameter (`store.py`, `seed.py`, `pmc.py`), so an engine here would be
a second connection pool against one Postgres that no caller asked for. A library that constructs its
own engine also has to be told a URL at import time, which is exactly the coupling the extraction is
undoing.

**Two registries, one `MetaData`.** The consuming application declares its own `DeclarativeBase` but
borrows this `metadata`:

    from pubmedcorpus.db import Base as CorpusBase

    class Base(DeclarativeBase):
        metadata = CorpusBase.metadata

That is deliberate and it is the Django shape. Django has no `MetaData` equivalent at all — one global
app registry, cross-app foreign keys written as strings and resolved through it — yet every app still
owns its own `migrations/` directory with its own history. Per-app migrations and per-app *namespaces*
are orthogonal, and Django takes the first without the second precisely so foreign keys always resolve.

Sharing the `MetaData` is what makes a consumer's `ForeignKey("paper.pmid")` resolvable: a string target
is looked up in the *owning table's own* metadata, so two `MetaData` objects would leave every
cross-boundary foreign key permanently dangling — and `create_all`, `sorted_tables` and Alembic's
autogenerate ordering all need that resolution. Separate registries are still worth keeping: they scope
`relationship()` name resolution, so the library and the application cannot collide over a class name.

The library owns its schema history in `pubmedcorpus/migrations/`, listed in the consumer's Alembic
`version_locations` as a second branch. One `alembic_version` table, two heads — so it is
`alembic upgrade heads`, not `head`.

**Postgres-only, by construction** rather than by accident: `JSONB`, `TSVECTOR`, a persisted generated
column and `on_conflict_do_update` are all load-bearing in the models and the write path.
"""

from sqlalchemy import MetaData
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    # An explicit `MetaData()` rather than the implicit one, because it is a shared object: the
    # consuming application's `Base` borrows it, so it deserves to be visible here.
    metadata = MetaData()
