"""What a corpus is built from — one frozen object, passed in rather than read from the environment.

**The guarantee this replaces, and where it went.** The project this was extracted from hardcoded its
query as a module constant with a comment worth carrying over verbatim: *"changing it changes what the
project is, so it's a versioned decision in code — not an env tuning knob."* It was deliberately not an
environment variable, so the corpus definition could not drift.

A library cannot keep that guarantee itself — it has no query to hardcode. So the guarantee **moves to
the consumer**, and the documented pattern is a module-level constant in the consuming project's source:

    # in your project, in source, not read from the environment
    CORPUS_QUERY = 'SUDEP OR "sudden unexpected death in epilepsy"'

    def corpus_config() -> IngestConfig:
        return IngestConfig(
            query=CORPUS_QUERY,
            ncbi_email=settings.ncbi_email,   # a credential: environment
            ncbi_tool="my-project",
        )

Then a change to what the corpus *is* remains a diff, while credentials stay where credentials belong.
Reading the query from the environment becomes something a consumer opts into deliberately rather than
the default shape.

**Three fields are required and none of them has a default.** `query` is obvious — a library shipping a
default corpus would be silently wrong for every consumer. `ncbi_email` and `ncbi_tool` are required for
a subtler reason: NCBI's terms want both, and the alternative was a default of `""` with validation that
rejects blanks, which is a default that always fails. Requiring them means the error names the
constructor call that forgot the field, rather than reporting a blank string from somewhere.

`frozen=True` is what "avoid accidental changes" means at runtime: no mutation after construction, and a
typo'd attribute raises `FrozenInstanceError` naming the field rather than silently creating a new one.
The version-controlled constant covers the rest.

**`slots=True` was specified alongside it and is deliberately not used.** On a *frozen* dataclass it
adds nothing for correctness — frozen already rejects both cases — and it makes one of them
considerably worse. `@dataclass(slots=True)` constructs a replacement class, while the generated
`__setattr__` still closes over the original; so setting a non-field attribute reaches an invalid
`super()` call and raises `TypeError: super(type, obj): obj must be an instance or subtype of type`
instead of naming the field. A few bytes per instance is not worth handing that to whoever mistypes a
field name, and there is exactly one of these objects per process.
"""

import hashlib
import json
from dataclasses import dataclass, fields


@dataclass(frozen=True)
class IngestConfig:
    """One corpus definition plus the credentials and throughput to fetch it.

    Construct it once and pass it to `NCBIClient`; `sync` and the PMC fetch read everything they need
    from `client.config` rather than taking their own copies, so there is exactly one statement of what
    the corpus is per run.
    """

    # --- required: no sensible default exists for any of these -------------------------
    query: str
    # NCBI's terms require a contact address — they use it to reach you before blocking. A credential in
    # the sense that matters here: it identifies the caller, so it comes from the environment.
    ncbi_email: str
    # The `tool` parameter NCBI asks every E-utilities client to send. Required rather than defaulted
    # because any default a library picked would misattribute the consumer's traffic.
    ncbi_tool: str

    # --- the rest of the corpus definition: these three also decide which papers land --
    year_min: int | None = None
    year_max: int | None = None
    max_records: int | None = None

    # --- throughput ---------------------------------------------------------------------
    batch_size: int = 200
    pmc_batch_size: int = 20
    # Requests/sec. The default is deliberately well under NCBI's ceiling (3/sec, 10 with a key):
    # Biopython throttles underneath regardless, so raising this buys nothing until the corpus is large.
    rate_limit: float = 1.0

    # --- optional credential ------------------------------------------------------------
    # `None` means no key, which is a supported mode — it lowers NCBI's ceiling from 10/sec to 3/sec.
    ncbi_api_key: str | None = None

    def __post_init__(self) -> None:
        """Fail at construction rather than at the first NCBI call or the first empty result.

        Everything here is a misconfiguration that would otherwise surface far from its cause: a blank
        query returns nothing and looks like an empty corpus, inverted year bounds return nothing and
        look the same, and a zero batch size loops forever fetching nothing.
        """
        for name in ("query", "ncbi_email", "ncbi_tool"):
            if not (getattr(self, name) or "").strip():
                raise ValueError(f"IngestConfig.{name} is required and must not be blank")

        # **Rejected, not coerced to None.** Coercing inside a frozen dataclass needs
        # `object.__setattr__`, which quietly undermines the `frozen=True` guarantee that is half the
        # reason for this shape. A consumer that reads the key from an environment variable should
        # normalise `MY_KEY=` to `None` on its side, where the empty string came from.
        if self.ncbi_api_key is not None and not self.ncbi_api_key.strip():
            raise ValueError(
                "IngestConfig.ncbi_api_key is blank — pass None for 'no key' rather than an "
                "empty string, so 'unset' and 'set to nothing' cannot be confused"
            )

        if self.year_min is not None and self.year_max is not None and self.year_min > self.year_max:
            raise ValueError(
                f"IngestConfig year range is inverted: {self.year_min} > {self.year_max}"
            )

        for name in ("batch_size", "pmc_batch_size"):
            if getattr(self, name) <= 0:
                raise ValueError(f"IngestConfig.{name} must be positive, got {getattr(self, name)}")

        if self.rate_limit < 0:
            raise ValueError(f"IngestConfig.rate_limit must not be negative, got {self.rate_limit}")

    # --- what gets recorded ---------------------------------------------------------------

    # Every field except the API key, which is a credential and does not belong in a table meant to be
    # read. Note what recording all of the rest costs, accepted deliberately: only `query`, `year_min`,
    # `year_max` and `max_records` decide which papers land, so **tuning `batch_size` writes a new
    # row**. The table is a full audit of what the library was invoked with, and a reader tells a
    # throughput change from a corpus change by looking at those four.
    _EXCLUDED_FROM_RECORD = frozenset({"ncbi_api_key"})

    def recorded(self) -> dict:
        """The fields written to `corpus_config`, and the fields the hash covers.

        **One method for both, because they are the pair most likely to drift apart.** If the hash were
        computed over a different field set than the row, two genuinely different definitions could
        collide on one hash — and since the insert is `on_conflict_do_nothing`, the database would
        silently keep the first and never record the second.
        """
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.name not in self._EXCLUDED_FROM_RECORD
        }

    def config_hash(self) -> str:
        """sha256 of the canonical JSON of `recorded()` — the `corpus_config` primary key.

        **A hash rather than a UNIQUE constraint over the columns**, because Postgres treats NULLs as
        distinct: two rows with `year_min IS NULL` would both be admitted by a UNIQUE, and null year
        bounds are the common case. `sort_keys` is what makes the digest independent of field order.
        """
        canonical = json.dumps(self.recorded(), sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
