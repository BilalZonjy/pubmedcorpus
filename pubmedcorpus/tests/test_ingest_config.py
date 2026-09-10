"""`IngestConfig` — validation at construction, and the hash that keys `corpus_config`.

**Named for the class, not for `pubmedcorpus/config.py`, and deliberately so.** `backend/tests/` already
has a `test_config.py` covering `sudep.config`, and neither test directory is a package, so pytest
derives module names from the basename alone — two `test_config.py` files collide at collection with an
"import file mismatch" and the whole run aborts. This repo already met the same trap with two modules
named `client.py` and resolved it the same way, by keeping the test basenames distinct.

**Why validation is worth testing rather than trusting.** Every rule in `__post_init__` catches a
misconfiguration whose natural symptom is *an empty result*, not an error: a blank query matches
nothing, inverted year bounds match nothing, and both look exactly like a corpus that has no papers.
The whole value of failing at construction is that the error names the field instead of the operator
inferring it from a zero.

Offline and instant — this is a frozen dataclass and a sha256.
"""

import pytest
from pubmedcorpus.config import IngestConfig

REQUIRED = dict(query="a", ncbi_email="who@example.org", ncbi_tool="test")


def cfg(**kw) -> IngestConfig:
    return IngestConfig(**{**REQUIRED, **kw})


class TestRequiredFields:
    """All three are required with no default — see the module docstring in `config.py` on why."""

    @pytest.mark.parametrize("field", ["query", "ncbi_email", "ncbi_tool"])
    def test_missing_is_a_type_error(self, field):
        """No default at all, rather than a default of `""` that always fails validation."""
        with pytest.raises(TypeError):
            IngestConfig(**{k: v for k, v in REQUIRED.items() if k != field})

    @pytest.mark.parametrize("field", ["query", "ncbi_email", "ncbi_tool"])
    @pytest.mark.parametrize("blank", ["", "   "])
    def test_blank_or_whitespace_is_rejected(self, field, blank):
        with pytest.raises(ValueError, match=field):
            cfg(**{field: blank})


class TestValidation:
    def test_inverted_year_bounds_are_rejected(self):
        """Would otherwise return zero records and look like an empty corpus."""
        with pytest.raises(ValueError, match="inverted"):
            cfg(year_min=2020, year_max=2010)

    def test_equal_year_bounds_are_fine(self):
        """A single-year corpus is a legitimate thing to ask for."""
        assert cfg(year_min=2020, year_max=2020).year_min == 2020

    @pytest.mark.parametrize("field", ["batch_size", "pmc_batch_size"])
    @pytest.mark.parametrize("bad", [0, -1])
    def test_non_positive_batch_sizes_are_rejected(self, field, bad):
        """A zero batch size loops forever fetching nothing."""
        with pytest.raises(ValueError, match=field):
            cfg(**{field: bad})

    def test_a_negative_rate_limit_is_rejected(self):
        with pytest.raises(ValueError, match="rate_limit"):
            cfg(rate_limit=-1.0)

    def test_a_zero_rate_limit_is_allowed(self):
        """Zero means "no throttling", which `_throttle` handles explicitly. Not the same as negative."""
        assert cfg(rate_limit=0).rate_limit == 0

    def test_a_blank_api_key_is_rejected_not_coerced(self):
        """**Rejected deliberately.** Coercing `""` to None inside a frozen dataclass needs
        `object.__setattr__`, which undermines the `frozen=True` guarantee. A consumer normalises on
        its own side, where the empty string came from — as this project's pydantic validator does.
        """
        with pytest.raises(ValueError, match="ncbi_api_key"):
            cfg(ncbi_api_key="")

    def test_no_api_key_is_a_supported_mode(self):
        """It lowers NCBI's ceiling from 10/sec to 3/sec; it is not an error."""
        assert cfg().ncbi_api_key is None


class TestFrozen:
    """Both halves of "avoid accidental changes at runtime", and both come from `frozen=True` alone.

    `FrozenInstanceError` subclasses `AttributeError`, so these assert the narrower type — a change that
    started raising something else would be a real behaviour change worth failing on.
    """

    def test_a_field_cannot_be_reassigned(self):
        with pytest.raises(AttributeError, match="query"):
            cfg().query = "something else"  # type: ignore[misc]

    def test_a_typod_attribute_raises_and_names_the_field(self):
        """**This is why `slots=True` is not used**, despite being specified for this dataclass.

        Frozen alone raises `FrozenInstanceError: cannot assign to field 'querry'` — it names the typo.
        Adding `slots=True` replaces that with `TypeError: super(type, obj): obj must be an instance or
        subtype of type`, because `@dataclass(slots=True)` builds a new class while the generated
        `__setattr__` still closes over the original, making its `super()` call invalid. Slots buys
        nothing here that frozen does not already provide, so the clearer error wins.

        If this ever fails with `TypeError`, someone re-added `slots=True`.
        """
        with pytest.raises(AttributeError, match="querry"):
            cfg().querry = "typo"  # type: ignore[attr-defined]


class TestRecordedAndHash:
    def test_the_api_key_is_never_recorded(self):
        """A credential does not belong in a table meant to be read."""
        assert "ncbi_api_key" not in cfg(ncbi_api_key="secret").recorded()

    def test_the_api_key_does_not_affect_the_hash(self):
        """Follows from the above, and is the property that matters: rotating a key must not read as a
        new corpus definition."""
        assert cfg(ncbi_api_key="one").config_hash() == cfg(ncbi_api_key="two").config_hash()

    def test_the_hash_is_stable_across_construction_order(self):
        """`sort_keys` is what guarantees this — without it the digest would depend on field order."""
        a = IngestConfig(query="a", ncbi_email="e@x.org", ncbi_tool="t", year_min=2000)
        b = IngestConfig(ncbi_tool="t", year_min=2000, ncbi_email="e@x.org", query="a")
        assert a.config_hash() == b.config_hash()

    @pytest.mark.parametrize("field,value", [
        ("query", "different"),
        ("year_min", 2000),
        ("year_max", 2020),
        ("max_records", 50),
        ("batch_size", 100),
        ("pmc_batch_size", 5),
        ("ncbi_email", "other@example.org"),
        ("ncbi_tool", "other"),
        ("rate_limit", 2.0),
    ])
    def test_every_recorded_field_changes_the_hash(self, field, value):
        """Each recorded field must reach the digest, or two distinct definitions would collide — and
        since the insert is `on_conflict_do_nothing`, the second would be silently discarded.
        """
        assert cfg().config_hash() != cfg(**{field: value}).config_hash()

    def test_recorded_matches_the_table_columns(self):
        """**The drift this design could develop.** `recorded()` feeds both the hash and the row, so if
        it ever disagrees with `corpus_config`'s columns the insert breaks — or worse, records a subset
        while hashing a superset. Compared against the model rather than a hand-written list, so adding
        a column without adding it here fails right away.
        """
        from pubmedcorpus.models import CorpusConfig

        columns = {c.name for c in CorpusConfig.__table__.columns}
        assert set(cfg().recorded()) == columns - {"config_hash", "created_at"}
