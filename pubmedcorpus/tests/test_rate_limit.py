"""Throttle behaviour.

Uses a fake monotonic clock and a fake sleep so the tests are instant and
deterministic — a test that really waits a second per call would make the suite
slow enough that people stop running it.

**Simpler than it was, because the rate is now an argument.** These tests used to need
`monkeypatch.setattr("sudep.config.NCBI_RATE_LIMIT", rate)` — reaching into a module constant because
`settings.ncbi_rate_limit` was a read-only property that could not be set. `_throttle(rate)` takes the
value from the calling client's config, so the rate is just a parameter now.

The two assertions about the rate being a hardcoded, non-env-configurable *constant* stayed behind in
`backend/tests/test_config.py`: that is a policy this application chose, not behaviour of the gate.
"""

import pytest
from pubmedcorpus import ncbi


@pytest.fixture
def clock(monkeypatch):
    class Clock:
        def __init__(self):
            self.now = 1000.0
            self.slept: list[float] = []

        def monotonic(self):
            return self.now

        def sleep(self, seconds):
            self.slept.append(seconds)
            self.now += seconds

        def advance(self, seconds):
            self.now += seconds

    c = Clock()
    monkeypatch.setattr(ncbi.time, "monotonic", c.monotonic)
    monkeypatch.setattr(ncbi.time, "sleep", c.sleep)
    # Module-level state, reset per test. It is module-level deliberately — NCBI's limit applies per
    # caller, so two clients must share one gate rather than each keeping its own and jointly
    # exceeding it. That shared-ness is exactly what makes resetting it here necessary.
    monkeypatch.setattr(ncbi, "_last_call", 0.0)
    return c


def test_first_call_does_not_sleep(clock):
    ncbi._throttle(1.0)
    assert clock.slept == []


def test_immediate_second_call_waits_full_interval(clock):
    ncbi._throttle(1.0)
    ncbi._throttle(1.0)
    assert clock.slept == [pytest.approx(1.0)]


def test_wait_accounts_for_elapsed_time(clock):
    # Work between calls counts toward the interval — the gate paces requests,
    # it doesn't add a fixed delay to each one.
    ncbi._throttle(1.0)
    clock.advance(0.6)
    ncbi._throttle(1.0)
    assert clock.slept == [pytest.approx(0.4)]


def test_no_sleep_when_interval_already_elapsed(clock):
    ncbi._throttle(1.0)
    clock.advance(5.0)
    ncbi._throttle(1.0)
    assert clock.slept == []


def test_higher_rate_shortens_the_interval(clock):
    ncbi._throttle(4.0)
    ncbi._throttle(4.0)
    assert clock.slept == [pytest.approx(0.25)]


def test_zero_disables_the_gate(clock):
    # 0 means "leave only Biopython's own limiter".
    ncbi._throttle(0.0)
    ncbi._throttle(0.0)
    assert clock.slept == []


def test_the_gate_is_shared_between_clients(clock):
    """**The property that makes module-level state correct rather than sloppy.**

    NCBI's limit applies to the caller, not to an object. Two clients that each kept their own
    `_last_call` would each observe 1/sec and together issue 2/sec. Throttling through separate client
    instances must still serialise.
    """
    from pubmedcorpus.config import IngestConfig

    def client(rate):
        return ncbi.NCBIClient(IngestConfig(
            query="a", ncbi_email="e@x.org", ncbi_tool="t", rate_limit=rate
        ))

    one, two = client(1.0), client(1.0)
    ncbi._throttle(one.config.rate_limit)
    ncbi._throttle(two.config.rate_limit)
    assert clock.slept == [pytest.approx(1.0)]
