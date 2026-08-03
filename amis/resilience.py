"""
Fetch-path fault tolerance: TTL cache, circuit breaker, retry with backoff.
=================================================================

Adapted from the Tattva terminal's ``data/cache.py`` and
``data/circuit_breaker.py``.  Three primitives, each guarding a different
failure mode of a free market-data API:

* :class:`TTLCache` -- two tiers (memory, then disk), versioned keys, and a
  *stale fallback* that serves last-good data when a live fetch fails.
* :class:`CircuitBreaker` -- stops hammering a service that is already down
  (CLOSED -> OPEN -> HALF_OPEN), so one outage costs one timeout rather than
  one timeout per instrument.
* :class:`RetryWithBackoff` -- absorbs the transient whole-batch failure.

One adaptation matters and is deliberate.  Tattva caches *price frames* under
a TTL, so a refresh overwrites the previous frame.  That is right for Tattva
and wrong here: AMIS's revision-invariance guarantee says a value published
for 2015-03-04 is the same value forever, and it cannot survive a store that
rewrites history on refresh.  So the append-only parquet store in
:mod:`amis.data` remains the system of record for prices, and this module
guards the *fetch* that feeds it.  The TTL cache is used for the things where
overwriting is the correct behaviour -- index constituent lists, symbol
resolutions -- which are lookups, not observations.

References
----------
Nygard, M. T. (2018). *Release It!*, 2nd ed. -- the circuit breaker and its
    state machine.
"""

from __future__ import annotations

import hashlib
import logging
import pickle
import threading
import time
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger(__name__)

CACHE_ROOT = Path(__file__).resolve().parent.parent / ".amis_cache"
CACHE_ROOT.mkdir(parents=True, exist_ok=True)

#: Disk snapshots older than this are pruned on write.
SNAPSHOT_RETENTION_DAYS = 14

#: Per-session force-refresh deadlines.  Streamlit serves every concurrent
#: session from one process, so a single global deadline would make one user's
#: "Refresh" force every other session's fetches to bypass cache too.
_FORCE_UNTIL: dict[str, float] = {}


def _session_key() -> str:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        try:
            ctx = get_script_run_ctx(suppress_warning=True)
        except TypeError:                       # older signature
            ctx = get_script_run_ctx()
        if ctx is not None:
            return ctx.session_id
    except Exception:
        pass
    return "_no_session_"


# ===========================================================================
# TTL cache
# ===========================================================================
class TTLCache:
    """Two-tier cache with versioned keys and a last-good stale fallback."""

    def __init__(self, ttl: int = 3600, version: str = "v1",
                 namespace: str = "", disk_dir: Path | None = None) -> None:
        self.ttl = ttl
        self.version = version
        self._memory: dict[str, tuple[Any, float]] = {}
        base = disk_dir or CACHE_ROOT
        self._disk_dir = base / namespace if namespace else base
        self._disk_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.hits = self.misses = self.stale_hits = self.writes = 0
        self.last_fetch_time: float | None = None

    def _key(self, *args: Any) -> str:
        raw = f"{self.version}|" + "|".join(str(a) for a in args)
        return hashlib.md5(raw.encode()).hexdigest()

    def get(self, *args: Any) -> Any | None:
        """Fresh value, or None.

        During a force-refresh window this returns None so the caller
        re-fetches live, *without* deleting the disk snapshot -- so a failed
        forced refresh degrades to stale rather than to empty, which is the
        safety a naive "clear the cache" button lacks.
        """
        if time.time() < _FORCE_UNTIL.get(_session_key(), 0.0):
            with self._lock:
                self.misses += 1
            return None
        key = self._key(*args)
        with self._lock:
            hit = self._memory.get(key)
            if hit is not None:
                val, ts = hit
                if time.time() - ts < self.ttl:
                    self.hits += 1
                    return val
                del self._memory[key]

        path = self._disk_dir / f"{key}.pkl"
        if path.exists():
            try:
                with open(path, "rb") as fh:
                    val, ts = pickle.load(fh)
                if time.time() - ts < self.ttl:
                    with self._lock:
                        self._memory[key] = (val, ts)
                        self.hits += 1
                    return val
            except Exception as exc:            # noqa: BLE001
                log.warning("cache read failed for %s: %s", key[:8], exc)

        with self._lock:
            self.misses += 1
        return None

    def get_stale(self, *args: Any) -> Any | None:
        """Last-good value even if expired.  The fetch-failure fallback."""
        key = self._key(*args)
        with self._lock:
            hit = self._memory.get(key)
            if hit is not None:
                self.stale_hits += 1
                return hit[0]
        path = self._disk_dir / f"{key}.pkl"
        if path.exists():
            try:
                with open(path, "rb") as fh:
                    val, ts = pickle.load(fh)
                with self._lock:
                    self._memory[key] = (val, ts)
                    self.stale_hits += 1
                return val
            except Exception:                   # noqa: BLE001
                pass
        return None

    def put(self, *args: Any, value: Any) -> None:
        key = self._key(*args)
        ts = time.time()
        with self._lock:
            self._memory[key] = (value, ts)
            self.writes += 1
            self.last_fetch_time = ts
        try:
            with open(self._disk_dir / f"{key}.pkl", "wb") as fh:
                pickle.dump((value, ts), fh)
        except Exception as exc:                # noqa: BLE001
            log.warning("cache write failed for %s: %s", key[:8], exc)
        self._prune()

    def _prune(self) -> None:
        cutoff = time.time() - SNAPSHOT_RETENTION_DAYS * 86400
        try:
            for p in self._disk_dir.glob("*.pkl"):
                try:
                    if p.stat().st_mtime < cutoff:
                        p.unlink()
                except OSError:
                    continue
        except Exception:                       # noqa: BLE001
            pass

    def invalidate(self, *args: Any) -> None:
        key = self._key(*args)
        with self._lock:
            self._memory.pop(key, None)
        p = self._disk_dir / f"{key}.pkl"
        if p.exists():
            try:
                p.unlink()
            except Exception:                   # noqa: BLE001
                pass

    def stats(self) -> dict:
        with self._lock:
            total = self.hits + self.misses
            return {
                "namespace": self._disk_dir.name,
                "version": self.version,
                "ttl_seconds": self.ttl,
                "hits": self.hits,
                "misses": self.misses,
                "stale_hits": self.stale_hits,
                "writes": self.writes,
                "hit_rate": (self.hits / total) if total else 0.0,
                "memory_entries": len(self._memory),
                "disk_entries": len(list(self._disk_dir.glob("*.pkl"))),
                "last_fetch_time": self.last_fetch_time,
            }


def begin_force_refresh(window: float = 300.0) -> None:
    """Bypass cache for this session for `window` seconds; snapshots survive."""
    now = time.time()
    for k in [k for k, d in _FORCE_UNTIL.items() if d < now]:
        del _FORCE_UNTIL[k]
    _FORCE_UNTIL[_session_key()] = now + window


# ===========================================================================
# Circuit breaker
# ===========================================================================
class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerError(Exception):
    """Raised when a call is blocked because the circuit is OPEN."""


class CircuitBreaker:
    """Per-service breaker.  Thread-safe.

    CLOSED -> OPEN on `failure_threshold` consecutive failures; OPEN ->
    HALF_OPEN once `recovery_timeout` has elapsed; HALF_OPEN -> CLOSED on the
    next success, or back to OPEN if the probe also fails.
    """

    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0,
                 half_open_max_calls: int = 1, name: str = "default") -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self.name = name
        self.state = CircuitState.CLOSED
        self.failure_count = self.success_count = 0
        self.last_failure_time: float | None = None
        self.last_success_time: float | None = None
        self.half_open_calls = 0
        self._lock = threading.Lock()

    def call(self, func: Callable, *args, **kwargs) -> Any:
        with self._lock:
            if self.state is CircuitState.OPEN:
                if self.last_failure_time is None:
                    raise CircuitBreakerError(f"circuit '{self.name}' is OPEN")
                elapsed = time.time() - self.last_failure_time
                if elapsed > self.recovery_timeout:
                    self.state = CircuitState.HALF_OPEN
                    self.half_open_calls = 0
                else:
                    raise CircuitBreakerError(
                        f"circuit '{self.name}' is OPEN — retry in "
                        f"{self.recovery_timeout - elapsed:.0f}s")
            if self.state is CircuitState.HALF_OPEN:
                self.half_open_calls += 1
                if self.half_open_calls > self.half_open_max_calls:
                    raise CircuitBreakerError(
                        f"circuit '{self.name}' HALF_OPEN — probe in flight")
        try:
            result = func(*args, **kwargs)
        except Exception:
            self._on_failure()
            raise
        self._on_success()
        return result

    def _on_success(self) -> None:
        with self._lock:
            prev = self.state
            self.success_count += 1
            self.last_success_time = time.time()
            if self.state is CircuitState.HALF_OPEN:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.half_open_calls = 0
        if prev is CircuitState.HALF_OPEN:
            log.info("circuit '%s' CLOSED — service recovered", self.name)

    def _on_failure(self) -> None:
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.state is CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
            elif (self.state is CircuitState.CLOSED
                  and self.failure_count >= self.failure_threshold):
                self.state = CircuitState.OPEN
                log.error("circuit '%s' OPEN — %d failures", self.name,
                          self.failure_count)

    def get_state(self) -> dict:
        return {"name": self.name, "state": self.state.value,
                "failure_count": self.failure_count,
                "success_count": self.success_count,
                "failure_threshold": self.failure_threshold,
                "recovery_timeout": self.recovery_timeout,
                "last_failure": self.last_failure_time,
                "last_success": self.last_success_time}

    def reset(self) -> None:
        with self._lock:
            self.state = CircuitState.CLOSED
            self.failure_count = self.success_count = 0
            self.last_failure_time = self.last_success_time = None
            self.half_open_calls = 0


class RetryWithBackoff:
    """Exponential-backoff retry decorator (delay, delay*f, delay*f^2, ...)."""

    def __init__(self, max_retries: int = 2, backoff_factor: float = 2.0,
                 initial_delay: float = 1.5, max_delay: float = 30.0,
                 exceptions: tuple = (Exception,)) -> None:
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exceptions = exceptions

    def __call__(self, func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last: Exception | None = None
            delay = self.initial_delay
            for attempt in range(self.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except self.exceptions as exc:
                    last = exc
                    if attempt < self.max_retries:
                        log.warning("attempt %d/%d failed (%s) — retry in %.1fs",
                                    attempt + 1, self.max_retries + 1,
                                    type(exc).__name__, delay)
                        time.sleep(delay)
                        delay = min(delay * self.backoff_factor, self.max_delay)
            assert last is not None
            raise last
        return wrapper


# ---------------------------------------------------------------------------
# Process-wide instances
# ---------------------------------------------------------------------------
yfinance_circuit = CircuitBreaker(name="yfinance", failure_threshold=5,
                                  recovery_timeout=60.0)

#: Constituent lists change slowly; a day is generous and the stale fallback
#: covers a blocked scrape.
constituent_cache = TTLCache(ttl=86_400, version="v1", namespace="constituents")

#: Only *successful* resolutions are memoised to disk — a transient outage
#: must not brand a symbol invalid for a week.
symbol_cache = TTLCache(ttl=7 * 86_400, version="v1", namespace="symbols")


def all_caches() -> list[TTLCache]:
    return [constituent_cache, symbol_cache]


def all_circuits() -> list[CircuitBreaker]:
    return [yfinance_circuit]
