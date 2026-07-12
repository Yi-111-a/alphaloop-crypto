from __future__ import annotations

import ast
from pathlib import Path

import pytest

from LOCKED.clock import Clock, FakeClock, SystemClock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCANNED_DIRS = [PROJECT_ROOT / "LOCKED", PROJECT_ROOT / "ASSET"]
# clock.py is the ONE legitimate place time.time() is called.
EXEMPT_FILES = {PROJECT_ROOT / "LOCKED" / "clock.py"}

_WALLCLOCK_ATTRS = {"time", "now", "utcnow", "today"}
_WALLCLOCK_MODULES = {"time", "datetime"}


def _find_wallclock_calls(tree: ast.AST) -> list[str]:
    """Returns a list of human-readable descriptions of any wall-clock call
    found in the AST (module.func() shaped calls like time.time()/datetime.now()
    etc). Mirrors the AST-check pattern already established in
    ASSET/memory/engine.py / LOCKED/cold_start.py / LOCKED/reflector.py."""
    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            attr = node.func.attr
            value = node.func.value
            if attr in _WALLCLOCK_ATTRS and isinstance(value, ast.Name) and value.id in _WALLCLOCK_MODULES:
                hits.append(f"{value.id}.{attr}()")
    return hits


def _iter_project_py_files():
    for d in SCANNED_DIRS:
        for path in d.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            if path in EXEMPT_FILES:
                continue
            yield path


def test_no_wallclock_calls_anywhere_in_locked_or_asset():
    """M5 requirement 3, generalized project-wide from the pattern M2 introduced
    for a single module (ASSET/memory/engine.py): the ENTIRE project (except
    LOCKED/clock.py's SystemClock implementation) must be free of direct
    time.time()/datetime.now()/datetime.utcnow()/datetime.today() calls. Every
    module gets "now" from an injected Clock (or an explicit ts/now_ms/now_date
    parameter supplied by whatever ultimately reads a Clock) -- never from the
    wall clock directly."""
    offenders: dict[str, list[str]] = {}
    for path in _iter_project_py_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        hits = _find_wallclock_calls(tree)
        if hits:
            offenders[str(path.relative_to(PROJECT_ROOT))] = hits

    assert not offenders, (
        "Found direct wall-clock calls outside LOCKED/clock.py -- route these "
        f"through an injected Clock instead: {offenders}"
    )


def test_system_clock_returns_plausible_utc_ms():
    clock = SystemClock()
    now = clock.now_ms()
    # sanity bound: sometime after 2024-01-01 UTC, well before any test rots into
    # implausibility. Not a determinism test -- SystemClock is the one place
    # allowed to touch the real wall clock.
    assert now > 1_700_000_000_000
    assert isinstance(now, int)


def test_fake_clock_set_and_advance():
    clock = FakeClock(start_ms=1_700_000_000_000)
    assert clock.now_ms() == 1_700_000_000_000

    clock.advance_ms(1000)
    assert clock.now_ms() == 1_700_000_001_000

    clock.set_ms(1_700_000_100_000)
    assert clock.now_ms() == 1_700_000_100_000


def test_fake_clock_rejects_backwards_movement():
    clock = FakeClock(start_ms=1_700_000_000_000)
    with pytest.raises(ValueError):
        clock.set_ms(1_699_999_999_999)
    with pytest.raises(ValueError):
        clock.advance_ms(-1)


def test_clock_is_abstract_and_fake_system_both_implement_it():
    assert isinstance(SystemClock(), Clock)
    assert isinstance(FakeClock(0), Clock)
    with pytest.raises(TypeError):
        Clock()  # abstract, cannot instantiate directly
