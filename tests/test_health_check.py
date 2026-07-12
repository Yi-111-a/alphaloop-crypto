"""
tests/test_health_check.py -- scripts/health_check.py 的纯函数单测。

与 webui 同一纪律:health_check.py 不 import 任何 LOCKED/ASSET 模块,只读
LOG/state 文件,这里用 monkeypatch 把模块级 LOG_ROOT/STATE_ROOT 指向
tmp_path,不触碰真实项目目录。
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))


@pytest.fixture
def hc(tmp_path, monkeypatch):
    import health_check as hc_module
    importlib.reload(hc_module)

    log_root = tmp_path / "LOG"
    state_root = tmp_path / "state"
    log_root.mkdir(parents=True, exist_ok=True)
    state_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(hc_module, "LOG_ROOT", log_root)
    monkeypatch.setattr(hc_module, "STATE_ROOT", state_root)
    return hc_module, log_root, state_root


def test_empty_state_all_red(hc):
    module, _log_root, _state_root = hc
    now = 1_700_000_000_000
    assert module.check_decision_gaps(now)[0] is False
    assert module.check_funding_coverage(now)[0] is False
    assert module.check_nav_updated(now)[0] is False
    assert module.check_process_heartbeat(now)[0] is False


def test_decision_gap_green_when_recent(hc):
    module, log_root, _state_root = hc
    now = 1_700_000_000_000
    with open(log_root / "decisions.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": now - 60_000, "branch": "main"}) + "\n")
    ok, _detail = module.check_decision_gaps(now)
    assert ok is True


def test_decision_gap_red_when_stale(hc):
    module, log_root, _state_root = hc
    now = 1_700_000_000_000
    with open(log_root / "decisions.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": now - 100 * 3_600_000, "branch": "main"}) + "\n")
    ok, _detail = module.check_decision_gaps(now)
    assert ok is False


def test_funding_coverage_green_when_all_settle_hours_present(hc):
    module, log_root, _state_root = hc
    day_start = (1_700_000_000_000 // module.DAY_MS) * module.DAY_MS
    now = day_start + 16 * 3_600_000 + 60_000  # just past the 16:00 UTC settlement
    with open(log_root / "funding.jsonl", "a", encoding="utf-8") as f:
        for hour in (0, 8, 16):
            f.write(json.dumps({"ts": day_start + hour * 3_600_000}) + "\n")
    ok, _detail = module.check_funding_coverage(now)
    assert ok is True


def test_funding_coverage_red_when_missing_expected_hour(hc):
    module, log_root, _state_root = hc
    day_start = (1_700_000_000_000 // module.DAY_MS) * module.DAY_MS
    now = day_start + 16 * 3_600_000 + 60_000
    with open(log_root / "funding.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": day_start}) + "\n")  # only 00:00, missing 08:00 and 16:00
    ok, detail = module.check_funding_coverage(now)
    assert ok is False
    assert "8" in detail and "16" in detail


def test_nav_updated_green_with_valid_row(hc):
    module, log_root, _state_root = hc
    (log_root / "nav.tsv").write_text(
        "date\tnav_agent\tnav_benchmark\tnav_random\n2026-01-01\t100000\t100000\t100000\n",
        encoding="utf-8",
    )
    ok, _detail = module.check_nav_updated(1_700_000_000_000)
    assert ok is True


def test_process_heartbeat_green_when_fresh(hc):
    module, _log_root, state_root = hc
    now = 1_700_000_000_000
    (state_root / "ignite_heartbeat.json").write_text(json.dumps({"ts_ms": now - 30_000}), encoding="utf-8")
    ok, _detail = module.check_process_heartbeat(now)
    assert ok is True


def test_process_heartbeat_red_when_stale(hc):
    module, _log_root, state_root = hc
    now = 1_700_000_000_000
    (state_root / "ignite_heartbeat.json").write_text(
        json.dumps({"ts_ms": now - 10 * 60_000}), encoding="utf-8"
    )
    ok, _detail = module.check_process_heartbeat(now)
    assert ok is False


def test_health_check_script_has_zero_locked_or_asset_imports():
    """铁律:health_check.py 不 import 任何 LOCKED/ASSET 业务模块,与
    webui/app.py 同一纪律(见 tests/test_webui.py)。"""
    import ast

    source = (SCRIPTS_DIR / "health_check.py").read_text(encoding="utf-8")
    tree = ast.parse(source, filename="health_check.py")
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in ("LOCKED", "ASSET"):
                    offenders.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] in ("LOCKED", "ASSET"):
                offenders.append(node.module)
    assert offenders == []
