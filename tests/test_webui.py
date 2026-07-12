"""
tests/test_webui.py -- 只读监控面板验收测试。

覆盖用户对面板提出的两条硬性要求:
  1. "严格只读...不import任何LOCKED/ASSET业务模块" -- test 1/2 用 AST 静态扫描
     webui/app.py,分别验证零 LOCKED/ASSET 导入、零写文件/写数据库能力。
  2. "空数据兜底...不许报错" -- 剩余测试用假 LOG 数据(以及完全没有数据的
     情况)驱动一个真实的 FastAPI TestClient,确认每个端点在任何数据状态下
     都返回 200,不 500。
"""
from __future__ import annotations

import ast
import json
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

WEBUI_DIR = Path(__file__).resolve().parent.parent / "webui"
APP_PY = WEBUI_DIR / "app.py"


# ---------------------------------------------------------------------------
# 1. 结构性只读校验(AST,不依赖运行时行为)
# ---------------------------------------------------------------------------


def test_app_py_has_zero_locked_or_asset_imports():
    tree = ast.parse(APP_PY.read_text(encoding="utf-8"), filename=str(APP_PY))
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in ("LOCKED", "ASSET"):
                    offenders.append(f"import {alias.name} (line {node.lineno})")
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] in ("LOCKED", "ASSET"):
                offenders.append(f"from {node.module} import ... (line {node.lineno})")
    assert offenders == [], f"webui/app.py must never import LOCKED/ASSET business modules: {offenders}"


def test_app_py_has_no_write_capability_ast():
    """静态证明本文件里不存在任何写文件/写数据库的代码路径:
    - 没有 .write(...) 方法调用(读文件只会用 .read()/.read_text()/迭代)
    - 没有 open(...) 以写模式("w"/"a"/"x"/"+")打开
    - 每一次 sqlite3.connect(...) 调用的字面量参数里都出现 "mode=ro"
    """
    source = APP_PY.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(APP_PY))

    def _string_value(node: ast.AST) -> str:
        """Best-effort literal string extraction, including simple f-strings
        (JoinedStr) so `f"file:{path}?mode=ro"` is recognized as containing
        the literal 'mode=ro' text even though part of it is an interpolated
        expression."""
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.JoinedStr):
            return "".join(
                part.value for part in node.values if isinstance(part, ast.Constant) and isinstance(part.value, str)
            )
        return ""

    # Resolve simple `name = <string-or-fstring>` assignments so a
    # `sqlite3.connect(uri, ...)` call where `uri` was built via an f-string
    # a few lines earlier is still checked against its real literal content,
    # not just the bare Name reference.
    string_assignments: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            value_str = _string_value(node.value)
            if value_str:
                string_assignments[node.targets[0].id] = value_str

    write_method_calls = []
    write_mode_opens = []
    unsafe_sqlite_connects = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "write":
                write_method_calls.append(f"line {node.lineno}")

            if isinstance(func, ast.Name) and func.id == "open":
                for arg in list(node.args) + [kw.value for kw in node.keywords]:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        mode = arg.value
                        if any(c in mode for c in ("w", "a", "x", "+")):
                            write_mode_opens.append(f"line {node.lineno}: mode={mode!r}")

            is_sqlite_connect = (
                (isinstance(func, ast.Attribute) and func.attr == "connect")
                or (isinstance(func, ast.Name) and func.id == "connect")
            )
            if is_sqlite_connect:
                resolved_args = []
                for a in list(node.args) + [kw.value for kw in node.keywords]:
                    if isinstance(a, ast.Name) and a.id in string_assignments:
                        resolved_args.append(string_assignments[a.id])
                    else:
                        resolved_args.append(_string_value(a))
                joined = " ".join(resolved_args)
                if "mode=ro" not in joined:
                    unsafe_sqlite_connects.append(f"line {node.lineno}: resolved_args={resolved_args}")

    assert write_method_calls == [], f"found .write(...) call(s) in webui/app.py: {write_method_calls}"
    assert write_mode_opens == [], f"found write-mode open(...) call(s) in webui/app.py: {write_mode_opens}"
    assert unsafe_sqlite_connects == [], (
        f"found sqlite3.connect(...) call(s) without a 'mode=ro' read-only URI: {unsafe_sqlite_connects}"
    )


def test_app_py_has_no_fastapi_write_verbs():
    """没有任何 @app.post/put/patch/delete 路由 -- 面板不提供任何控制/写接口,
    只有 @app.get。"""
    source = APP_PY.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(APP_PY))
    forbidden_decorators = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for dec in node.decorator_list:
                if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute):
                    if dec.func.attr in ("post", "put", "patch", "delete"):
                        forbidden_decorators.append(f"{dec.func.attr} on {node.name} (line {node.lineno})")
    assert forbidden_decorators == [], f"webui must only expose GET routes: {forbidden_decorators}"


# ---------------------------------------------------------------------------
# 2. 运行时冒烟测试(真实 FastAPI TestClient,假 LOG 数据 + 完全空数据两种场景)
# ---------------------------------------------------------------------------


@pytest.fixture
def client(tmp_path, monkeypatch):
    import webui.app as webui_app

    log_root = tmp_path / "LOG"
    state_root = tmp_path / "state"
    log_root.mkdir(parents=True, exist_ok=True)
    state_root.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(webui_app, "LOG_ROOT", log_root)
    monkeypatch.setattr(webui_app, "STATE_ROOT", state_root)

    return TestClient(webui_app.app), log_root, state_root


def test_empty_data_fallback_every_endpoint_returns_200_not_500(client):
    """铁律:系统还没跑起来、LOG目录里什么都没有时,面板不能报错。"""
    test_client, _log_root, _state_root = client
    endpoints = [
        "/", "/api/status", "/api/nav", "/api/positions", "/api/latest_advice",
        "/api/funding_summary", "/api/fees_summary", "/api/ratchet_log", "/api/health",
    ]
    for ep in endpoints:
        resp = test_client.get(ep)
        assert resp.status_code == 200, f"{ep} returned {resp.status_code} on empty data"

    assert test_client.get("/api/nav").json()["has_data"] is False
    assert test_client.get("/api/positions").json()["has_data"] is False
    assert test_client.get("/api/latest_advice").json()["has_data"] is False
    status = test_client.get("/api/status").json()
    assert status["cold_start_state"] is None
    assert status["circuit_state"] is None
    assert status["disclaimer"] == "本内容为AI模拟实验输出,非投资建议,不构成任何交易依据"


def test_disclaimer_present_in_index_html(client):
    test_client, _log_root, _state_root = client
    resp = test_client.get("/")
    assert resp.status_code == 200
    assert "本内容为AI模拟实验输出,非投资建议,不构成任何交易依据" in resp.text


def test_nav_endpoint_renders_fake_nav_tsv(client):
    test_client, log_root, _state_root = client
    (log_root / "nav.tsv").write_text(
        "date\tnav_agent\tnav_benchmark\tnav_random\n"
        "2026-01-01\t100000.0\t100000.0\t100000.0\n"
        "2026-01-02\t101000.0\t100500.0\t99900.0\n",
        encoding="utf-8",
    )
    data = test_client.get("/api/nav").json()
    assert data["has_data"] is True
    assert len(data["rows"]) == 2
    assert data["rows"][1]["nav_agent"] == 101000.0


def test_status_endpoint_reads_cold_start_and_circuit_state(client):
    test_client, log_root, state_root = client
    (state_root / "cold_start_state.json").write_text(json.dumps({"state": "NORMAL"}), encoding="utf-8")
    with open(log_root / "circuit_breaker_state.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"previous_state": "NORMAL", "new_state": "FROZEN_24H"}) + "\n")

    data = test_client.get("/api/status").json()
    assert data["cold_start_state"] == "NORMAL"
    assert data["circuit_state"] == "FROZEN_24H"


def test_positions_endpoint_reads_real_readonly_sqlite(client):
    test_client, _log_root, state_root = client
    db_path = state_root / "portfolio_main.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE wallet (branch TEXT PRIMARY KEY, balance REAL NOT NULL, branch_dead INTEGER NOT NULL DEFAULT 0)"
    )
    conn.execute(
        """CREATE TABLE positions (
            branch TEXT NOT NULL, symbol TEXT NOT NULL, side TEXT NOT NULL,
            notional REAL NOT NULL, entry_price REAL NOT NULL, margin REAL NOT NULL, leverage INTEGER NOT NULL,
            PRIMARY KEY (branch, symbol)
        )"""
    )
    conn.execute("INSERT INTO wallet VALUES ('main', 95000.0, 0)")
    conn.execute("INSERT INTO positions VALUES ('main', 'BTC/USDT:USDT', 'long', 30000.0, 50000.0, 15000.0, 2)")
    conn.commit()
    conn.close()

    data = test_client.get("/api/positions").json()
    assert data["has_data"] is True
    assert data["wallet_balance"] == 95000.0
    assert data["branch_dead"] is False
    assert len(data["positions"]) == 1
    assert data["positions"][0]["symbol"] == "BTC/USDT:USDT"
    assert data["positions"][0]["leverage"] == 2


def test_latest_advice_rendered_verbatim(client):
    test_client, log_root, _state_root = client
    content = (
        "# AlphaLoop 最新建议\n\n"
        "本内容为AI模拟实验输出,非投资建议,不构成任何交易依据\n\n"
        "thesis: 基于H3的独一无二占位标记ZZQVXK\n"
        "falsifier: price<48000\n"
    )
    (log_root / "latest_advice.md").write_text(content, encoding="utf-8")
    data = test_client.get("/api/latest_advice").json()
    assert data["has_data"] is True
    assert data["content"] == content
    assert "ZZQVXK" in data["content"]


def test_funding_and_fees_summary(client):
    test_client, log_root, _state_root = client
    with open(log_root / "funding.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": 1, "symbol": "BTC/USDT:USDT", "branch": "main", "amount": 5.0}) + "\n")
        f.write(json.dumps({"ts": 2, "symbol": "BTC/USDT:USDT", "branch": "main", "amount": -2.0}) + "\n")
    with open(log_root / "trades.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": 1, "branch": "main", "fee": 25.0}) + "\n")

    funding = test_client.get("/api/funding_summary").json()
    assert funding["has_data"] is True
    assert funding["total_amount"] == pytest.approx(3.0)

    fees = test_client.get("/api/fees_summary").json()
    assert fees["has_data"] is True
    assert fees["total_fee"] == pytest.approx(25.0)


def test_ratchet_log_returns_last_n_reversed(client):
    test_client, log_root, _state_root = client
    with open(log_root / "ratchet_verdicts.jsonl", "a", encoding="utf-8") as f:
        for i in range(15):
            f.write(json.dumps({
                "branch": f"evo/{i}", "decision": "ARCHIVE", "score": 0.0,
                "edge_vs_main_pct": 0.0, "max_drawdown_pct": 0.0, "reason": "test", "now_date": "2026-01-01",
            }) + "\n")

    data = test_client.get("/api/ratchet_log").json()
    assert data["has_data"] is True
    assert len(data["verdicts"]) == 10
    assert data["verdicts"][0]["branch"] == "evo/14"  # most recent first


def test_health_endpoint_flags_stale_data_as_red(client):
    test_client, log_root, _state_root = client
    with open(log_root / "decisions.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": 1_000_000_000, "branch": "main"}) + "\n")  # ancient timestamp

    data = test_client.get("/api/health").json()
    assert data["decision_gap"]["status"] == "red"
