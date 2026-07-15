"""
webui/app.py -- AlphaLoop-Crypto 只读监控面板(FastAPI)。

铁律(用户明确要求,写进代码而非注释里):
  - 严格只读。零写入、零控制接口——没有暂停/下单/改参数/解冻熔断器这类按钮
    或端点,本文件里也确实不存在任何写文件、写数据库的代码路径(见本文件
    末尾的 tests/test_webui.py::test_app_py_has_no_write_capability_ast 做的
    AST 静态校验)。
  - 不 import 任何 LOCKED/ 或 ASSET/ 下的业务模块。本文件只直接读
    LOG/ 下的 jsonl/tsv/md 文件和 state/portfolio_*.db 这个 sqlite 文件本身,
    不通过 Simulator/Scorer/CircuitBreaker/ColdStartGate/EvolutionOrchestrator
    等任何业务对象去读——这样即使那些模块本身有 bug,面板依然能独立工作;
    更重要的是,面板的代码路径里根本不存在"调用一个业务方法"这个动作,
    从结构上排除了面板意外触发任何写操作或业务逻辑的可能性,不是靠"我们
    保证不这样做"这种约定层面的自律。
  - sqlite 连接一律用 `mode=ro` 的只读 URI 打开(见 _read_portfolio_db),
    即使本文件出现 bug 试图写入,sqlite 驱动本身也会拒绝——双重保险。
  - 只监听 127.0.0.1:8080(见 webui/README.md 的启动命令),不对外网暴露。
  - 空数据兜底:系统尚未产生任何 LOG 数据时,每个 API 端点都返回一个明确的
    "尚无数据"结构,前端渲染"等待系统启动"文案,不抛 500——面板本来就应该
    先于交易系统本身上线,不能因为系统还没跑起来就自己先挂掉。

本文件不做任何计算/推导财务数字的"业务逻辑"——所有数字都是对 LOG 文件里
已经算好、已经落盘的字段做直接读取/求和/取最后一条,不重新计算 NAV、不重新
判断爆仓、不重新算资金费率。这条边界本身也是"面板只读"这件事的一部分:
面板不应该有能力"算出一个和 LOCKED 区不一样的数字"。
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional

import yaml
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

APP_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = APP_ROOT.parent

LOG_ROOT = PROJECT_ROOT / "LOG"
STATE_ROOT = PROJECT_ROOT / "state"

DISCLAIMER = "本内容为AI模拟实验输出,非投资建议,不构成任何交易依据"

MAIN_BRANCH = "main"
DECISION_INTERVAL_MS = 4 * 3_600_000  # 与 config.yaml 的 cycle.decision_interval_hours 默认值一致
SETTLE_INTERVAL_MS = 8 * 3_600_000  # 与 config.yaml 的 funding.settle_hours_utc 默认间隔一致
STALE_MULTIPLE = 2.5  # 超过 STALE_MULTIPLE 倍的正常间隔没有新记录 -> 判红灯


def _read_initial_capital_usdt() -> float:
    """直接读 config.yaml 的顶层 capital_usdt——这是个静态配置常量,不是业务
    模块,读它不违反本文件"不 import LOCKED/ASSET"的规矩。用于把已经算好的
    NAV(见 _read_nav_intraday*)换算成"总盈亏"时做减法,这个减法本身不是
    重新推导财务公式,只是拿两个已经落盘的/静态的数字做一次算术。"""
    config_path = PROJECT_ROOT / "config.yaml"
    if not config_path.exists():
        return 100_000.0
    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError:
        return 100_000.0
    return float((data or {}).get("capital_usdt", 100_000.0))


INITIAL_CAPITAL_USDT = _read_initial_capital_usdt()


def _now_ms() -> int:
    return int(time.time() * 1000)


# ---------------------------------------------------------------------------
# 纯文件读取小工具——全部只读,不存在任何 open(..., "w") / write() 调用。
# ---------------------------------------------------------------------------


def _read_jsonl(relative_path: str) -> list[dict]:
    path = LOG_ROOT / relative_path
    if not path.exists():
        return []
    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # 容忍读到半行(比如恰好在另一个进程追加写入的中途读到)
    return records


def _read_text(relative_path: str) -> Optional[str]:
    path = LOG_ROOT / relative_path
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def _read_json_state_file(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _read_nav_tsv() -> list[dict]:
    path = LOG_ROOT / "nav.tsv"
    if not path.exists():
        return []
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        header = None
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if header is None:
                header = parts
                continue
            if len(parts) != len(header):
                continue
            row = dict(zip(header, parts))
            for key in ("nav_agent", "nav_benchmark", "nav_random"):
                if key in row:
                    try:
                        row[key] = float(row[key])
                    except ValueError:
                        row[key] = None
            rows.append(row)
    return rows


def _read_nav_intraday(hours: int) -> list[dict]:
    """LOG/nav_intraday.jsonl 是每小时一笔的三线实时净值(见 scripts/ignite.py
    的 nav_intraday 记录点),与 nav.tsv 的日线历史是两份独立、互不覆盖的数据——
    这里只是原样读回、按 hours 窗口截断,不做任何插值/重采样等计算。"""
    records = _read_jsonl("nav_intraday.jsonl")
    if not records:
        return []
    cutoff = _now_ms() - hours * 3_600_000
    return [r for r in records if isinstance(r.get("ts"), (int, float)) and r["ts"] >= cutoff]


def _read_nav_intraday_branches(hours: int) -> list[dict]:
    """LOG/nav_intraday_branches.jsonl 是random对照组+每个evo候选分支的小时级
    净值(long格式:{ts, branch, nav}),与 nav_intraday.jsonl 的固定三线schema
    是两份独立文件——见 scripts/ignite.py 里两者分别写入的说明。同样只原样
    读回按 hours 窗口截断。"""
    records = _read_jsonl("nav_intraday_branches.jsonl")
    if not records:
        return []
    cutoff = _now_ms() - hours * 3_600_000
    return [r for r in records if isinstance(r.get("ts"), (int, float)) and r["ts"] >= cutoff]


def _read_registered_branches() -> list[dict]:
    """LOG/branch_registrations.jsonl 是 LOCKED.evolution_orchestrator.
    EvolutionOrchestrator.register_branch() 每次成功注册都追加的一条记录,
    这里原样读回,用于面板"总面板对比+点进某个分支看仓位"的分支列表——
    不去问orchestrator对象本身(面板不import LOCKED),直接读它自己维护的
    这份LOG。main/random不在这份日志里(它们不是"候选分支",是固定的两条
    对照线),前端会把它们和这里读到的evo分支合并成完整的可选分支列表。"""
    return _read_jsonl("branch_registrations.jsonl")


def _read_portfolio_db(branch: str) -> Optional[dict]:
    """只读打开 state/portfolio_{branch}.db。用 sqlite3 的 `mode=ro` URI 连接——
    这不是约定层面的"我们保证不写",而是驱动层面的强制:任何写操作在这个连接
    上都会直接失败。"""
    safe_branch = branch.replace("/", "_").replace(":", "_").replace(" ", "_")
    db_path = STATE_ROOT / f"portfolio_{safe_branch}.db"
    if not db_path.exists():
        return None

    uri = f"file:{db_path.as_posix()}?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True)
    except sqlite3.OperationalError:
        return None

    try:
        wallet_row = conn.execute(
            "SELECT balance, branch_dead FROM wallet WHERE branch = ?", (branch,)
        ).fetchone()
        if wallet_row is None:
            return None
        positions = []
        for r in conn.execute(
            "SELECT symbol, side, notional, entry_price, margin, leverage FROM positions WHERE branch = ?",
            (branch,),
        ):
            positions.append({
                "symbol": r[0], "side": r[1], "notional": r[2],
                "entry_price": r[3], "margin": r[4], "leverage": r[5],
            })
        return {
            "branch": branch,
            "wallet_balance": wallet_row[0],
            "branch_dead": bool(wallet_row[1]),
            "positions": positions,
        }
    finally:
        conn.close()


def _read_positions_marked(branch: str) -> dict[str, dict]:
    """state/positions_marked_{branch}.json 是 scripts/ignite.py 每小时用
    LOCKED.simulator.Simulator._unrealized_pnl(与真实结算同一套公式)算好、
    落盘的"标记价+未实现盈亏"快照——本文件只原样读回按symbol建索引,不做
    任何financial计算,遵守面板"零计算"的规矩。数据可能比当前时间落后最多
    约1小时(下一次hourly tick之前),这是有意的权衡,不是bug。"""
    safe_branch = branch.replace("/", "_").replace(":", "_").replace(" ", "_")
    data = _read_json_state_file(STATE_ROOT / f"positions_marked_{safe_branch}.json")
    if data is None:
        return {}
    return {p["symbol"]: p for p in data.get("positions", []) if "symbol" in p}


def _read_latest_nav_for_branch(branch: str) -> Optional[float]:
    """取某个分支最近一条已落盘的 NAV(总账户价值 = wallet_balance + Σ未实现
    盈亏,由 LOCKED.simulator.Simulator.get_portfolio() 算好、由 ignite.py
    落盘),不重新计算——main分支读 nav_intraday.jsonl 的 nav_agent 字段,
    其余分支读 nav_intraday_branches.jsonl 里对应 branch 的 nav 字段,两份
    文件都只取最后一条("取最后一条"是本文件docstring里明确允许的读取方式,
    不是重新推导)。"""
    if branch == MAIN_BRANCH:
        records = _read_jsonl("nav_intraday.jsonl")
        for record in reversed(records):
            if isinstance(record.get("nav_agent"), (int, float)):
                return float(record["nav_agent"])
        return None
    records = _read_jsonl("nav_intraday_branches.jsonl")
    for record in reversed(records):
        if record.get("branch") == branch and isinstance(record.get("nav"), (int, float)):
            return float(record["nav"])
    return None


def _read_trades(branch: str, limit: int) -> list[dict]:
    """LOG/trades.jsonl 是 LOCKED.simulator.Simulator.execute() 每笔真实成交
    都会追加的记录(见该文件模块docstring),按 branch 过滤、按 ts 降序取最近
    limit 条——原样读回,不做任何盈亏再计算(单笔成交的已实现盈亏不在 Trade
    记录本身里,这是 LOCKED.schemas.Trade 的既有字段形状,面板不去凭 entry
    price 反推,避免和 LOCKED 区的真实结算逻辑出现第二份、可能不一致的实现)。"""
    records = [r for r in _read_jsonl("trades.jsonl") if r.get("branch", MAIN_BRANCH) == branch]
    records.sort(key=lambda r: r.get("ts", 0), reverse=True)
    return records[:limit]


def _read_cold_start_state() -> Optional[str]:
    data = _read_json_state_file(STATE_ROOT / "cold_start_state.json")
    if data is None:
        return None
    return data.get("state")


def _read_circuit_state() -> Optional[str]:
    records = _read_jsonl("circuit_breaker_state.jsonl")
    if not records:
        return None
    return records[-1].get("new_state")


# ---------------------------------------------------------------------------
# 健康体征灯——都只是"读回LOG,对比预期节奏",不重新计算任何业务数字。
# ---------------------------------------------------------------------------


def _decision_gap_health() -> dict:
    records = [r for r in _read_jsonl("decisions.jsonl") if r.get("branch", "main") == MAIN_BRANCH]
    if not records:
        return {"status": "unknown", "detail": "尚无决策记录", "last_ts": None}
    last_ts = max(r["ts"] for r in records if "ts" in r)
    gap_ms = _now_ms() - last_ts
    status = "green" if gap_ms <= DECISION_INTERVAL_MS * STALE_MULTIPLE else "red"
    return {"status": status, "detail": f"距上一条决策 {gap_ms // 60000} 分钟", "last_ts": last_ts}


def _funding_health() -> dict:
    records = _read_jsonl("funding.jsonl")
    if not records:
        return {"status": "unknown", "detail": "尚无资金费率结算记录", "last_ts": None}
    last_ts = max(r["ts"] for r in records if "ts" in r)
    gap_ms = _now_ms() - last_ts
    status = "green" if gap_ms <= SETTLE_INTERVAL_MS * STALE_MULTIPLE else "red"
    return {"status": status, "detail": f"距上一次结算 {gap_ms // 60000} 分钟", "last_ts": last_ts}


def _heartbeat_health() -> dict:
    """没有专门的心跳文件时,退化为"LOG目录里任意文件最近一次被修改的时间"
    作为进程存活的间接证据——这是一个明确标注的启发式代理,不是精确的进程
    存活检测(面板本身不 import 任何业务模块,也就没有办法直接问"main.py
    这个进程还活着吗")。"""
    if not LOG_ROOT.exists():
        return {"status": "unknown", "detail": "LOG目录尚不存在", "last_mtime_ms": None}
    latest_mtime = None
    for path in LOG_ROOT.rglob("*"):
        if path.is_file():
            mtime_ms = int(path.stat().st_mtime * 1000)
            if latest_mtime is None or mtime_ms > latest_mtime:
                latest_mtime = mtime_ms
    if latest_mtime is None:
        return {"status": "unknown", "detail": "LOG目录尚无任何文件", "last_mtime_ms": None}
    gap_ms = _now_ms() - latest_mtime
    status = "green" if gap_ms <= DECISION_INTERVAL_MS * STALE_MULTIPLE else "red"
    return {
        "status": status,
        "detail": f"LOG目录最近一次写入 {gap_ms // 60000} 分钟前(启发式心跳代理,非精确进程探测)",
        "last_mtime_ms": latest_mtime,
    }


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="AlphaLoop-Crypto Monitor (read-only)")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (APP_ROOT / "static" / "index.html").read_text(encoding="utf-8")


@app.get("/api/status")
def api_status() -> dict:
    return {
        "disclaimer": DISCLAIMER,
        "cold_start_state": _read_cold_start_state(),
        "circuit_state": _read_circuit_state(),
        "now_ms": _now_ms(),
    }


@app.get("/api/nav")
def api_nav() -> dict:
    rows = _read_nav_tsv()
    return {"rows": rows, "has_data": bool(rows)}


@app.get("/api/nav_intraday")
def api_nav_intraday(hours: int = 72) -> dict:
    rows = _read_nav_intraday(hours)
    return {"rows": rows, "has_data": bool(rows)}


@app.get("/api/nav_intraday_branches")
def api_nav_intraday_branches(hours: int = 72) -> dict:
    rows = _read_nav_intraday_branches(hours)
    return {"rows": rows, "has_data": bool(rows)}


@app.get("/api/branches")
def api_branches() -> dict:
    # random对照组已按用户要求(2026-07-14)下线,不再出现在分支列表里——
    # 见 scripts/ignite.py 模块docstring里对应的说明。
    evo = _read_registered_branches()
    branches = [
        {"branch": "main", "label": "main", "kind": "main"},
    ] + [
        {"branch": r["branch"], "label": r["branch"], "kind": "evo", "created_date": r.get("created_date")}
        for r in evo
        if "branch" in r
    ]
    return {"branches": branches, "has_data": True}


@app.get("/api/positions")
def api_positions(branch: str = MAIN_BRANCH) -> dict:
    portfolio = _read_portfolio_db(branch)
    if portfolio is None:
        return {
            "has_data": False, "branch": branch, "positions": [], "wallet_balance": None,
            "branch_dead": None, "total_pnl": None, "nav": None,
        }
    marks = _read_positions_marked(branch)
    for p in portfolio["positions"]:
        mark = marks.get(p["symbol"])
        p["mark_price"] = mark["mark_price"] if mark else None
        p["unrealized_pnl"] = mark["unrealized_pnl"] if mark else None
    nav = _read_latest_nav_for_branch(branch)
    total_pnl = (nav - INITIAL_CAPITAL_USDT) if nav is not None else None
    return {"has_data": True, "nav": nav, "total_pnl": total_pnl, **portfolio}


@app.get("/api/trades")
def api_trades(branch: str = MAIN_BRANCH, limit: int = 50) -> dict:
    trades = _read_trades(branch, limit)
    return {"has_data": bool(trades), "branch": branch, "trades": trades}


@app.get("/api/latest_advice")
def api_latest_advice() -> dict:
    text = _read_text("latest_advice.md")
    return {"has_data": text is not None, "content": text}


@app.get("/api/funding_summary")
def api_funding_summary(branch: str = MAIN_BRANCH) -> dict:
    records = [r for r in _read_jsonl("funding.jsonl") if r.get("branch") == branch]
    total = sum(r.get("amount", 0.0) for r in records)
    return {"has_data": bool(records), "branch": branch, "total_amount": total, "count": len(records)}


@app.get("/api/fees_summary")
def api_fees_summary(branch: str = MAIN_BRANCH) -> dict:
    records = [r for r in _read_jsonl("trades.jsonl") if r.get("branch") == branch]
    total = sum(r.get("fee", 0.0) for r in records)
    return {"has_data": bool(records), "branch": branch, "total_fee": total, "count": len(records)}


@app.get("/api/ratchet_log")
def api_ratchet_log(limit: int = 10) -> dict:
    records = _read_jsonl("ratchet_verdicts.jsonl")
    recent = records[-limit:][::-1] if records else []
    return {"has_data": bool(records), "verdicts": recent}


@app.get("/api/health")
def api_health() -> dict:
    return {
        "decision_gap": _decision_gap_health(),
        "funding_completeness": _funding_health(),
        "heartbeat": _heartbeat_health(),
    }


app.mount("/static", StaticFiles(directory=str(APP_ROOT / "static")), name="static")
