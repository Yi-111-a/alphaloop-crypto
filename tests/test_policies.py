"""
tests/test_policies.py —— M7 验收测试:策略协议包 + AST 静态检查 + 5个种子
策略(aggressive/conservative/momentum/carry/diversified)。

覆盖范围:
  1. lint_policy_source / lint_policy_file:分别构造含 requests.get /
     datetime.now() / random.random() 的违规源码,断言各自被拦;干净源码
     通过;5个真实种子策略文件自身全部通过 lint。
  2. load_policy:能加载5个真实策略;构造缺 DESCRIPTION 的临时模块断言
     报错清晰(消息里点名缺的是哪个属性)。
  3. 确定性:对每个策略,用合成的 StrategyContext 连续调用两次 decide,
     断言两次输出的 Decision 列表逐字段相等。
  4. 合法性:每个策略在"有明显趋势"和"横盘"两种合成ctx下的输出,逐条通过
     与 ASSET/strategy/trader.py._validate_decision_dict 等价的校验。
"""
from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import pytest
import yaml

from ASSET.strategy import trader as trader_module
from ASSET.strategy.policies import POLICIES_DIR, PolicyLoadError, StrategyContext, load_policy
from ASSET.strategy.policy_lint import lint_policy_file, lint_policy_source

REPO_ROOT = Path(__file__).resolve().parents[1]

POLICY_IDS = [
    "aggressive_v1",
    "conservative_v1",
    "momentum_v1",
    "carry_v1",
    "diversified_v1",
]

# 与 config.yaml 里 leverage.max 保持一致,而不是硬编码一个随意的数字——
# 这是"这条决策是否会被真实Simulator接受"的校验口径,理应对齐真实配置。
_CONFIG = yaml.safe_load((REPO_ROOT / "config.yaml").read_text(encoding="utf-8"))
_MAX_LEVERAGE = int(_CONFIG["leverage"]["max"])


# ---------------------------------------------------------------------------
# 合成K线/ctx 构造工具
# ---------------------------------------------------------------------------

_INTERVAL_MS = 4 * 3600 * 1000  # 与项目里4h K线周期一致
_BASE_TS = 1_700_000_000_000


def _make_bars(
    n: int,
    base_price: float,
    drift_pct: float,
    wiggle_amp: float = 0.0,
    wiggle_period: int = 7,
    flat_pattern: list[float] | None = None,
) -> pd.DataFrame:
    """构造 n 根确定性合成K线(无随机性,同一组参数每次结果完全一致)。

    - drift_pct != 0 时:复利趋势 base_price * (1+drift_pct)**i,叠加一个
      小幅正弦扰动(wiggle_amp),模拟"有明显趋势但不是完全平滑直线"的
      真实K线形状。
    - flat_pattern 给定时:忽略 drift/wiggle,直接按固定百分比偏移列表
      (以 base_price 为中枢,循环取用)生成横盘震荡数据,末尾刻意落在
      pattern 的最后一个元素上,用于制造"贴近区间边缘"的横盘测试场景。
    """
    rows = []
    prev_close = base_price
    for i in range(n):
        if flat_pattern is not None:
            offset_pct = flat_pattern[i % len(flat_pattern)]
            close = base_price * (1 + offset_pct / 100.0)
        else:
            trend = (1 + drift_pct) ** i
            wiggle = 1 + wiggle_amp * math.sin(2 * math.pi * i / wiggle_period)
            close = base_price * trend * wiggle
        open_ = prev_close
        high = max(open_, close) * 1.002
        low = min(open_, close) * 0.998
        rows.append(
            {
                "timestamp": _BASE_TS + i * _INTERVAL_MS,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": 1000.0,
            }
        )
        prev_close = close
    return pd.DataFrame(rows)


def _make_ctx(recent_bars: dict[str, pd.DataFrame]) -> StrategyContext:
    snapshot = {
        symbol: {"last": float(df["close"].iloc[-1])} for symbol, df in recent_bars.items()
    }
    last_ts = max(int(df["timestamp"].iloc[-1]) for df in recent_bars.values())
    return StrategyContext(
        ts=last_ts + _INTERVAL_MS,
        positions={},
        snapshot=snapshot,
        recent_bars=recent_bars,
        memory_context=[],
    )


def _trending_ctx(n: int = 80) -> StrategyContext:
    """两个标的:一个明显上涨趋势、一个明显下跌趋势,叠加小幅正弦扰动。"""
    up = _make_bars(n, base_price=100.0, drift_pct=0.006, wiggle_amp=0.004, wiggle_period=7)
    down = _make_bars(n, base_price=50.0, drift_pct=-0.006, wiggle_amp=0.004, wiggle_period=7)
    return _make_ctx({"AAA/USDT:USDT": up, "BBB/USDT:USDT": down})


def _flat_ctx(n: int = 80) -> StrategyContext:
    """横盘震荡数据,振幅约±0.9%,末尾落在pattern最后一个点(贴近区间下沿),
    用于同时验证:动量类策略在无趋势时保持观望(返回空列表)、carry代理策略
    在低波动+贴近区间边缘时能够触发均值回归信号。"""
    pattern = [0.0, 0.3, 0.6, 0.9, 0.6, 0.3, 0.0, -0.3, -0.6, -0.9]
    assert n % len(pattern) == 0  # 保证末尾落在 pattern[-1](区间下沿)
    a = _make_bars(n, base_price=100.0, drift_pct=0.0, flat_pattern=pattern)
    b = _make_bars(n, base_price=50.0, drift_pct=0.0, flat_pattern=pattern)
    return _make_ctx({"AAA/USDT:USDT": a, "BBB/USDT:USDT": b})


def _validate_decision(decision, valid_symbols: set[str]) -> list[str]:
    d = asdict(decision)
    return trader_module._validate_decision_dict(
        d, max_leverage=_MAX_LEVERAGE, valid_symbols=valid_symbols
    )


# ---------------------------------------------------------------------------
# 1. lint_policy_source / lint_policy_file
# ---------------------------------------------------------------------------


def test_lint_forbids_network_import_and_call():
    source = """
import requests

def decide(ctx):
    requests.get("http://example.com")
    return []

REQUIRED_HISTORY_BARS = 1
DESCRIPTION = "dirty policy for lint test"
"""
    violations = lint_policy_source(source)
    assert violations, "expected at least one violation for requests import/call"
    assert any("requests" in v for v in violations)


def test_lint_forbids_wallclock_read():
    source = """
import datetime

def decide(ctx):
    now = datetime.datetime.now()
    return []

REQUIRED_HISTORY_BARS = 1
DESCRIPTION = "dirty policy for lint test"
"""
    violations = lint_policy_source(source)
    assert violations, "expected at least one violation for datetime.now()"
    assert any("wallclock" in v for v in violations)


def test_lint_forbids_random_module():
    source = """
import random

def decide(ctx):
    x = random.random()
    return []

REQUIRED_HISTORY_BARS = 1
DESCRIPTION = "dirty policy for lint test"
"""
    violations = lint_policy_source(source)
    assert violations, "expected at least one violation for random module usage"
    assert any("random" in v.lower() or "import" in v for v in violations)


def test_lint_forbids_os_environ_subprocess_exec_eval_and_open_write():
    source = """
import os
import subprocess

def decide(ctx):
    _ = os.environ["SECRET"]
    subprocess.run(["ls"])
    exec("print(1)")
    eval("1+1")
    with open("out.txt", "w") as f:
        f.write("x")
    return []

REQUIRED_HISTORY_BARS = 1
DESCRIPTION = "dirty policy for lint test"
"""
    violations = lint_policy_source(source)
    joined = " | ".join(violations)
    assert "environ" in joined
    assert "subprocess" in joined
    assert "exec" in joined
    assert "eval" in joined
    assert "open" in joined


def test_lint_clean_source_passes():
    source = """
def decide(ctx):
    return []

REQUIRED_HISTORY_BARS = 1
DESCRIPTION = "clean policy for lint test"
"""
    assert lint_policy_source(source) == []


def test_lint_syntax_error_reported_not_raised():
    violations = lint_policy_source("def decide(ctx:\n    return [")
    assert violations
    assert any("syntax" in v.lower() for v in violations)


@pytest.mark.parametrize("policy_id", POLICY_IDS)
def test_seed_policy_files_pass_lint(policy_id):
    path = POLICIES_DIR / f"{policy_id}.py"
    assert path.exists(), f"expected seed policy file at {path}"
    violations = lint_policy_file(path)
    assert violations == [], f"{policy_id}.py failed lint: {violations}"


# ---------------------------------------------------------------------------
# 2. load_policy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("policy_id", POLICY_IDS)
def test_load_policy_loads_seed_policies(policy_id):
    module = load_policy(policy_id)
    assert callable(module.decide)
    assert isinstance(module.REQUIRED_HISTORY_BARS, int)
    assert module.REQUIRED_HISTORY_BARS > 0
    assert isinstance(module.DESCRIPTION, str)
    assert len(module.DESCRIPTION) > 0


def test_load_policy_missing_file_raises_clear_error():
    with pytest.raises(PolicyLoadError, match="not found"):
        load_policy("this_policy_does_not_exist_v1")


def test_load_policy_missing_description_raises_clear_error(tmp_path):
    # load_policy() 按spec只从 ASSET/strategy/policies/ 固定目录加载,所以要
    # 测试"缺失导出属性"的报错行为,必须把临时坏模块真的放进这个目录里,
    # 用完立刻删除,不影响其他测试或真实策略清单。
    broken_id = "_test_broken_missing_description_v1"
    broken_path = POLICIES_DIR / f"{broken_id}.py"
    broken_path.write_text(
        "def decide(ctx):\n    return []\n\nREQUIRED_HISTORY_BARS = 1\n",
        encoding="utf-8",
    )
    try:
        with pytest.raises(PolicyLoadError, match="DESCRIPTION"):
            load_policy(broken_id)
    finally:
        broken_path.unlink(missing_ok=True)


def test_load_policy_missing_decide_raises_clear_error(tmp_path):
    broken_id = "_test_broken_missing_decide_v1"
    broken_path = POLICIES_DIR / f"{broken_id}.py"
    broken_path.write_text(
        "REQUIRED_HISTORY_BARS = 1\nDESCRIPTION = 'no decide()'\n",
        encoding="utf-8",
    )
    try:
        with pytest.raises(PolicyLoadError, match="decide"):
            load_policy(broken_id)
    finally:
        broken_path.unlink(missing_ok=True)


def test_load_policy_reloads_latest_version_from_disk():
    # 内环会反复重写同一个 policy_id 对应的文件——load_policy 每次都必须
    # 从磁盘重新读取,而不是复用进程内第一次 import 时的缓存版本。
    reload_id = "_test_reload_tmp_v1"
    reload_path = POLICIES_DIR / f"{reload_id}.py"
    try:
        reload_path.write_text(
            "def decide(ctx):\n    return []\n\n"
            "REQUIRED_HISTORY_BARS = 1\nDESCRIPTION = 'version 1'\n",
            encoding="utf-8",
        )
        module_v1 = load_policy(reload_id)
        assert module_v1.DESCRIPTION == "version 1"

        reload_path.write_text(
            "def decide(ctx):\n    return []\n\n"
            "REQUIRED_HISTORY_BARS = 1\nDESCRIPTION = 'version 2'\n",
            encoding="utf-8",
        )
        module_v2 = load_policy(reload_id)
        assert module_v2.DESCRIPTION == "version 2"
    finally:
        reload_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# 3 & 4. 确定性 + 合法性(逐策略、逐场景)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("policy_id", POLICY_IDS)
@pytest.mark.parametrize("ctx_factory", [_trending_ctx, _flat_ctx], ids=["trending", "flat"])
def test_policy_is_deterministic_and_produces_valid_decisions(policy_id, ctx_factory):
    module = load_policy(policy_id)
    ctx = ctx_factory()

    decisions_1 = module.decide(ctx)
    decisions_2 = module.decide(ctx)

    # 确定性:同一个ctx连续调用两次,必须逐字段相同。
    assert decisions_1 == decisions_2

    # 合法性:每条产出的Decision都必须通过与Simulator九步校验链里
    # "决策本身是否合法"等价的规则(Trader._validate_decision_dict)。
    valid_symbols = set(ctx.snapshot.keys())
    for decision in decisions_1:
        errors = _validate_decision(decision, valid_symbols)
        assert errors == [], f"{policy_id} produced invalid decision: {errors}\n{decision}"
        # falsifier_condition 的方向必须与action一致(做多用price<N,做空用price>N)。
        if decision.action == "open_long":
            assert decision.falsifier_condition.startswith("price<")
        elif decision.action == "open_short":
            assert decision.falsifier_condition.startswith("price>")


def test_at_least_one_policy_trades_in_trending_scenario():
    ctx = _trending_ctx()
    any_decisions = False
    for policy_id in POLICY_IDS:
        module = load_policy(policy_id)
        if module.decide(ctx):
            any_decisions = True
    assert any_decisions, "expected at least one seed policy to act on a clear trend"


def test_carry_policy_trades_in_flat_scenario():
    module = load_policy("carry_v1")
    ctx = _flat_ctx()
    decisions = module.decide(ctx)
    assert decisions, "carry_v1 (range/low-vol proxy) should act on the flat synthetic data"
    assert decisions[0].action in {"open_long", "open_short"}


def test_momentum_family_holds_in_flat_scenario():
    ctx = _flat_ctx()
    for policy_id in ["momentum_v1", "aggressive_v1", "conservative_v1"]:
        module = load_policy(policy_id)
        assert module.decide(ctx) == [], f"{policy_id} should stay flat (no signal) on sideways data"


def test_diversified_policy_can_open_multiple_symbols_in_trending_scenario():
    module = load_policy("diversified_v1")
    ctx = _trending_ctx()
    decisions = module.decide(ctx)
    symbols = {d.symbol for d in decisions}
    assert symbols == {"AAA/USDT:USDT", "BBB/USDT:USDT"}
    actions = {d.symbol: d.action for d in decisions}
    assert actions["AAA/USDT:USDT"] == "open_long"
    assert actions["BBB/USDT:USDT"] == "open_short"


def test_no_signal_data_returns_empty_list_not_hold_decision():
    """规范约定:无信号/数据不足时,policy返回空列表(而不是显式hold Decision)。
    用历史不足的合成ctx(远少于任何策略的REQUIRED_HISTORY_BARS)验证这一点。"""
    short_bars = _make_bars(3, base_price=100.0, drift_pct=0.01)
    ctx = _make_ctx({"AAA/USDT:USDT": short_bars})
    for policy_id in POLICY_IDS:
        module = load_policy(policy_id)
        assert module.decide(ctx) == [], f"{policy_id} should return [] when history is insufficient"


# ---------------------------------------------------------------------------
# 5. M7资金费率感官注入:StrategyContext.recent_funding 默认值 + 旧策略不受影响
# ---------------------------------------------------------------------------


def test_strategy_context_recent_funding_defaults_to_empty_dict_without_crashing():
    """不传 recent_funding 构造 StrategyContext 不能报错——所有既有构造点
    (测试/LOCKED/backtest_engine.py/ASSET/strategy/policy_trader.py 升级前的
    调用方式)在这个字段存在之前就已经写好,必须继续可用、且默认值是空dict
    (而不是None——policy代码里`ctx.recent_funding.get(symbol)`这种写法不该
    因为默认值是None而崩)。"""
    ctx = StrategyContext(
        ts=1_700_000_000_000,
        positions={},
        snapshot={},
        recent_bars={},
    )
    assert ctx.recent_funding == {}


def test_strategy_context_recent_spot_and_recent_oi_default_to_empty_dict_without_crashing():
    """M9新增的recent_spot/recent_oi字段同recent_funding一样,不传也不能
    报错,默认值是空dict(而不是None)。"""
    ctx = StrategyContext(
        ts=1_700_000_000_000,
        positions={},
        snapshot={},
        recent_bars={},
    )
    assert ctx.recent_spot == {}
    assert ctx.recent_oi == {}


@pytest.mark.parametrize("policy_id", ["momentum_v1", "aggressive_v1", "conservative_v1", "diversified_v1"])
@pytest.mark.parametrize("ctx_factory", [_trending_ctx, _flat_ctx], ids=["trending", "flat"])
def test_non_carry_seed_policies_ignore_recent_funding_field(policy_id, ctx_factory):
    """规范约定:除carry_v1外,种子策略完全不读ctx.recent_funding——本次升级
    给StrategyContext加字段不应该悄悄改变这些策略在任何合成场景下的输出。
    往ctx里塞进非空的recent_funding数据,断言输出与不塞时逐字段相同。"""
    module = load_policy(policy_id)

    ctx_without = ctx_factory()
    decisions_without = module.decide(ctx_without)

    ctx_base = ctx_factory()
    funding_df = pd.DataFrame(
        {"timestamp": [ctx_base.ts - 1000], "funding_rate": [0.001]}
    )
    ctx_with_funding = StrategyContext(
        ts=ctx_base.ts,
        positions=ctx_base.positions,
        snapshot=ctx_base.snapshot,
        recent_bars=ctx_base.recent_bars,
        memory_context=ctx_base.memory_context,
        recent_funding={sym: funding_df for sym in ctx_base.snapshot},
    )
    decisions_with = module.decide(ctx_with_funding)

    assert decisions_without == decisions_with


@pytest.mark.parametrize("policy_id", POLICY_IDS)
@pytest.mark.parametrize("ctx_factory", [_trending_ctx, _flat_ctx], ids=["trending", "flat"])
def test_all_seed_policies_ignore_recent_spot_and_recent_oi_fields(policy_id, ctx_factory):
    """M9新增的recent_spot/recent_oi字段:全部5个种子策略(含carry_v1)目前
    都不读这两个字段——只有recent_funding被carry_v1消费,basis/OI感官暂时
    还没有任何种子策略消费它。往ctx里塞进非空的recent_spot/recent_oi数据,
    断言输出与不塞时逐字段相同,确保加字段这件事本身不悄悄改变任何既有
    策略在任何合成场景下的输出。"""
    module = load_policy(policy_id)

    ctx_without = ctx_factory()
    decisions_without = module.decide(ctx_without)

    ctx_base = ctx_factory()
    spot_df = pd.DataFrame(
        {
            "timestamp": [ctx_base.ts - 1000],
            "open": [99.0], "high": [100.0], "low": [98.0], "close": [99.5], "volume": [1000.0],
        }
    )
    oi_df = pd.DataFrame({"timestamp": [ctx_base.ts - 1000], "open_interest": [5_000_000.0]})
    ctx_with_extra_senses = StrategyContext(
        ts=ctx_base.ts,
        positions=ctx_base.positions,
        snapshot=ctx_base.snapshot,
        recent_bars=ctx_base.recent_bars,
        memory_context=ctx_base.memory_context,
        recent_funding=ctx_base.recent_funding,
        recent_spot={sym: spot_df for sym in ctx_base.snapshot},
        recent_oi={sym: oi_df for sym in ctx_base.snapshot},
    )
    decisions_with = module.decide(ctx_with_extra_senses)

    assert decisions_without == decisions_with
