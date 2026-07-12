from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from ASSET.memory.engine import MemoryStore
from ASSET.strategy.researcher import Researcher, GENERIC_HYPOTHESIS_TEMPLATES


# ---------------------------------------------------------------------------
# 公共 fakes
# ---------------------------------------------------------------------------


class FakeMemoryStore:
    """记录每次 write() 调用参数,便于断言 layer/ts 是否符合预期,不做真实检索。"""

    def __init__(self):
        self.writes = []

    def write(self, content, ts, layer, importance=1.0):
        rec = {"content": content, "ts": ts, "layer": layer, "importance": importance}
        self.writes.append(rec)
        return rec


def make_fixed_llm(response_text: str):
    """忽略prompt,永远返回固定文本的 fake llm_client。"""

    def _client(prompt: str) -> str:
        return response_text

    return _client


def make_fake_search(canned_results):
    def _search(query: str):
        return list(canned_results)

    return _search


# ---------------------------------------------------------------------------
# 1. daily_research: 有 search_client,fake llm 产出规范 findings -> 写文件
# ---------------------------------------------------------------------------


def test_daily_research_with_search_client_writes_findings_md(tmp_path):
    notes_dir = tmp_path / "research_notes"

    canned_search = [
        {
            "source": "arXiv:2401.99999",
            "title": "LLM Agents for Crypto Trading",
            "summary": "A survey of LLM-driven trading agents.",
            "url": "https://arxiv.org/abs/2401.99999",
        },
        {
            "source": "github.com/example/quant-trending",
            "title": "Trending quant repo",
            "summary": "Momentum-based perp strategy.",
            "url": "https://github.com/example/quant-trending",
        },
    ]

    findings = [
        {
            "source": "arXiv:2401.99999",
            "core_idea": "LLM agents can synthesize structured trading hypotheses from literature.",
            "testable_hypothesis": "Agents referencing layered memory outperform stateless baselines.",
            "suggested_experiment": "A/B test agent with/without L1-L3 memory over 30 days.",
        },
        {
            "source": "github.com/example/quant-trending",
            "core_idea": "Momentum persists in trending majors over multi-day windows.",
            "testable_hypothesis": "BTC/ETH show positive autocorrelation in 4h returns after breakout.",
            "suggested_experiment": "Backtest momentum entry after N consecutive up-closes.",
        },
    ]

    llm = make_fixed_llm(json.dumps(findings))
    search = make_fake_search(canned_search)

    researcher = Researcher(
        llm_client=llm,
        memory_store=FakeMemoryStore(),
        search_client=search,
        research_notes_dir=notes_dir,
    )

    out_path = researcher.daily_research(ts=1_700_000_000_000, date_str="2026-07-12")

    expected_path = notes_dir / "2026-07-12.md"
    assert out_path == expected_path
    assert out_path.exists()

    content = out_path.read_text(encoding="utf-8")
    for f in findings:
        assert f["source"] in content
        assert f["core_idea"] in content
        assert f["testable_hypothesis"] in content
        assert f["suggested_experiment"] in content


# ---------------------------------------------------------------------------
# 2. daily_research: search_client=None,只靠 llm_client 合成也不崩溃
# ---------------------------------------------------------------------------


def test_daily_research_without_search_client(tmp_path):
    notes_dir = tmp_path / "research_notes"

    findings = [
        {
            "source": "internal-knowledge",
            "core_idea": "Funding-rate carry is a recurring theme in perp research.",
            "testable_hypothesis": "Persistently positive funding predicts short-side crowding.",
            "suggested_experiment": "Correlate funding rate sign with next-8h realized returns.",
        }
    ]
    llm = make_fixed_llm(json.dumps(findings))

    researcher = Researcher(
        llm_client=llm,
        memory_store=FakeMemoryStore(),
        search_client=None,
        research_notes_dir=notes_dir,
    )

    out_path = researcher.daily_research(ts=1_700_000_000_000, date_str="2026-07-13")

    assert out_path == notes_dir / "2026-07-13.md"
    assert out_path.exists()
    content = out_path.read_text(encoding="utf-8")
    assert "Funding-rate carry" in content
    assert "Persistently positive funding" in content


def test_daily_research_falls_back_when_llm_never_produces_valid_json(tmp_path):
    notes_dir = tmp_path / "research_notes"
    llm = make_fixed_llm("not json at all")

    researcher = Researcher(
        llm_client=llm,
        memory_store=FakeMemoryStore(),
        search_client=None,
        research_notes_dir=notes_dir,
        max_retries=2,
    )

    out_path = researcher.daily_research(ts=1_700_000_000_000, date_str="2026-07-14")
    assert out_path.exists()
    content = out_path.read_text(encoding="utf-8")
    # fallback finding is written, file is still well-formed markdown with all 4 fields
    assert "Source" in content and "Core idea" in content
    assert "Testable hypothesis" in content and "Suggested experiment" in content


# ---------------------------------------------------------------------------
# 3. run_cold_start_research: 6条真实假设 -> 补足到 >= min_hypotheses(10)
# ---------------------------------------------------------------------------


def _six_fake_hypotheses():
    return [
        {
            "hypothesis": "Momentum breakout signal AlphaTestUniqueTerm123 for BTC",
            "rationale": "BTC shows short-term continuation after high-volume breakout bars.",
        },
        {
            "hypothesis": "Mean reversion after funding rate extremes",
            "rationale": "Extreme positive funding tends to precede short-term pullbacks.",
        },
        {
            "hypothesis": "Correlation breakdown as risk signal",
            "rationale": "Altcoin/BTC correlation collapse often precedes liquidity stress.",
        },
        {
            "hypothesis": "Volatility regime clustering",
            "rationale": "Realized volatility clusters into distinguishable high/low regimes.",
        },
        {
            "hypothesis": "Reflection-driven position sizing discipline",
            "rationale": "Sizing should shrink after consecutive falsified theses.",
            "permanent": True,
        },
        {
            "hypothesis": "Sentiment lag relative to price action",
            "rationale": "Price often leads sentiment confirmation by several hours.",
        },
    ]


def test_cold_start_pads_to_min_hypotheses(tmp_path):
    genesis_path = tmp_path / "research_notes" / "genesis.md"
    llm = make_fixed_llm(json.dumps(_six_fake_hypotheses()))
    fake_memory = FakeMemoryStore()

    researcher = Researcher(
        llm_client=llm,
        memory_store=fake_memory,
        search_client=None,
        genesis_path=genesis_path,
    )

    universe = ["BTC/USDT:USDT", "ETH/USDT:USDT"]
    price_history = {
        "BTC/USDT:USDT": [100.0, 102.0, 101.0, 105.0, 103.0],
        "ETH/USDT:USDT": [10.0, 10.1, 9.9, 10.3, 10.2],
    }

    result = researcher.run_cold_start_research(
        ts=1_700_000_000_000,
        universe_symbols=universe,
        price_history=price_history,
        min_hypotheses=10,
    )

    assert result["hypothesis_count"] >= 10
    assert len(result["hypothesis_ids"]) == result["hypothesis_count"]
    assert result["hypothesis_ids"][:3] == ["H1", "H2", "H3"]

    assert genesis_path.exists()
    content = genesis_path.read_text(encoding="utf-8")

    header_re = re.compile(r"^## H(\d+):", re.MULTILINE)
    found_ids = sorted(int(m.group(1)) for m in header_re.finditer(content))
    assert found_ids == list(range(1, result["hypothesis_count"] + 1))
    assert result["hypothesis_count"] >= 10

    # padded entries should be honestly labeled, not silently passed off as real findings
    assert "[GENERIC]" in content

    # each hypothesis was also written to memory_store
    assert len(fake_memory.writes) == result["hypothesis_count"]
    for w in fake_memory.writes:
        assert w["layer"] in ("L2", "L3")
        assert w["ts"] == 1_700_000_000_000


# ---------------------------------------------------------------------------
# 4. run_cold_start_research + 真实 MemoryStore: 写入可被检索到(真实往返)
# ---------------------------------------------------------------------------


def test_cold_start_hypotheses_written_and_retrievable_from_real_memory_store(tmp_path):
    genesis_path = tmp_path / "research_notes" / "genesis.md"
    db_path = tmp_path / "memory.db"
    memory_store = MemoryStore(db_path=db_path)

    llm = make_fixed_llm(json.dumps(_six_fake_hypotheses()))
    researcher = Researcher(
        llm_client=llm,
        memory_store=memory_store,
        search_client=None,
        genesis_path=genesis_path,
    )

    ts = 1_700_000_000_000
    result = researcher.run_cold_start_research(
        ts=ts,
        universe_symbols=["BTC/USDT:USDT"],
        price_history={"BTC/USDT:USDT": [100.0, 101.0, 99.0, 103.0]},
        min_hypotheses=10,
    )

    assert result["hypothesis_count"] >= 10

    # 直接查sqlite确认每条都落盘且layer合法(L2/L3),数量与返回值一致
    rows = memory_store._conn.execute("SELECT layer, content FROM memory_records").fetchall()
    assert len(rows) == result["hypothesis_count"]
    for row in rows:
        assert row["layer"] in ("L2", "L3")

    # 真实检索往返(不是mock):用一个只出现在其中一条假设里的独特词查询
    retrieved = memory_store.retrieve("AlphaTestUniqueTerm123 momentum breakout", query_ts=ts, top_k=3)
    assert len(retrieved) > 0
    assert any("AlphaTestUniqueTerm123" in record.content for record, _score in retrieved)

    memory_store.close()


# ---------------------------------------------------------------------------
# 5. 波动/趋势画像:真实数值出现在 genesis.md 里,不是占位符
# ---------------------------------------------------------------------------


def test_cold_start_profile_numbers_appear_in_genesis(tmp_path):
    genesis_path = tmp_path / "research_notes" / "genesis.md"
    llm = make_fixed_llm(json.dumps(_six_fake_hypotheses()))
    researcher = Researcher(
        llm_client=llm,
        memory_store=FakeMemoryStore(),
        search_client=None,
        genesis_path=genesis_path,
    )

    btc_closes = [100.0, 110.0, 90.0, 105.0]
    result = researcher.run_cold_start_research(
        ts=1_700_000_000_000,
        universe_symbols=["BTC/USDT:USDT"],
        price_history={"BTC/USDT:USDT": btc_closes},
        min_hypotheses=10,
    )

    expected_trend_pct = (btc_closes[-1] - btc_closes[0]) / btc_closes[0] * 100
    content = genesis_path.read_text(encoding="utf-8")

    assert "n_points: 4" in content
    assert f"{expected_trend_pct:.4f}%" in content

    # profile dict returned alongside also carries real numbers (not None) for this symbol
    btc_profile = next(p for p in result["profiles"] if p["symbol"] == "BTC/USDT:USDT")
    assert btc_profile["n_points"] == 4
    assert btc_profile["trend_pct"] == pytest.approx(expected_trend_pct)
    assert btc_profile["volatility_pct"] is not None


def test_cold_start_funding_rate_summary_appears_when_provided(tmp_path):
    genesis_path = tmp_path / "research_notes" / "genesis.md"
    llm = make_fixed_llm(json.dumps(_six_fake_hypotheses()))
    researcher = Researcher(
        llm_client=llm,
        memory_store=FakeMemoryStore(),
        search_client=None,
        genesis_path=genesis_path,
    )

    result = researcher.run_cold_start_research(
        ts=1_700_000_000_000,
        universe_symbols=["BTC/USDT:USDT"],
        price_history={"BTC/USDT:USDT": [100.0, 101.0, 99.5, 102.0]},
        funding_rate_history={"BTC/USDT:USDT": [0.0001, 0.0002, -0.0001, 0.0003]},
        min_hypotheses=10,
    )

    content = genesis_path.read_text(encoding="utf-8")
    assert "Funding Rate Distribution" in content
    assert len(result["funding_summaries"]) == 1
    fs = result["funding_summaries"][0]
    assert f"{fs['mean']:.6f}" in content


# ---------------------------------------------------------------------------
# 6. 静态防回归护栏:researcher.py 里不允许出现墙钟时间调用
# ---------------------------------------------------------------------------


def test_no_wallclock_calls_in_researcher_source():
    import ast

    import ASSET.strategy.researcher as researcher_module

    source = Path(researcher_module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)

    forbidden_modules = {"time", "datetime"}
    offenders: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in forbidden_modules:
                    offenders.append(f"import {alias.name} (line {node.lineno})")
        elif isinstance(node, ast.ImportFrom):
            if node.module in forbidden_modules:
                offenders.append(f"from {node.module} import ... (line {node.lineno})")
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr in {"time", "now", "utcnow", "today"}:
                offenders.append(f"call .{func.attr}(...) (line {node.lineno})")

    assert offenders == [], f"forbidden wall-clock references found in researcher.py: {offenders}"


# ---------------------------------------------------------------------------
# sanity: generic template pool itself has enough distinct entries to be a
# reasonable padding source (documentation-as-test)
# ---------------------------------------------------------------------------


def test_generic_hypothesis_template_pool_has_at_least_ten_entries():
    assert len(GENERIC_HYPOTHESIS_TEMPLATES) >= 10
