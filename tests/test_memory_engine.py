"""
tests/test_memory_engine.py —— M2 分层记忆引擎验收测试。

按 spec §6 M2 验收标准中的"记忆检索返回结果随时间衰减"，以及本任务简报里
明确要求的"THE TIME-TRAVEL INJECTION TEST"（与 M1 插针爆仓测试同等优先级）。

测试全程使用一个自定义的、确定性的、纯本地的 fake embedder(见
`_make_word_overlap_embedder`)注入 MemoryStore，不依赖网络、不依赖任何一次性
模型下载 —— 整个测试套件应该在毫秒级跑完。sentence-transformers 的真实
embedder 只在一个可选的、失败即跳过的冒烟测试里被触碰一次。
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import pytest

from ASSET.memory.engine import MemoryStore, hashing_embedder

_MS_PER_DAY = 86_400_000


def _make_word_overlap_embedder():
    """确定性的 fake embedder，供测试注入使用。

    离线、无 ML 依赖：把文本切成词，用一个惰性增长的词表把每个词映射到一个
    固定维度向量里的一个下标(计数向量)，最后 L2 归一化。同一份文本(哪怕是不
    同调用)只要词表已经见过其中的词，就会得到完全相同的向量——这保证了
    "content 与 query 逐字相同" ⇒ "cosine similarity 精确等于 1.0"，测试断言
    才能做到数值精确而不是"大致差不多"。

    每个测试都应该调用本函数拿一个新的 embedder 实例（配一个独立的
    MemoryStore），避免测试之间通过词表产生隐式耦合。
    """
    vocab: dict[str, int] = {}
    dim = 256

    def embed(text: str) -> list[float]:
        tokens = re.findall(r"[a-z0-9]+", (text or "").lower())
        vec = [0.0] * dim
        for tok in tokens:
            idx = vocab.setdefault(tok, len(vocab) % dim)
            vec[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec] if norm > 0.0 else vec

    return embed


def _make_store(tmp_path, name: str = "memory.db", embedder=None) -> MemoryStore:
    embedder = embedder if embedder is not None else _make_word_overlap_embedder()
    return MemoryStore(db_path=tmp_path / name, embedder=embedder)


# ---------------------------------------------------------------------------
# 1. 衰减测试(M2 显式验收标准)
# ---------------------------------------------------------------------------


def test_memory_decay_l1_matches_exact_formula(tmp_path):
    """L1(时间常数τ=3天,spec v1.3术语勘误:是e-折时间不是半衰期)的记忆，3天后
    的检索得分必须严格低于写入时刻，且数值上精确匹配 §3.4 公式 exp(-Δt/τ)
    （不是"约等于0.5"，见 engine.py 模块docstring里对这个公式的说明：
    exp(-1)≈0.3679，不是0.5——产品侧裁决保留此公式，只改名字）。
    """
    store = _make_store(tmp_path)
    query = "BTC funding rate spiked sharply positive overnight"
    write_ts = 1_700_000_000_000

    written = store.write(query, ts=write_ts, layer="L1")

    results_at_write = store.retrieve(query, query_ts=write_ts, top_k=5)
    assert len(results_at_write) == 1
    record0, score0 = results_at_write[0]
    assert record0.id == written.id
    # 写入时刻 Δt=0 ⇒ decay=exp(0)=1 ⇒ score = relevance(=1.0,逐字匹配) * 1 * importance(=1.0)
    assert score0 == pytest.approx(1.0, abs=1e-9)

    three_days_later = write_ts + 3 * _MS_PER_DAY
    results_later = store.retrieve(query, query_ts=three_days_later, top_k=5)
    assert len(results_later) == 1
    record1, score1 = results_later[0]
    assert record1.id == written.id

    expected_decay = math.exp(-1.0)  # Δt=τ=3天 ⇒ exp(-3/3)=exp(-1)
    assert score1 == pytest.approx(score0 * expected_decay, rel=1e-9)
    assert score1 < score0  # 严格更低,不只是"大致差不多"


# ---------------------------------------------------------------------------
# 2. THE TIME-TRAVEL INJECTION TEST —— 与 M1 插针爆仓测试同等优先级
# ---------------------------------------------------------------------------


def test_time_travel_injection_future_record_is_invisible(tmp_path):
    """一条 ts 晚于 query_ts 的"未来"记录，即使 content 与 query 逐字相同
    (完美语义匹配，若无时间闸门必然排名第一)，也必须被 retrieve() 完全排除
    ——不能出现在结果列表里，哪怕位置排在最后。这是防"未来数据污染当下决策"
    信息泄漏的核心断言。
    """
    store = _make_store(tmp_path)
    injected_query = "SOL on-chain whale accumulation detected before breakout"

    decision_ts = 1_700_000_000_000
    future_ts = decision_ts + 10 * _MS_PER_DAY  # 记录被"未来"污染,晚于决策时刻

    future_record = store.write(injected_query, ts=future_ts, layer="L1")

    results = store.retrieve(injected_query, query_ts=decision_ts, top_k=10)

    returned_ids = [record.id for record, _score in results]
    assert future_record.id not in returned_ids
    # 更强断言:结果集应该完全为空(店里唯一一条记录就是这条未来记录)
    assert results == []


def test_time_travel_boundary_ts_equal_query_ts_is_visible(tmp_path):
    """边界情况:record.ts == query_ts(不是严格大于)。约定为"可见"——决策发生
    的那一刻写入的记忆，在同一时刻的决策里是可用的。spec 原文是
    "record.ts > query_ts 视为不存在"(严格大于)，因此相等不落入不可见区间。
    """
    store = _make_store(tmp_path)
    query = "ETH liquidity depth thinning across major venues"
    decision_ts = 1_700_000_000_000

    written = store.write(query, ts=decision_ts, layer="L1")  # ts 恰好等于 query_ts

    results = store.retrieve(query, query_ts=decision_ts, top_k=10)
    returned_ids = [record.id for record, _score in results]
    assert written.id in returned_ids


def test_time_travel_one_ms_future_is_already_invisible(tmp_path):
    """比 query_ts 晚哪怕 1 毫秒的记录也必须不可见 —— 确认边界是"严格大于"
    而不是被某个粗粒度的天/小时取整意外放过。
    """
    store = _make_store(tmp_path)
    query = "DOGE social sentiment surge ahead of listing rumor"
    decision_ts = 1_700_000_000_000

    store.write(query, ts=decision_ts + 1, layer="L1")

    results = store.retrieve(query, query_ts=decision_ts, top_k=10)
    assert results == []


# ---------------------------------------------------------------------------
# 3. L3 永久层不衰减
# ---------------------------------------------------------------------------


def test_l3_records_do_not_decay(tmp_path):
    store = _make_store(tmp_path)
    query = "leverage above 8x on illiquid alts has historically preceded forced unwinds"
    write_ts = 1_700_000_000_000

    written = store.write(query, ts=write_ts, layer="L3")

    immediate = store.retrieve(query, query_ts=write_ts, top_k=5)
    much_later = store.retrieve(query, query_ts=write_ts + 100 * _MS_PER_DAY, top_k=5)

    assert len(immediate) == 1 and len(much_later) == 1
    assert immediate[0][0].id == written.id
    assert much_later[0][0].id == written.id
    assert immediate[0][1] == pytest.approx(much_later[0][1], abs=1e-12)


# ---------------------------------------------------------------------------
# 4. top_k 截断
# ---------------------------------------------------------------------------


def test_top_k_truncates_and_ranks_by_score_descending(tmp_path):
    store = _make_store(tmp_path)
    query = "funding rate turned deeply negative across perp venues"
    ts = 1_700_000_000_000

    # 用不同的 importance 制造严格递增/递减的分数，避免并列名次带来的排序歧义。
    importances = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    written = [store.write(query, ts=ts, layer="L1", importance=imp) for imp in importances]

    top_k = 3
    results = store.retrieve(query, query_ts=ts, top_k=top_k)

    assert len(results) == top_k
    scores = [score for _record, score in results]
    assert scores == sorted(scores, reverse=True)

    # 应该是 importance 最高的三条(6,5,4),按分数降序
    expected_ids = {written[5].id, written[4].id, written[3].id}
    assert {record.id for record, _score in results} == expected_ids


# ---------------------------------------------------------------------------
# 5. layers 过滤
# ---------------------------------------------------------------------------


def test_layers_filter_restricts_to_requested_layers(tmp_path):
    store = _make_store(tmp_path)
    query = "BTC dominance rotating into large-cap alts"
    ts = 1_700_000_000_000

    l1 = store.write(query, ts=ts, layer="L1", importance=1.0)
    l2 = store.write(query, ts=ts, layer="L2", importance=5.0)  # 分数会比L1高很多
    l3 = store.write(query, ts=ts, layer="L3", importance=5.0)  # 分数也会比L1高

    results = store.retrieve(query, query_ts=ts, top_k=10, layers=["L1"])

    returned_ids = {record.id for record, _score in results}
    assert returned_ids == {l1.id}
    assert l2.id not in returned_ids
    assert l3.id not in returned_ids
    assert all(record.layer == "L1" for record, _score in results)


# ---------------------------------------------------------------------------
# 6. importance 加权
# ---------------------------------------------------------------------------


def test_importance_weighting_scales_score_linearly(tmp_path):
    store = _make_store(tmp_path)
    query = "open interest climbing while price is flat, classic squeeze setup"
    ts = 1_700_000_000_000

    low = store.write(query, ts=ts, layer="L2", importance=1.0)
    high = store.write(query, ts=ts, layer="L2", importance=3.0)

    results = store.retrieve(query, query_ts=ts, top_k=10)
    scores_by_id = {record.id: score for record, score in results}

    assert scores_by_id[high.id] > scores_by_id[low.id]
    # 同 content/ts/layer ⇒ relevance、decay 完全相同,分数比值应精确等于
    # importance 比值(3倍)
    assert scores_by_id[high.id] == pytest.approx(scores_by_id[low.id] * 3.0, rel=1e-9)


# ---------------------------------------------------------------------------
# 7. score_one() 独立打分入口,与 retrieve() 内部逻辑一致
# ---------------------------------------------------------------------------


def test_score_one_matches_retrieve_and_returns_none_for_future_record(tmp_path):
    store = _make_store(tmp_path)
    query = "stablecoin depeg risk rising on secondary markets"
    ts = 1_700_000_000_000

    record = store.write(query, ts=ts, layer="L1")

    direct_score = store.score_one(record, query, query_ts=ts)
    [(_r, retrieve_score)] = store.retrieve(query, query_ts=ts, top_k=5)
    assert direct_score == pytest.approx(retrieve_score, rel=1e-12)

    # 未来record: score_one 必须返回 None(不可见),而不是一个"很低但存在"的分数
    future_record = store.write(query, ts=ts + 1, layer="L1")
    assert store.score_one(future_record, query, query_ts=ts) is None


# ---------------------------------------------------------------------------
# 8. 默认 embedder(hashing_embedder)基本健全性 —— 不需要网络/ML依赖
# ---------------------------------------------------------------------------


def test_default_hashing_embedder_is_deterministic_and_offline(tmp_path):
    text = "ETH/BTC ratio breaking down, rotation into BTC dominance"
    v1 = hashing_embedder(text)
    v2 = hashing_embedder(text)
    assert v1 == v2  # 确定性:同输入同输出
    assert len(v1) > 0
    norm = math.sqrt(sum(x * x for x in v1))
    assert norm == pytest.approx(1.0, abs=1e-9)  # 已 L2 归一化

    store = MemoryStore(db_path=tmp_path / "hashing.db")  # 不注入 embedder,走默认值
    ts = 1_700_000_000_000
    written = store.write(text, ts=ts, layer="L1")
    results = store.retrieve(text, query_ts=ts, top_k=1)
    assert len(results) == 1
    assert results[0][0].id == written.id
    assert results[0][1] > 0.0  # 逐字匹配,cosine similarity 应该是正的(接近1)


# ---------------------------------------------------------------------------
# 9. 可选:sentence-transformers 真实语义 embedder 冒烟测试(不可用则跳过)
# ---------------------------------------------------------------------------


def test_sentence_transformer_embedder_smoke_if_available(tmp_path):
    try:
        from ASSET.memory.engine import build_sentence_transformer_embedder

        embedder = build_sentence_transformer_embedder()
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"sentence-transformers unavailable/uncached, skipping smoke test: {exc}")

    store = MemoryStore(db_path=tmp_path / "st.db", embedder=embedder)
    ts = 1_700_000_000_000
    store.write("Bitcoin funding rate spiked sharply positive overnight", ts=ts, layer="L1")
    store.write("A recipe for chocolate chip cookies", ts=ts, layer="L1")

    results = store.retrieve("BTC funding rate spike", query_ts=ts, top_k=2)
    assert len(results) == 2
    # 语义相关的那条(funding rate)应该排在不相关的(cookie食谱)前面
    assert "funding" in results[0][0].content.lower()


# ---------------------------------------------------------------------------
# 10. 静态防回归护栏:retrieve/write 所在源文件里不允许出现墙钟时间调用
# ---------------------------------------------------------------------------


def test_no_wallclock_calls_anywhere_in_engine_source():
    """回归防护:防止未来有人在 engine.py 里悄悄加一行 time.time()/datetime.now()
    之类的墙钟调用。用 ast 解析源码后检查真正的 Import/Call 节点,而不是朴素
    字符串扫描 —— 模块docstring和方法docstring里本来就会提到这些名字作为
    "禁止对象"的说明文字，字符串扫描分不清"提到"和"调用"，ast 可以。
    """
    import ast

    import ASSET.memory.engine as engine_module

    source = Path(engine_module.__file__).read_text(encoding="utf-8")
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
            # 匹配形如 time.time() / datetime.now() / datetime.utcnow() 的属性调用
            if isinstance(func, ast.Attribute) and func.attr in {"time", "now", "utcnow", "today"}:
                offenders.append(f"call .{func.attr}(...) (line {node.lineno})")

    assert offenders == [], f"forbidden wall-clock references found in engine.py: {offenders}"
