r"""
researcher.py —— 研究员 Researcher(§3.1,以及 §4.0 COLD_START 步骤3)。

ASSET 区代码(策略agent自由读写区)。与 trader.py 不同,Researcher 天生就是要
"把外部信息拉进来"的角色 —— §3.5 明确把 Researcher 和 Trader 列为仅有的两个
被允许随时间进化 ASSET/strategy/ 代码的角色,Researcher 没有 Reflector 那种
"决策函数参数签名里不能有新闻数据入口"式的隔离要求(那是 §3.3 对 Reflector 的
专门约束,Reflector 因此被放进结构隔离的 LOCKED,这里不适用)。

本模块有两个入口:

  1. daily_research(ts, date_str, queries=None)
     稳态每日调用一次(§3.1):检索(可选) -> llm_client 合成 -> 写
     research_notes/{date_str}.md,返回写入文件的 Path。

  2. run_cold_start_research(ts, universe_symbols, price_history, ...)
     §4.0 COLD_START 步骤3专用,只在冷启动期间跑一次:计算波动/趋势/相关性
     画像 + 资金费率分布 -> 检索(可选)+ llm_client 合成初始假设 -> 不足
     min_hypotheses 条时用模板补足 -> 写 genesis.md -> 把每条假设写入
     memory_store 的 L2/L3 层。

时间边界纪律(与 ASSET/memory/engine.py、LOCKED/cold_start.py 同一条纪律):
本文件任何执行路径都不调用墙钟时间(time.time()/datetime.now()/
datetime.utcnow()/等价物)。写入 memory_store 用的 ts、写进 genesis.md /
daily research 文件名用的 date_str,都是调用方显式传入的参数,不在本模块内
派生"今天是哪天"。测试 test_no_wallclock_calls_in_researcher_source 用 ast
解析源码做静态回归防护,方式与 test_memory_engine.py 里的同名测试一致。

依赖注入(与 trader.py 同一约定):
  - llm_client: Callable[[str], str] —— prompt 进,原始文本出(约定输出应为
    JSON,但由调用方的 fake 决定;生产环境接一个真实 Claude API 调用,可能
    用 web_search 工具调用能力)。
  - search_client: Callable[[str], list[dict]] | None —— 查询字符串进,
    形如 {"source","title","summary","url"} 的 dict 列表出。生产环境接真实
    的 arXiv / GitHub / web 搜索;None 时(允许的合法配置,不是异常路径),
    本模块退化成"纯 llm_client 合成"模式 —— 不调用任何搜索,直接把
    "无检索结果可用,请基于你对 FinMem/TradingAgents/TradingGroup 这类
    项目和文献的已有知识合成"写进 prompt。两条路径共用同一份解析/重试/
    兜底逻辑,daily_research 和 run_cold_start_research 都能在
    search_client=None 时正常产出合法文件,这是设计要求而非退化容忍。
  - memory_store: 与 trader.py 相同的 duck-type 约定,只要求有
    .write(content, ts, layer, importance) -> MemoryRecord(ASSET.memory.
    engine.MemoryStore 的真实签名,见该模块)。

genesis.md 的假设编号格式(供下游解析,精确约定 —— 写这里是因为将来某个
"数一数记忆库/genesis.md 里有几条假设"的模块很可能要解析这个文件,格式必须
稳定):每条假设是一个二级标题,形如

    ## H3: <一句话假设标题>

    <正文:rationale,以及若是补足用的模板假设,会有 "[GENERIC]" 前缀标注>

编号从 H1 开始连续递增,不跳号。用正则 `^## H(\d+):` 逐行扫描即可数出假设数、
提取编号与标题 —— 这正是 hypothesis_ids 返回值的来源(而不是重新解析文件)。
"""
from __future__ import annotations

import json
import re
import statistics
from pathlib import Path
from typing import Any, Callable, Optional

from LOCKED.schemas import MemoryLayer

# ---------------------------------------------------------------------------
# 默认路径(与 ASSET/memory/engine.py 用 __file__ 派生默认路径同一手法,不依赖
# 进程当前工作目录 —— 测试从任意 cwd 跑都应该拿到同一个"项目内默认路径"。)
# ---------------------------------------------------------------------------

_ASSET_DIR = Path(__file__).resolve().parent.parent
_DEFAULT_RESEARCH_NOTES_DIR = _ASSET_DIR / "research_notes"
_DEFAULT_GENESIS_PATH = _DEFAULT_RESEARCH_NOTES_DIR / "genesis.md"

_REQUIRED_FINDING_FIELDS = ("source", "core_idea", "testable_hypothesis", "suggested_experiment")

_DEFAULT_DAILY_QUERIES = [
    "arXiv q-fin trading agent LLM",
    "arXiv cs.AI autonomous trading agent",
    "github trending quantitative trading strategy",
    "crypto perpetual futures funding rate strategy research",
]

_COLD_START_SEARCH_QUERIES = [
    "FinMem LLM stock trading layered memory",
    "TradingAgents multi-agent LLM trading framework",
    "TradingGroup reflection trading pipeline",
    "crypto perpetual funding rate carry trade research",
]

# ---------------------------------------------------------------------------
# 通用兜底假设模板 —— 只在 run_cold_start_research 合成出的真实假设数不足
# min_hypotheses 时用来补足(§4.0 要求 assert 至少10条,不足10条冷启动永远
# 无法结束,比"补几条诚实标注为通用模板的假设"更糟)。内容取材于 FinMem /
# TradingAgents / TradingGroup 这类项目公开介绍里反复出现的几类经典想法
# (动量、均值回归、资金费率carry、波动率机制切换、情绪滞后、流动性风险、
# 相关性崩溃、仓位风控纪律、反思驱动的假设淘汰、多空对冲、链上数据背离、
# 新闻/情绪滞后反应),不是对任何单一论文/仓库代码的照抄 —— §3.1 允许的是
# "移植其模块/理念"级别的借鉴,不是抓取代码。
# ---------------------------------------------------------------------------

GENERIC_HYPOTHESIS_TEMPLATES: list[tuple[str, str]] = [
    (
        "Momentum persistence in trending majors",
        "FinMem/TradingAgents 类项目反复验证:主流币种(BTC/ETH)在确认趋势"
        "形成后,短期(数小时到数天)存在动量延续,可作为初始追踪信号来源。",
    ),
    (
        "Mean reversion after volatility spikes",
        "剧烈波动(如插针、清算瀑布)后价格短期内有向均值回归的倾向,"
        "TradingGroup 一系的反思管线常用此类假设作为反向仓位的初始候选。",
    ),
    (
        "Funding rate carry signal",
        "持续偏离0的资金费率反映多空拥挤程度,长期正费率环境下做空拥挤方向、"
        "吃资金费率是 perp 市场的经典carry策略雏形。",
    ),
    (
        "Volatility regime switching",
        "波动率本身存在低-高机制切换(regime switching),同一动量/均值回归"
        "信号在不同波动率机制下的胜率显著不同,值得分开建模。",
    ),
    (
        "Cross-asset correlation breakdown as risk signal",
        "山寨币与BTC相关性平时较高,极端行情下相关性崩溃(decoupling)"
        "往往伴随流动性风险,可作为降杠杆/减仓的早期预警信号。",
    ),
    (
        "Liquidity-driven slippage asymmetry",
        "低流动性币种在同等仓位下滑点显著更高,初始假设应偏向对24h成交额"
        "更低的标的采用更保守的目标名义仓位。",
    ),
    (
        "Reflection-driven thesis pruning",
        "TradingGroup 的核心理念:被反复证伪的thesis类型应该被显式记录进L3"
        "永久层并在后续决策中降权,而不是每个周期重新试错同一个错误假设。",
    ),
    (
        "Sentiment lag relative to price action",
        "FinMem 一系强调市场情绪/新闻对价格的反应存在滞后,价格领先、"
        "情绪滞后确认的窗口期可能是一个可被验证的交易机会。",
    ),
    (
        "Position sizing discipline under drawdown",
        "TradingAgents 强调风控优先于收益最大化:连续亏损后应主动降低目标"
        "名义仓位百分比,而不是等熔断触发才被动降杠杆。",
    ),
    (
        "Funding-rate-history distribution informs entry timing",
        "资金费率历史分布(均值/极值)可用于判断当前费率是否处于历史极端"
        "分位,极端分位本身可作为独立于价格趋势的择时信号。",
    ),
    (
        "Correlation clustering for diversification",
        "币池内标的可按收益相关性聚类,同一决策周期内避免在高度相关的"
        "多个标的上重复下注同一方向,变相突破组合层面的风险预算。",
    ),
    (
        "New-listing / thin-history caution",
        "上市时间较短或历史K线不足的标的,波动率画像统计意义有限,初始"
        "假设应对这类标的采用更低的默认杠杆与更短的持仓周期。",
    ),
]


def _hashing_content_key(text: str) -> str:
    """轻量去重键:小写+去空白,避免同义模板因大小写/空格差异被判定为不同。"""
    return re.sub(r"\s+", "", (text or "").lower())


# ---------------------------------------------------------------------------
# 通用:LLM JSON 输出的解析 + 重试(与 trader.py 的重试反馈风格一致,但这里
# 解析目标是"findings 列表"或"hypotheses 列表",不是 Decision)
# ---------------------------------------------------------------------------


def _call_llm_for_json_list(
    llm_client: Callable[[str], str],
    build_prompt: Callable[[Optional[str]], str],
    validate_item: Callable[[dict], list[str]],
    max_retries: int = 3,
) -> tuple[list[dict], Optional[str]]:
    """通用"prompt -> llm_client -> JSON列表校验"重试循环。

    返回 (items, error)。items 在全部重试失败时为空列表,error 携带最后一次
    失败原因(供调用方决定是否要走兜底/模板补足路径,而不是崩溃)。
    """
    retry_feedback: Optional[str] = None
    last_error: Optional[str] = None

    for _attempt in range(1, max_retries + 1):
        prompt = build_prompt(retry_feedback)
        raw = llm_client(prompt)
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError, ValueError):
            last_error = "response_not_valid_json"
            retry_feedback = last_error
            continue

        if not isinstance(parsed, list):
            last_error = "response_not_a_json_list"
            retry_feedback = last_error
            continue

        valid_items: list[dict] = []
        item_errors: list[str] = []
        for i, item in enumerate(parsed):
            if not isinstance(item, dict):
                item_errors.append(f"item_{i}_not_an_object")
                continue
            errs = validate_item(item)
            if errs:
                item_errors.append(f"item_{i}: {'; '.join(errs)}")
                continue
            valid_items.append(item)

        if valid_items:
            # 部分条目不合格也不整体作废 —— 研究产出不是交易决策,单条findings/
            # hypothesis 的格式瑕疵不应该让整批可用结果被丢弃(这点与 trader.py
            # "任何一条不合格就整批重试"的严格性刻意不同:Trader 的输出直接驱动
            # 真实下单校验链,必须全对;Researcher 的输出是研究笔记,宁可保留
            # 部分可用结果 + 事后补足,也不要因小瑕疵整批作废重来。)
            return valid_items, (None if not item_errors else "; ".join(item_errors))

        last_error = "; ".join(item_errors) if item_errors else "empty_list"
        retry_feedback = last_error

    return [], last_error


# ---------------------------------------------------------------------------
# daily_research 的 findings 校验
# ---------------------------------------------------------------------------


def _validate_finding(item: dict) -> list[str]:
    errors = []
    for field in _REQUIRED_FINDING_FIELDS:
        val = item.get(field)
        if not isinstance(val, str) or not val.strip():
            errors.append(f"{field}_missing_or_empty")
    return errors


_FALLBACK_FINDING = {
    "source": "internal-fallback (no LLM synthesis available after retries)",
    "core_idea": "检索/合成管线本次未能产出可解析结果,记录一个占位条目以保证"
    "research_notes文件始终存在且格式合法,不阻塞下游调度。",
    "testable_hypothesis": "占位假设:下个研究周期重新检索/合成应产出真实findings,"
    "若连续多个周期都退化到此fallback,应人工检查search_client/llm_client配置。",
    "suggested_experiment": "人工检查上一轮llm_client原始响应与search_client连通性,"
    "确认prompt格式与预期输出schema是否漂移。",
}


# ---------------------------------------------------------------------------
# genesis.md 的 hypothesis 校验
# ---------------------------------------------------------------------------


def _validate_hypothesis(item: dict) -> list[str]:
    errors = []
    hyp = item.get("hypothesis")
    if not isinstance(hyp, str) or not hyp.strip():
        errors.append("hypothesis_missing_or_empty")
    rationale = item.get("rationale")
    if rationale is not None and not isinstance(rationale, str):
        errors.append("rationale_must_be_str_if_present")
    permanent = item.get("permanent")
    if permanent is not None and not isinstance(permanent, bool):
        errors.append("permanent_must_be_bool_if_present")
    return errors


# ---------------------------------------------------------------------------
# 价格序列画像计算(§4.0:波动/趋势/相关性画像)
# ---------------------------------------------------------------------------


def _extract_close_series(series: Any) -> list[float]:
    """从调用方提供的 price_history[symbol] 里尽力提取一串收盘价。

    支持的形状(caller-supplied,与 data_pipeline 解耦,见类 docstring):
      - list[float | int]                          直接当作收盘价序列
      - list[dict]，每个dict含 "close" 或 "c" 键     取该键
      - dict 含 "closes" 键，值是上面两种之一          递归展开一层

    任何无法识别的形状返回空列表(画像计算对该symbol跳过,不抛异常 —— 冷启动
    调研不应该因为某一个symbol的数据格式意外而整体失败)。
    """
    if isinstance(series, dict) and "closes" in series:
        series = series["closes"]

    if not isinstance(series, (list, tuple)):
        return []

    out: list[float] = []
    for item in series:
        if isinstance(item, (int, float)) and not isinstance(item, bool):
            out.append(float(item))
        elif isinstance(item, dict):
            val = item.get("close", item.get("c"))
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                out.append(float(val))
    return out


def _pct_returns(closes: list[float]) -> list[float]:
    returns = []
    for i in range(1, len(closes)):
        prev = closes[i - 1]
        if prev == 0:
            continue
        returns.append((closes[i] - prev) / prev)
    return returns


def _pearson(a: list[float], b: list[float]) -> Optional[float]:
    n = min(len(a), len(b))
    if n < 2:
        return None
    a, b = a[:n], b[:n]
    mean_a, mean_b = sum(a) / n, sum(b) / n
    cov = sum((a[i] - mean_a) * (b[i] - mean_b) for i in range(n))
    var_a = sum((x - mean_a) ** 2 for x in a)
    var_b = sum((x - mean_b) ** 2 for x in b)
    denom = (var_a * var_b) ** 0.5
    if denom == 0:
        return None
    return cov / denom


def _compute_symbol_profile(symbol: str, closes: list[float]) -> dict:
    """单个symbol的波动/趋势画像。选定的统计口径(工程判断,非spec强制):
      - volatility_pct: 收益率序列的总体标准差(pstdev) * 100,单位:百分比
      - trend_pct: (末值-首值)/首值 * 100,整段窗口的总回报率
      - n_points: 参与计算的价格点数,数据不足时下游据此显示"insufficient_data"
    """
    returns = _pct_returns(closes)
    profile: dict[str, Any] = {"symbol": symbol, "n_points": len(closes)}
    if len(closes) < 2:
        profile["volatility_pct"] = None
        profile["trend_pct"] = None
        return profile
    profile["trend_pct"] = (closes[-1] - closes[0]) / closes[0] * 100 if closes[0] != 0 else None
    profile["volatility_pct"] = statistics.pstdev(returns) * 100 if len(returns) >= 1 else None
    profile["_returns"] = returns
    return profile


def _compute_funding_summary(symbol: str, raw: Any) -> Optional[dict]:
    values: list[float] = []
    if isinstance(raw, dict) and "rates" in raw:
        raw = raw["rates"]
    if isinstance(raw, (list, tuple)):
        for item in raw:
            if isinstance(item, (int, float)) and not isinstance(item, bool):
                values.append(float(item))
            elif isinstance(item, dict):
                val = item.get("funding_rate", item.get("rate"))
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    values.append(float(val))
    if not values:
        return None
    return {
        "symbol": symbol,
        "n": len(values),
        "mean": statistics.mean(values),
        "min": min(values),
        "max": max(values),
        "stdev": statistics.pstdev(values) if len(values) >= 2 else 0.0,
    }


# ---------------------------------------------------------------------------
# Researcher
# ---------------------------------------------------------------------------


class Researcher:
    """研究员 Researcher(§3.1,§4.0步骤3)。"""

    def __init__(
        self,
        llm_client: Callable[[str], str],
        memory_store: Any,
        search_client: Optional[Callable[[str], list[dict]]] = None,
        research_notes_dir: "str | Path | None" = None,
        genesis_path: "str | Path | None" = None,
        max_retries: int = 3,
    ):
        self.llm_client = llm_client
        self.memory_store = memory_store
        self.search_client = search_client
        self.research_notes_dir = (
            Path(research_notes_dir) if research_notes_dir is not None else _DEFAULT_RESEARCH_NOTES_DIR
        )
        self.genesis_path = (
            Path(genesis_path) if genesis_path is not None else (self.research_notes_dir / "genesis.md")
        )
        self.max_retries = max_retries

    # ------------------------------------------------------------------
    # search_client 调用封装(容错:单个query失败不拖垮整批)
    # ------------------------------------------------------------------

    def _run_searches(self, queries: list[str]) -> list[dict]:
        if self.search_client is None:
            return []
        results: list[dict] = []
        for q in queries:
            try:
                r = self.search_client(q)
            except Exception:
                # 搜索源抖动不应该让整个研究周期失败 —— 退化为"该query无结果",
                # 后续 llm_client 合成 prompt 里会如实反映"部分/无检索结果"。
                continue
            if isinstance(r, list):
                results.extend(r)
        return results

    # ------------------------------------------------------------------
    # daily_research (§3.1)
    # ------------------------------------------------------------------

    def _build_daily_prompt(
        self, date_str: str, search_results: list[dict], retry_feedback: Optional[str]
    ) -> str:
        lines = [
            "You are the Researcher agent for AlphaLoop-Crypto (§3.1), running daily "
            f"research for {date_str}. Search arXiv (q-fin, cs.AI trading-agent topics) and "
            "GitHub trending quant projects conceptually inform your synthesis; draw on "
            "FinMem (pipiku915/FinMem-LLM-StockTrading), TradingAgents, and TradingGroup style "
            "ideas for what KINDS of hypotheses a crypto trading research agent should produce "
            "(do not fabricate specific citations you cannot support).",
            "",
            "Respond with a JSON list ONLY, each item shaped exactly as: "
            '{"source": str, "core_idea": str, "testable_hypothesis": str, '
            '"suggested_experiment": str}. All four fields are required non-empty strings.',
            "",
        ]
        if search_results:
            lines.append(f"Search results available ({len(search_results)} items):")
            lines.append(json.dumps(search_results, ensure_ascii=False))
        else:
            lines.append(
                "No search results available this run (search_client not configured, or "
                "returned nothing). Synthesize findings from your existing knowledge of "
                "crypto-trading-agent research directions (FinMem/TradingAgents/TradingGroup-"
                "style ideas: layered memory, reflection-driven pruning, momentum/mean-"
                "reversion, funding-rate carry, sentiment lag, etc.)."
            )
        if retry_feedback:
            lines.append("")
            lines.append(
                f"Your previous response failed validation with: {retry_feedback}. "
                "Fix the issues and resubmit strictly as a JSON list."
            )
        return "\n".join(lines)

    def _findings_to_markdown(self, date_str: str, findings: list[dict]) -> str:
        parts = [f"# Research Notes — {date_str}", ""]
        for i, f in enumerate(findings, start=1):
            parts.append(f"## Finding {i}")
            parts.append(f"- **Source**: {f['source']}")
            parts.append(f"- **Core idea**: {f['core_idea']}")
            parts.append(f"- **Testable hypothesis**: {f['testable_hypothesis']}")
            parts.append(f"- **Suggested experiment**: {f['suggested_experiment']}")
            parts.append("")
        return "\n".join(parts)

    def daily_research(
        self,
        ts: int,
        date_str: str,
        queries: Optional[list[str]] = None,
    ) -> Path:
        """稳态每日调研(§3.1)。ts 只用于潜在的记忆写入时间戳(当前实现不在
        daily_research 里写memory——spec §3.1原文只规定输出research_notes文件；
        往memory写只在§4.0冷启动步骤明确要求，见run_cold_start_research)，
        date_str 由调用方显式传入,本方法内部不派生"今天"是哪天。
        """
        queries = queries if queries is not None else list(_DEFAULT_DAILY_QUERIES)
        search_results = self._run_searches(queries)

        def build_prompt(retry_feedback: Optional[str]) -> str:
            return self._build_daily_prompt(date_str, search_results, retry_feedback)

        findings, _error = _call_llm_for_json_list(
            self.llm_client, build_prompt, _validate_finding, max_retries=self.max_retries
        )
        if not findings:
            findings = [dict(_FALLBACK_FINDING)]

        content = self._findings_to_markdown(date_str, findings)

        self.research_notes_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.research_notes_dir / f"{date_str}.md"
        out_path.write_text(content, encoding="utf-8")
        return out_path

    # ------------------------------------------------------------------
    # run_cold_start_research (§4.0 步骤3)
    # ------------------------------------------------------------------

    def _build_cold_start_prompt(
        self,
        universe_symbols: list[str],
        profiles: list[dict],
        funding_summaries: list[dict],
        search_results: list[dict],
        retry_feedback: Optional[str],
    ) -> str:
        lines = [
            "You are the Researcher agent for AlphaLoop-Crypto performing the ONE-TIME "
            "COLD_START synthesis (§4.0 step 3). Universe symbols: "
            f"{universe_symbols}.",
            "",
            "Computed volatility/trend profiles (for context, do not restate verbatim, use "
            "them to ground your hypotheses in this specific universe):",
            json.dumps(profiles, ensure_ascii=False),
            "Funding rate distribution summaries:",
            json.dumps(funding_summaries, ensure_ascii=False),
            "",
            "Synthesize initial TRADING HYPOTHESES inspired by FinMem "
            "(pipiku915/FinMem-LLM-StockTrading), TradingAgents, and TradingGroup style "
            "research (layered memory, reflection-driven pruning, momentum/mean-reversion, "
            "funding-rate carry, sentiment lag, volatility regimes, correlation structure, "
            "position-sizing discipline, etc.) applied to THIS crypto universe.",
            "",
            'Respond with a JSON list ONLY, each item shaped as: {"hypothesis": str, '
            '"rationale": str, "permanent": bool (optional, default false)}. Set '
            '"permanent": true only for a structural/general risk-management lesson meant '
            "to persist indefinitely rather than a market-specific testable bet.",
        ]
        if search_results:
            lines.append("")
            lines.append(f"Search results available ({len(search_results)} items):")
            lines.append(json.dumps(search_results, ensure_ascii=False))
        if retry_feedback:
            lines.append("")
            lines.append(
                f"Your previous response failed validation with: {retry_feedback}. "
                "Fix the issues and resubmit strictly as a JSON list."
            )
        return "\n".join(lines)

    def _pad_with_generic_hypotheses(
        self, hypotheses: list[dict], min_hypotheses: int
    ) -> list[dict]:
        """不足 min_hypotheses 条时,用 GENERIC_HYPOTHESIS_TEMPLATES 循环补足,
        并显式标注 generic=True(genesis.md 正文里会体现为 "[GENERIC]" 前缀)。
        循环取模而不是"模板用完就报错" —— min_hypotheses 是调用方可配置的参数,
        不能假设它 <= 模板数量;循环补足时对第二轮起的重复模板加 variant 后缀
        避免文本完全重复。
        """
        out = list(hypotheses)
        existing_keys = {_hashing_content_key(h["hypothesis"]) for h in out}
        i = 0
        variant_round = 1
        while len(out) < min_hypotheses:
            title, rationale = GENERIC_HYPOTHESIS_TEMPLATES[i % len(GENERIC_HYPOTHESIS_TEMPLATES)]
            if i >= len(GENERIC_HYPOTHESIS_TEMPLATES):
                variant_round = i // len(GENERIC_HYPOTHESIS_TEMPLATES) + 1
                title = f"{title} (variant {variant_round})"
            key = _hashing_content_key(title)
            i += 1
            if key in existing_keys:
                continue
            existing_keys.add(key)
            out.append(
                {
                    "hypothesis": title,
                    "rationale": rationale,
                    "permanent": False,
                    "generic": True,
                }
            )
        return out

    def _profiles_to_markdown(
        self, profiles: list[dict], funding_summaries: list[dict], correlation: dict
    ) -> str:
        parts = ["## Volatility / Trend Profiles", ""]
        for p in profiles:
            parts.append(f"### {p['symbol']}")
            if p["n_points"] < 2:
                parts.append(f"- insufficient_data (n_points={p['n_points']})")
            else:
                parts.append(f"- n_points: {p['n_points']}")
                vol = p["volatility_pct"]
                trend = p["trend_pct"]
                parts.append(
                    f"- volatility_pct (stdev of returns): "
                    f"{vol:.4f}%" if vol is not None else "- volatility_pct: n/a"
                )
                parts.append(
                    f"- trend_pct (total return over window): "
                    f"{trend:.4f}%" if trend is not None else "- trend_pct: n/a"
                )
            parts.append("")

        parts.append("## Correlation Matrix (returns, Pearson)")
        parts.append("")
        symbols = [p["symbol"] for p in profiles]
        if len(symbols) >= 2:
            header = "| symbol | " + " | ".join(symbols) + " |"
            sep = "|---" * (len(symbols) + 1) + "|"
            parts.append(header)
            parts.append(sep)
            for s1 in symbols:
                row = [s1]
                for s2 in symbols:
                    corr = correlation.get((s1, s2))
                    row.append(f"{corr:.4f}" if corr is not None else "n/a")
                parts.append("| " + " | ".join(row) + " |")
        else:
            parts.append("insufficient symbols for a correlation matrix (need >= 2)")
        parts.append("")

        parts.append("## Funding Rate Distribution")
        parts.append("")
        if funding_summaries:
            for fs in funding_summaries:
                parts.append(f"### {fs['symbol']}")
                parts.append(f"- n: {fs['n']}")
                parts.append(f"- mean: {fs['mean']:.6f}")
                parts.append(f"- min: {fs['min']:.6f}")
                parts.append(f"- max: {fs['max']:.6f}")
                parts.append(f"- stdev: {fs['stdev']:.6f}")
                parts.append("")
        else:
            parts.append("no funding_rate_history provided for this cold start run")
            parts.append("")
        return "\n".join(parts)

    def _hypotheses_to_markdown(self, hypotheses: list[dict]) -> str:
        parts = ["## Initial Hypotheses", ""]
        for i, h in enumerate(hypotheses, start=1):
            tag = "[GENERIC] " if h.get("generic") else ""
            parts.append(f"## H{i}: {tag}{h['hypothesis']}")
            parts.append("")
            rationale = h.get("rationale") or ""
            if rationale:
                parts.append(rationale)
                parts.append("")
            parts.append(f"- layer: {'L3' if h.get('permanent') else 'L2'}")
            parts.append("")
        return "\n".join(parts)

    def run_cold_start_research(
        self,
        ts: int,
        universe_symbols: list[str],
        price_history: dict[str, Any],
        funding_rate_history: Optional[dict[str, Any]] = None,
        min_hypotheses: int = 10,
    ) -> dict:
        """§4.0 COLD_START 步骤3。产出 genesis.md + 把每条假设写入 memory_store。

        L2 vs L3 归属规则(工程判断,spec未强制指定具体分割方式,这里显式记录
        选择的理由):genesis阶段产出的假设默认全部是 L2(30天时间常数) ——
        因为它们此刻全部是"未经验证的初始猜想",§3.4 对 L3 的定义是"被证伪的
        教训、经过多次验证的规律",这两者都需要走过至少一轮 Reflector 的证伪/
        验证周期才成立,genesis阶段的假设天然不满足。唯一例外:如果 llm_client
        合成结果里某条假设被显式标记 "permanent": true(代表这是一条更接近
        "通用风控原则/结构性事实"而非"针对当前市场的可证伪交易猜想",例如
        "亏损后应主动降杠杆"这类原则本身不太可能被单次证伪/证实),则允许直接
        进 L3。补足用的通用模板假设(generic=True)一律强制 L2,不允许通过模板
        路径绕过"L3应该是被验证过的"这条原则。
        """
        # ---- 1. 画像计算 ----
        profiles = []
        returns_by_symbol: dict[str, list[float]] = {}
        for symbol in universe_symbols:
            closes = _extract_close_series(price_history.get(symbol))
            profile = _compute_symbol_profile(symbol, closes)
            returns_by_symbol[symbol] = profile.pop("_returns", [])
            profiles.append(profile)

        correlation: dict[tuple[str, str], Optional[float]] = {}
        for s1 in universe_symbols:
            for s2 in universe_symbols:
                correlation[(s1, s2)] = _pearson(returns_by_symbol.get(s1, []), returns_by_symbol.get(s2, []))

        funding_summaries = []
        if funding_rate_history:
            for symbol in universe_symbols:
                if symbol in funding_rate_history:
                    fs = _compute_funding_summary(symbol, funding_rate_history[symbol])
                    if fs is not None:
                        funding_summaries.append(fs)

        # ---- 2. 检索 + LLM 合成初始假设 ----
        search_results = self._run_searches(list(_COLD_START_SEARCH_QUERIES))

        def build_prompt(retry_feedback: Optional[str]) -> str:
            return self._build_cold_start_prompt(
                universe_symbols, profiles, funding_summaries, search_results, retry_feedback
            )

        raw_hypotheses, _error = _call_llm_for_json_list(
            self.llm_client, build_prompt, _validate_hypothesis, max_retries=self.max_retries
        )

        # ---- 3. 不足 min_hypotheses 条时用模板补足(见方法/模块docstring) ----
        hypotheses = self._pad_with_generic_hypotheses(raw_hypotheses, min_hypotheses)

        # ---- 4. 写 genesis.md ----
        md_parts = [
            "# Genesis Research Notes (COLD_START)",
            "",
            f"Universe: {universe_symbols}",
            "",
            self._profiles_to_markdown(profiles, funding_summaries, correlation),
            self._hypotheses_to_markdown(hypotheses),
        ]
        content = "\n".join(md_parts)

        self.genesis_path.parent.mkdir(parents=True, exist_ok=True)
        self.genesis_path.write_text(content, encoding="utf-8")

        # ---- 5. 每条假设写入 memory_store（L2/L3，见上方规则）----
        hypothesis_ids = []
        for i, h in enumerate(hypotheses, start=1):
            hid = f"H{i}"
            hypothesis_ids.append(hid)
            layer: MemoryLayer = "L3" if (h.get("permanent") and not h.get("generic")) else "L2"
            mem_content = f"{hid}: {h['hypothesis']}"
            if h.get("rationale"):
                mem_content += f"\n{h['rationale']}"
            self.memory_store.write(content=mem_content, ts=ts, layer=layer, importance=1.0)

        return {
            "genesis_path": self.genesis_path,
            "hypothesis_count": len(hypotheses),
            "hypothesis_ids": hypothesis_ids,
            "profiles": profiles,
            "funding_summaries": funding_summaries,
        }
