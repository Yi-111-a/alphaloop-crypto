"""
reflector.py —— 反思模块 Reflector(§3.3)。

placement 说明(为什么这个文件放在 LOCKED 而不是 spec §3 字面上说的"ASSET 区规格"):
  spec §3 的标题是"ASSET 区规格(策略 Agent 本体)"，Reflector 名义上属于这一节。
  但 M3 复审明确要求 Reflector 的输入面"结构性封死"——只有 decisions.jsonl 记录
  + 对应价格结果，没有新闻、没有 research_notes、没有记忆库自由检索。这个保证
  只有在 Reflector 本身**不能被运行时的策略agent修改**时才有意义:§0 铁律写明
  "LOCKED 区模块……实现完成后,策略agent的运行时权限对此目录只读"。如果把
  reflector.py 放进 ASSET(agent 可自由读写、可 git branch 进化，§3.5)，
  agent 理论上可以在某次"进化"里悄悄给 reflect() 加一个 news_context 参数，或者
  在 prompt 里夹带自我评分——静态 AST 测试能拦住*现在*的代码，但拦不住 agent
  下一次提交的新分支。放进 LOCKED，"输入面封死"这句话才是对运行时永真的断言，
  而不只是对"当前这一份代码"的断言。这与 ASSET/memory/engine.py 把"时间边界
  纪律"写成不可进化的硬约束是同一种理由，只是这次连文件本身的可写权限都收紧了。
  (对应 memory/engine.py 模块 docstring 里的同款"放置决策需要写文档"约定。)

M3 复审点名的两条不可退让的要求(逐字引用，见任务简报):
  1. "Reflector 的输入面必须结构性封死——函数签名里只有 decisions.jsonl 记录和
     对应价格结果,没有新闻、没有 research_notes、没有记忆库自由检索。" ->
     reflect() 的签名严格是 (self, now_ts, branch, window, price_lookup)，没有
     news/research_notes/自由文本 query 参数;本模块从不调用任何名为 retrieve
     的方法(只调用 memory_store.write，写入是允许的，检索不允许)。两条都各有
     一个 AST 回归测试守着(见 tests/test_reflector.py)。
  2. "falsifier 判定要代码化,不要全靠 LLM 自评……LLM 只负责在判定结果之上写
     经验摘要。" -> 应验/证伪/未决 的判定完全由 determine_thesis_status()(纯函数,
     不调用 llm_client)完成，用的是 LOCKED.schemas.parse_falsifier_condition /
     evaluate_falsifier_condition 这对确定性函数。self.llm_client 只在判定结果
     已经写进 prompt 文本之后才被调用一次，prompt 明确写"这些判定已经做出，不要
     重新评判对错，只写经验摘要"——见 _build_summary_prompt()，其输出文本里
     "证伪"/"应验"/"未决" 这些判定结果字面出现在 LLM 看到的 prompt 里,而不是
     由 LLM 自己产出。

时间边界纪律(与 ASSET/memory/engine.py、LOCKED/cold_start.py 同一纪律，同等
优先级):本模块任何函数都不调用墙钟时间(time.time()/datetime.now()/
datetime.utcnow()/等价物)。"现在"的唯一来源是调用方显式传入的 now_ts 参数。
reflect() 用 now_ts 来:(a) 过滤"还没发生"的决策记录(dec.ts > now_ts 的记录
不参与本轮判定)，(b) 作为 determine_thesis_status() 的"现在"基准，(c) 作为
写入 memory_store 的 L2/L3 记录的 ts。全文件没有 import time / import datetime。

价格采样密度(reflect() 内部选择，供 review):对每条待判定的决策，在
[decision.ts, min(now_ts, decision.ts + horizon_ms)] 这个区间内均匀取
_SAMPLE_COUNT(=6)个时间点(含两端)分别调用一次 price_lookup —— "a handful of
calls is fine"，6 个点在成本和"能不能抓到区间中段的插针"之间是一个粗但够用的
折中;调用方如果需要更细的采样密度，可以自行包一层 price_lookup 做插值，本模块
不替调用方决定"多细算够"。
"""
from __future__ import annotations

import re
from dataclasses import fields
from pathlib import Path
from typing import Any, Callable, Optional

from LOCKED import log_writer
from LOCKED.schemas import (
    Decision,
    ThesisMark,
    ThesisStatus,
    evaluate_falsifier_condition,
    parse_falsifier_condition,
)

# ---------------------------------------------------------------------------
# horizon 解析
# ---------------------------------------------------------------------------

_HORIZON_RE = re.compile(r"^(\d+)([hd])$")
_MS_PER_HOUR = 3_600_000
_MS_PER_DAY = 86_400_000


def parse_horizon_to_ms(horizon: str) -> int:
    """解析 "12h" / "3d" (spec 原文示例) 为毫秒。至少支持 "Nh" 与 "Nd"
    (N 为正整数)。任何不认识的格式一律 raise ValueError —— 不做静默猜测
    (比如不把 "12" 猜成 "12h"，不把 "1w" 猜成 7 天)。"""
    if not isinstance(horizon, str):
        raise ValueError(f"horizon must be a string, got {horizon!r}")
    m = _HORIZON_RE.match(horizon.strip())
    if not m:
        raise ValueError(
            f"unparseable horizon: {horizon!r}; expected format like '12h' or '3d' "
            "(positive integer followed by 'h' or 'd')"
        )
    n = int(m.group(1))
    if n <= 0:
        raise ValueError(f"horizon must have a positive integer count, got {horizon!r}")
    unit = m.group(2)
    return n * (_MS_PER_HOUR if unit == "h" else _MS_PER_DAY)


# ---------------------------------------------------------------------------
# 确定性判定 —— 纯函数，不调用 LLM，不调用墙钟
# ---------------------------------------------------------------------------


def determine_thesis_status(
    decision: Decision,
    price_samples: list[tuple[int, float]],
    now_ts: int,
) -> tuple[ThesisStatus, str]:
    """纯函数、确定性、不涉及 LLM。返回 (status, reason_string)。

    - falsifier_condition 不可解析/缺失 -> ("未决", "no machine-readable falsifier_condition")
    - horizon 不可解析 -> ("未决", "horizon unparseable: ...")(同样的"不做静默猜测"纪律)
    - 在 [decision.ts, min(now_ts, decision.ts + horizon_ms)] 区间内的任意一个
      样本触发了 evaluate_falsifier_condition -> ("证伪", 触发的那个样本的 ts/price)
    - 从未触发 且 now_ts >= decision.ts + horizon_ms(horizon 已完全走完)
      -> ("应验", ...)
    - 从未触发 且 horizon 还没走完 -> ("未决", "horizon not yet elapsed")
    """
    condition = parse_falsifier_condition(decision.falsifier_condition)
    if condition is None:
        return "未决", "no machine-readable falsifier_condition"

    try:
        horizon_ms = parse_horizon_to_ms(decision.horizon)
    except ValueError as exc:
        return "未决", f"horizon unparseable: {exc}"

    window_end = min(now_ts, decision.ts + horizon_ms)

    triggering: Optional[tuple[int, float]] = None
    for ts, price in sorted(price_samples, key=lambda sample: sample[0]):
        if ts < decision.ts or ts > window_end:
            continue
        if evaluate_falsifier_condition(condition, price):
            triggering = (ts, price)
            break

    if triggering is not None:
        trig_ts, trig_price = triggering
        return "证伪", (
            f"falsifier_condition {decision.falsifier_condition!r} triggered at "
            f"ts={trig_ts} price={trig_price}"
        )

    if now_ts >= decision.ts + horizon_ms:
        return "应验", (
            f"falsifier_condition {decision.falsifier_condition!r} never triggered "
            f"across full horizon [{decision.ts}, {decision.ts + horizon_ms}]"
        )

    return "未决", "horizon not yet elapsed"


# ---------------------------------------------------------------------------
# Reflector
# ---------------------------------------------------------------------------

_DECISION_FIELDS = {f.name for f in fields(Decision)}
_SAMPLE_COUNT = 6  # 见模块 docstring "价格采样密度" 一节
_MAX_SUMMARY_CHARS = 500


def _decision_from_dict(raw: dict) -> Decision:
    filtered = {k: v for k, v in raw.items() if k in _DECISION_FIELDS}
    return Decision(**filtered)


class Reflector:
    """反思模块(§3.3)。每日调用 2 次(§4.1: 08:00/20:00 UTC)，由调度器传入 now_ts。"""

    def __init__(
        self,
        llm_client: Callable[[str], str],
        memory_store: Any,
        log_root: Optional[str | Path] = None,
        decisions_log_path: str = "decisions.jsonl",
    ):
        self.llm_client = llm_client
        self.memory_store = memory_store
        self.log_root: Optional[Path] = Path(log_root) if log_root is not None else None
        self.decisions_log_path = decisions_log_path

    # ------------------------------------------------------------------
    # 价格采样(见模块 docstring "价格采样密度")
    # ------------------------------------------------------------------

    def _sample_prices(
        self,
        decision: Decision,
        now_ts: int,
        price_lookup: Callable[[str, int], float],
    ) -> list[tuple[int, float]]:
        try:
            horizon_ms = parse_horizon_to_ms(decision.horizon)
        except ValueError:
            # horizon 本身不可解析:determine_thesis_status 自己也会重新尝试解析
            # 并落到"未决"分支，这里只需要喂一个最小样本集，不影响最终判定。
            return [(decision.ts, price_lookup(decision.symbol, decision.ts))]

        window_end = min(now_ts, decision.ts + horizon_ms)
        if window_end <= decision.ts:
            ts_points = [decision.ts]
        else:
            span = window_end - decision.ts
            ts_points = sorted(
                {
                    decision.ts + round(span * i / (_SAMPLE_COUNT - 1))
                    for i in range(_SAMPLE_COUNT)
                }
            )

        return [(ts, price_lookup(decision.symbol, ts)) for ts in ts_points]

    # ------------------------------------------------------------------
    # LLM prompt 构造 —— 只在判定结果已确定之后调用，见模块 docstring 第2条
    # ------------------------------------------------------------------

    @staticmethod
    def _build_summary_prompt(
        verdicts: list[tuple[Decision, ThesisStatus, str]]
    ) -> str:
        lines = [
            "The following decisions have ALREADY been judged by deterministic code "
            "(LOCKED.schemas.evaluate_falsifier_condition against realized prices). "
            "Do NOT re-judge whether any of these were right or wrong -- that "
            "determination is final and not open for reconsideration. Your only job "
            f"is to write a concise (<= {_MAX_SUMMARY_CHARS} characters) experience "
            "summary of what can be learned from these already-decided outcomes.",
            "",
        ]
        for decision, status, reason in verdicts:
            lines.append(
                f"- decision_ts={decision.ts} symbol={decision.symbol} "
                f"action={decision.action} thesis={decision.thesis!r} "
                f"falsifier_condition={decision.falsifier_condition!r} "
                f"ALREADY JUDGED = {status} because: {reason}"
            )
        lines.append("")
        lines.append(
            f"Write the <= {_MAX_SUMMARY_CHARS} character experience summary now, "
            "and nothing else."
        )
        return "\n".join(lines)

    @staticmethod
    def _build_l3_lesson_content(decision: Decision, reason: str) -> str:
        """L3(永久层)教训记录:具体到 symbol / thesis / falsifier_condition / 是什么
        价格打破了它，供未来检索时能真正拿来用，而不是空洞的套话。"""
        return (
            f"[证伪教训] {decision.symbol}: 假设 {decision.thesis!r} "
            f"(证伪条件 falsifier_condition={decision.falsifier_condition!r}) 已被证伪 —— "
            f"{reason}。原始决策 ts={decision.ts}, horizon={decision.horizon}, "
            f"action={decision.action}。"
        )

    # ------------------------------------------------------------------
    # THE 结构性封死入口 —— 签名严格是 (now_ts, branch, window, price_lookup)
    # ------------------------------------------------------------------

    def reflect(
        self,
        now_ts: int,
        branch: str = "main",
        window: int = 20,
        price_lookup: Callable[[str, int], float] = None,
    ) -> list[ThesisMark]:
        """读取 decisions.jsonl 中 branch 匹配、action != "hold" 的最近 window 条
        决策(按 ts 排序取最后 window 条)，对每条用 determine_thesis_status() 做
        确定性判定，逐条写入 LOG/reflections/marks.jsonl，再调用一次 llm_client
        写入一条 <=500 字的经验摘要(L2)，并对每条被判"证伪"的决策额外写一条
        L3 教训记录。返回本轮产出的 ThesisMark 列表。
        """
        if price_lookup is None:
            raise ValueError("price_lookup is required")

        raw_records = log_writer.read_jsonl(self.decisions_log_path, root=self.log_root)

        candidates: list[dict] = []
        for rec in raw_records:
            if rec.get("action") == "hold":
                continue
            if rec.get("branch", "main") != branch:
                continue
            rec_ts = rec.get("ts")
            if rec_ts is None or rec_ts > now_ts:
                continue  # 不判定"还没发生"的决策，同一条 no-wall-clock 纪律
            candidates.append(rec)

        candidates.sort(key=lambda r: r["ts"])
        judged_raw = candidates[-window:] if window > 0 else []

        marks: list[ThesisMark] = []
        verdicts: list[tuple[Decision, ThesisStatus, str]] = []
        falsified: list[tuple[Decision, str]] = []

        for raw in judged_raw:
            decision = _decision_from_dict(raw)
            samples = self._sample_prices(decision, now_ts, price_lookup)
            status, reason = determine_thesis_status(decision, samples, now_ts)

            mark = ThesisMark(
                decision_ts=decision.ts,
                symbol=decision.symbol,
                thesis_status=status,
                note=reason,
            )
            marks.append(mark)
            log_writer.append_jsonl("reflections/marks.jsonl", mark, root=self.log_root)

            verdicts.append((decision, status, reason))
            if status == "证伪":
                falsified.append((decision, reason))

        if verdicts:
            prompt = self._build_summary_prompt(verdicts)
            summary = self.llm_client(prompt)
            if not isinstance(summary, str):
                summary = str(summary)
            summary = summary[:_MAX_SUMMARY_CHARS]
            # branch标签(2026-07-15,用户对多分支并行进化的隔离要求):反思
            # 摘要/教训是该分支自己的主观经验,写入时打上分支标签,检索侧
            # (ASSET/memory/engine.py)保证其他分支看不到——注意用duck-type
            # 兼容旧的memory_store实现(测试里的Fake可能没有branch参数)。
            try:
                self.memory_store.write(content=summary, ts=now_ts, layer="L2", importance=1.0, branch=branch)
            except TypeError:
                self.memory_store.write(content=summary, ts=now_ts, layer="L2", importance=1.0)

        for decision, reason in falsified:
            lesson = self._build_l3_lesson_content(decision, reason)
            try:
                self.memory_store.write(content=lesson, ts=now_ts, layer="L3", importance=1.0, branch=branch)
            except TypeError:
                self.memory_store.write(content=lesson, ts=now_ts, layer="L3", importance=1.0)

        return marks
