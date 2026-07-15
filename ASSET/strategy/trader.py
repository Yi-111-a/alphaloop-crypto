"""
trader.py —— 交易员 Trader(§3.2)。

ASSET 区代码(策略agent自由读写区,可被agent自我进化,见§3.5)。输入拼装顺序与
时间边界规则是 spec 明确规定的硬约束,不是风格选择,在这里严格照做:

  1. 当前持仓
  2. 记忆检索 top-k(通过 MemoryStore.retrieve,query_ts 必须是"决策自身的 ts"，
     绝不是墙钟时间 time.time()/datetime.now() —— 这是防止"未来记忆污染当下
     决策"这类信息泄漏的关键接线点。M1 的失败模式是算术bug，M2 的失败模式是
     信息泄漏；Trader 是实际调用记忆检索的那个函数，如果这里传错时间戳，
     记忆系统自己再怎么做好时间边界防护也会被静默绕过)
  3. 最新行情快照
  4. 最近一次反思摘要
  5. program.md 中的战术指令

输出 List[Decision]，严格按 LOCKED.schemas.Decision 的结构 + JSON schema 校验
(校验规则镜像 LOCKED/simulator.py execute() 里实际会拒绝决策的规则，保证一条
通过本模块校验的 Decision 也一定能通过 Simulator 自己的校验，而不只是"看起来
合理")。校验失败重试，最多 max_retries 次尝试；全部失败则默认输出一条安全的
action="hold" 决策。

M3 裁决新增:非 hold 决策必须额外产出 falsifier_condition —— 一个机器可读的
价格条件子句(如 "price<48000"，见 LOCKED.schemas.parse_falsifier_condition/
evaluate_falsifier_condition)，与自然语言的 falsifier 字段并存。这是为了让
下游 Reflector 能用确定性代码判定"这条决策是否被证伪"，而不是让 LLM 自己
评判自己的输出是否失败——后者是已知的自我评估偏差(LLM 倾向于把自己的失败
判成"未决")。hold 决策没有实际仓位、没有可证伪的对象，豁免这项要求。

对记忆系统接口的防御性处理:MemoryStore 是与本模块并行开发的模块，写这个文件时
其真实实现可能还不存在。按约定的接口形状 duck-type 处理 retrieve() 的返回值：
优先尝试 `.content` 属性；如果返回的是形如 (record, score) 的元组/列表，退化为
取 `record[0].content`；再不行退化为 dict 的 "content" 键，最后兜底为 str()。
等真实接口确认后，这里的防御性解包可以收紧。
"""
from __future__ import annotations

import json
import re
from dataclasses import fields
from typing import Any, Callable, Optional

from LOCKED.schemas import Decision, parse_falsifier_condition
from LOCKED.simulator import MIN_THESIS_LEN

# ---------------------------------------------------------------------------
# 校验规则常量 —— 镜像 LOCKED/simulator.py execute() 校验链中与"决策本身是否
# 合法"相关的规则(§2.2 步骤 3 / 5)。symbol∈universe(步骤4)、总敞口/单币敞口/
# 可用保证金(步骤6-8)等依赖账本状态的规则不在本模块职责内 —— 那些只有
# Simulator 在撮合时才能校验，Trader 产出 Decision 时还不知道成交后的账本状态。
# ---------------------------------------------------------------------------

VALID_ACTIONS = {"open_long", "open_short", "close", "adjust", "hold"}
MIN_OPEN_NOTIONAL_PCT = 5.0  # 开仓/调仓的最小名义仓位(净值百分点)。2026-07-15
                              # 用户两次真实反馈后定为5%:先是LLM给出0.02%粉尘仓,
                              # 后是全员仓位小到"很久都挣不了1U"。hold/close不受限。
MIN_LEVERAGE = 1
MAX_LEVERAGE = 10  # 模块级兜底默认值，仅在 Trader 未显式传入 max_leverage 时
                    # (如测试里的裸构造)生效。生产路径下 Trader(max_leverage=...)
                    # 由调用方注入 config.yaml 的 leverage.max，实际校验以实例的
                    # self.max_leverage 为准，而不是这个常量。

_DECISION_FIELDS = {f.name for f in fields(Decision)}

_HOLD_FALLBACK_THESIS = "连续3次LLM输出未通过schema校验,按安全默认值执行hold"
_HOLD_FALLBACK_FALSIFIER = "无有效决策依据,不设证伪条件,等待下个决策周期重新评估"
_HOLD_FALLBACK_HORIZON = "4h"
_HOLD_FALLBACK_SYMBOL = "BTC/USDT:USDT"

_HYPOTHESIS_RE = re.compile(r"H\d+")


def references_hypothesis(thesis: str) -> bool:
    """True if thesis contains a hypothesis-number reference like H1, H12, etc.

    这不是 decide() 里的硬校验门槛 —— spec 把"thesis 引用假设编号"框定为
    NORMAL 首日"好决策"的一个属性，由 prompt 设计驱动，而不是 schema 校验器
    该硬性拒绝的东西(一条缺失必填字段的决策，和一条没有引用 H 编号但字段齐全
    的决策，性质不同，不应被同等对待)。因此本函数是独立的 helper，供调用方
    自行 assert / 记录日志 / 做成软告警，不接入 decide() 的重试触发校验。
    """
    if not thesis or not isinstance(thesis, str):
        return False
    return bool(_HYPOTHESIS_RE.search(thesis))


def _validate_decision_dict(
    d: dict, max_leverage: int = MAX_LEVERAGE, valid_symbols: Optional[set] = None
) -> list[str]:
    """校验单条候选决策字典，返回错误原因列表(空列表 = 通过)。

    2026-07-14 新增两条语义级校验(不只是"字段类型对不对",而是"这个值在
    这个业务场景下讲不讲得通")——都是本session里签入agent反复手动拦下过的
    真实错误类别,不是假设性边界情况:
      - symbol 必须是本次决策时刻真实行情快照里存在的完整交易对(如
        "BTC/USDT:USDT"),不能是"BTC"这种截断写法。子代理这一类错误出现过
        不止一次,此前全靠人工审查发现。
      - target_notional_pct 不能是负数——LOCKED.simulator.py 内部会对它取
        abs(),所以负数不会导致仓位方向错误,但出现过at all说明子代理输出
        本身有问题(比如把"减仓"错误地编码成负的目标仓位,而不是用
        action="adjust"配合更小的正数),负数在这个字段里永远没有合理语义,
        应该在这里就拒绝重试,而不是靠abs()悄悄"纠正"成看似正常的结果。
    """
    errors: list[str] = []

    required = [
        "ts",
        "symbol",
        "action",
        "target_notional_pct",
        "leverage",
        "thesis",
        "falsifier",
        "horizon",
    ]
    missing = [k for k in required if k not in d]
    if missing:
        errors.append(f"missing_fields:{','.join(missing)}")
        return errors  # 缺字段时后续类型检查没有意义，直接返回

    action = d.get("action")
    if action not in VALID_ACTIONS:
        errors.append(f"invalid_action:{action!r}")

    thesis = d.get("thesis")
    if not isinstance(thesis, str) or len(thesis.strip()) < MIN_THESIS_LEN:
        errors.append(
            f"thesis_invalid: must be str with len>={MIN_THESIS_LEN} after strip"
        )

    falsifier = d.get("falsifier")
    if not isinstance(falsifier, str) or len(falsifier.strip()) < MIN_THESIS_LEN:
        errors.append(
            f"falsifier_invalid: must be str with len>={MIN_THESIS_LEN} after strip"
        )

    # M3 裁决:falsifier 的证伪判定不能全靠 LLM 自评(已知的自我评估偏差——
    # LLM 倾向于把自己的失败判成"未决")。Reflector 需要一个确定性代码可解析
    # 的价格条件来做判定，所以这里在自然语言 falsifier 之外，额外强制要求一个
    # 机器可读子句 falsifier_condition(格式 "price<48000" 之类，见
    # LOCKED.schemas.parse_falsifier_condition)。hold 决策没有实际仓位，没有
    # 可证伪的对象，因此豁免这项要求。
    action_for_falsifier_check = d.get("action")
    if action_for_falsifier_check != "hold":
        falsifier_condition = d.get("falsifier_condition")
        if parse_falsifier_condition(falsifier_condition) is None:
            errors.append(
                "falsifier_condition_invalid: non-hold decisions must include a "
                "machine-readable falsifier_condition matching 'price<NUMBER' / "
                "'price<=NUMBER' / 'price>NUMBER' / 'price>=NUMBER', got "
                f"{falsifier_condition!r}"
            )

    leverage = d.get("leverage")
    if (
        not isinstance(leverage, int)
        or isinstance(leverage, bool)
        or not (MIN_LEVERAGE <= leverage <= max_leverage)
    ):
        errors.append(
            f"leverage_invalid: must be int in [{MIN_LEVERAGE},{max_leverage}], got {leverage!r}"
        )

    target_notional_pct = d.get("target_notional_pct")
    if (
        not isinstance(target_notional_pct, (int, float))
        or isinstance(target_notional_pct, bool)
    ):
        errors.append("target_notional_pct_invalid: must be int or float")
    elif not (0 <= target_notional_pct <= 100):
        errors.append(
            f"target_notional_pct_invalid: must be in [0,100] (percent of NAV, "
            f"never negative -- direction comes from action, not sign), got {target_notional_pct!r}"
        )
    elif action in ("open_long", "open_short", "adjust") and target_notional_pct < MIN_OPEN_NOTIONAL_PCT:
        # 2026-07-15新增语义校验:真实观察到LLM给出0.02%/0.1%这种"连手续费都
        # 跑不赢"的仓位——要么是把百分点误当成了0-1小数比例(0.1其实想表达
        # 10%),要么是无意义的过度胆小。两种情况都应该拒绝重试,并在反馈里
        # 把单位讲清楚,而不是放行一笔毫无意义的决策。同日用户观察到全员
        # 仓位过小("别很久都挣不了1U"),底线从1%提到5%——100U本金下5%
        # 仍然只是5U名义,这是"有意义仓位"的地板,不是天花板。
        errors.append(
            f"target_notional_pct_invalid: {target_notional_pct!r} is below the "
            f"{MIN_OPEN_NOTIONAL_PCT} minimum for open/adjust decisions. NOTE the unit is "
            "PERCENTAGE POINTS of NAV: 15.0 means 15% of account value, 0.1 means one-tenth "
            "of one percent (almost nothing). If you intended a fraction (e.g. 0.15 = 15%), "
            "multiply by 100."
        )

    symbol = d.get("symbol")
    if not isinstance(symbol, str) or not symbol.strip():
        errors.append("symbol_invalid: must be non-empty str")
    elif valid_symbols is not None and symbol not in valid_symbols:
        errors.append(
            f"symbol_invalid: {symbol!r} is not in this cycle's tradeable universe "
            f"(must be the FULL pair form e.g. 'BTC/USDT:USDT', not a shortened ticker "
            f"like 'BTC'); valid symbols: {sorted(valid_symbols)}"
        )

    horizon = d.get("horizon")
    if not isinstance(horizon, str) or not horizon.strip():
        errors.append("horizon_invalid: must be non-empty str")

    ts = d.get("ts")
    if not isinstance(ts, int) or isinstance(ts, bool):
        errors.append("ts_invalid: must be int")

    return errors


class Trader:
    """交易员 Trader(§3.2)。每 4 小时调用一次 decide()，产出该周期的 List[Decision]。"""

    def __init__(
        self,
        llm_client: Callable[[str], str],
        memory_store: Any,
        max_retries: int = 3,
        max_leverage: int = MAX_LEVERAGE,
    ):
        self.llm_client = llm_client
        self.memory_store = memory_store
        self.max_retries = max_retries
        self.max_leverage = max_leverage

    # ------------------------------------------------------------------
    # 记忆检索结果的防御性解包(duck-typing，见模块顶部docstring)
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_content(item: Any) -> str:
        content = getattr(item, "content", None)
        if content is not None:
            return content
        if isinstance(item, dict) and "content" in item:
            return item["content"]
        if isinstance(item, (tuple, list)) and len(item) > 0:
            first = item[0]
            content = getattr(first, "content", None)
            if content is not None:
                return content
            if isinstance(first, dict) and "content" in first:
                return first["content"]
        return str(item)

    def _normalize_memory_results(self, raw: Any) -> list[str]:
        if not raw:
            return []
        return [self._extract_content(item) for item in raw]

    # ------------------------------------------------------------------
    # 输入拼装(§3.2 顺序 1-5)
    # ------------------------------------------------------------------

    def build_context(
        self,
        positions: list | dict,
        ts: int,
        latest_snapshot: dict,
        last_reflection_summary: Optional[str],
        program_tactics: Optional[str],
        memory_query_text: str,
        top_k: int = 5,
        branch: str = "main",
    ) -> dict:
        """按 spec §3.2 规定的顺序拼装 Trader 的输入上下文。

        关键点:memory_store.retrieve() 的 query_ts 参数被显式传成本次决策自己的
        ts(方法入参),这个方法及其调用的一切都不会碰 time.time()/datetime.now()。
        这是本模块防止"未来记忆泄漏进当下决策"的唯一防线所在 —— 记忆系统自身的
        时间边界闸门做得再好，只要这里传错一次时间戳，那个闸门就被绕过了。
        """
        # branch参数(2026-07-15分支记忆隔离):声明自己是哪个分支,检索时
        # 只拿共享记忆+本分支私有经验。旧的memory_store实现/测试Fake可能
        # 没有branch参数,duck-type降级兼容。
        try:
            memory_results_raw = self.memory_store.retrieve(
                memory_query_text, query_ts=ts, top_k=top_k, branch=branch
            )
        except TypeError:
            memory_results_raw = self.memory_store.retrieve(
                memory_query_text, query_ts=ts, top_k=top_k
            )
        memory_results = self._normalize_memory_results(memory_results_raw)

        # 顺序显式声明为 list[(name, value)]，既是拼装 prompt 的数据源，也让
        # "顺序是否正确"这件事在测试里是可断言的，而不是依赖隐式的 dict 插入顺序。
        ordered_parts: list[tuple[str, Any]] = [
            ("positions", positions),
            ("memory_results", memory_results),
            ("latest_snapshot", latest_snapshot),
            ("last_reflection_summary", last_reflection_summary),
            ("program_tactics", program_tactics),
        ]

        context: dict[str, Any] = {name: value for name, value in ordered_parts}
        context["_order"] = [name for name, _ in ordered_parts]
        return context

    # ------------------------------------------------------------------
    # prompt 格式化
    # ------------------------------------------------------------------

    def _format_prompt(self, context: dict, retry_feedback: Optional[str] = None, ts: Optional[int] = None) -> str:
        lines = [
            "You are the Trader agent for AlphaLoop-Crypto (§3.2). "
            "Respond with a JSON list of decision objects only, matching the Decision schema "
            "(ts, symbol, action, target_notional_pct, leverage, thesis, falsifier, horizon). "
            # 2026-07-15强化(API模式真实故障:flash模型带长战术文本时输出
            # action:'open'和ISO字符串ts,6个分支连续3次校验失败全部兜底hold):
            # ①action枚举逐字写死 ②ts直接给出权威整数值,不让模型发明时间戳
            # (桥模式时代是签入agent人工算好的,API模式下模型无从知晓)。
            f"STRICT FORMAT: \"action\" MUST be exactly one of: \"open_long\", \"open_short\", "
            f"\"close\", \"adjust\", \"hold\" -- never \"open\"/\"buy\"/\"sell\"/\"long\"/\"short\". "
            + (f"\"ts\" MUST be exactly the integer {ts} (epoch milliseconds, copy it verbatim, "
               f"NOT an ISO date string). " if ts is not None else "")
            + "UNITS: target_notional_pct is PERCENTAGE POINTS of account NAV -- 15.0 means a "
            "position worth 15% of account value, NOT a 0-1 fraction. Open/adjust decisions "
            "below 5.0 (i.e. under 5% of NAV) are rejected as economically meaningless. "
            "Size positions to match conviction: capital that sits idle earns nothing. "
            "For every decision whose action is NOT \"hold\", you MUST also include "
            "falsifier_condition: a machine-readable price clause in the exact format "
            "'price<NUMBER', 'price<=NUMBER', 'price>NUMBER', or 'price>=NUMBER' "
            "(e.g. \"price<48000\") that a deterministic checker can evaluate later to "
            "decide whether this decision's thesis was falsified -- it must express the "
            "SAME condition as your natural-language falsifier field, just in this fixed "
            "machine-parseable form. Omit it (or set null) for hold decisions.",
            "",
            f"1. Current positions: {context['positions']}",
            f"2. Memory retrieval (top-k): {context['memory_results']}",
            f"3. Latest market snapshot: {context['latest_snapshot']}",
            f"4. Most recent reflection summary: {context['last_reflection_summary']}",
            f"5. Tactical instructions (program.md): {context['program_tactics']}",
        ]
        if retry_feedback:
            lines.append("")
            lines.append(
                "Your previous response failed validation with these errors: "
                f"{retry_feedback}. Fix the issues and resubmit strictly as a JSON list."
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 解析 + 校验
    # ------------------------------------------------------------------

    def _parse_and_validate(
        self, raw_response: str, ts: int, branch: str, valid_symbols: Optional[set] = None
    ) -> tuple[Optional[list[Decision]], Optional[str]]:
        """返回 (decisions, error) —— 恰好一个为 None。

        valid_symbols 是本次决策时刻真实行情快照里覆盖的完整交易对集合(见
        decide() 里 set(latest_snapshot.keys())),用于语义级校验 symbol 字段
        (见 _validate_decision_dict 的说明)——传 None 时跳过这项检查,保持
        向后兼容(测试里裸调用不传快照的场景)。
        """
        try:
            parsed = json.loads(raw_response)
        except (json.JSONDecodeError, TypeError, ValueError):
            return None, "response_not_valid_json"

        if not isinstance(parsed, list):
            return None, "response_not_a_json_list"
        if len(parsed) == 0:
            return None, "response_is_empty_list"

        decisions: list[Decision] = []
        all_errors: list[str] = []
        for i, item in enumerate(parsed):
            if not isinstance(item, dict):
                all_errors.append(f"item_{i}_not_an_object")
                continue
            # ts是基础设施字段(本决策周期的时间戳),不是交易判断——权威值
            # 就是decide()的入参,模型输出什么都以系统值为准。2026-07-15
            # API模式实测模型会编造ISO字符串/错误时间戳,与其反复重试教它
            # 抄对一个它本来就不该负责的数字,不如直接覆盖。
            item["ts"] = ts
            errs = _validate_decision_dict(
                item, max_leverage=self.max_leverage, valid_symbols=valid_symbols
            )
            if errs:
                all_errors.append(f"item_{i}: {'; '.join(errs)}")
                continue
            filtered = {k: v for k, v in item.items() if k in _DECISION_FIELDS}
            filtered.setdefault("branch", branch)
            decisions.append(Decision(**filtered))

        if all_errors:
            return None, "; ".join(all_errors)
        return decisions, None

    @staticmethod
    def _fallback_hold_decision(ts: int, branch: str) -> Decision:
        return Decision(
            ts=ts,
            symbol=_HOLD_FALLBACK_SYMBOL,
            action="hold",
            target_notional_pct=0.0,
            leverage=1,
            thesis=_HOLD_FALLBACK_THESIS,
            falsifier=_HOLD_FALLBACK_FALSIFIER,
            horizon=_HOLD_FALLBACK_HORIZON,
            branch=branch,
        )

    # ------------------------------------------------------------------
    # decide()
    # ------------------------------------------------------------------

    def decide(
        self,
        ts: int,
        positions: list | dict,
        latest_snapshot: dict,
        last_reflection_summary: Optional[str] = None,
        program_tactics: Optional[str] = None,
        memory_query_text: str = "",
        top_k: int = 5,
        branch: str = "main",
    ) -> list[Decision]:
        context = self.build_context(
            positions=positions,
            ts=ts,
            latest_snapshot=latest_snapshot,
            last_reflection_summary=last_reflection_summary,
            program_tactics=program_tactics,
            memory_query_text=memory_query_text,
            top_k=top_k,
            branch=branch,
        )

        # 空快照(比如数据源暂时拉不到行情)时不做symbol语义校验而不是把它当成
        # "合法的空universe"去拒绝所有symbol——宁可退化成只做类型/范围校验,
        # 也不要在真实网络故障期间把这个新校验变成额外一层全部拒绝。
        valid_symbols = set(latest_snapshot.keys()) if latest_snapshot else None

        retry_feedback: Optional[str] = None
        for _attempt in range(1, self.max_retries + 1):
            prompt = self._format_prompt(context, retry_feedback, ts=ts)
            raw_response = self.llm_client(prompt)
            decisions, error = self._parse_and_validate(raw_response, ts, branch, valid_symbols=valid_symbols)
            if error is None:
                return decisions
            retry_feedback = error

        return [self._fallback_hold_decision(ts, branch)]
