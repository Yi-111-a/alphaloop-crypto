"""
circuit_breaker.py —— 熔断器(§2.4)。

职责边界:本模块只负责"算出当前应处于哪个熔断状态、把状态转换记录下来、
供调用方查询状态"。它不清仓、不冻结下单、不判分支FAIL —— 那些是
simulator.py / main.py 调度层看到 FROZEN_FULL / is_frozen()==True 之后
自己去做的事情。

判定规则(check() 内部按此优先级顺序评估,数值来自 config.yaml 的
constraints.max_drawdown_pct / constraints.daily_loss_freeze_pct):

1. 总回撤(相对于"运行到当前为止见过的最高NAV"的回撤)
   > max_drawdown_pct(默认20%) → FROZEN_FULL。
   这是"粘性(sticky)"状态:一旦触发,后续任何 check() 调用都会立刻直接
   返回 FROZEN_FULL,不会因为 NAV 回升而自动解除。只有人类显式调用
   manual_unfreeze() 才能清除 —— 方法名刻意这样命名,提醒它不是任何
   agent/调度代码路径可以自动调用的方法。
2. 否则,若"单日跌幅"(定义见下)> daily_loss_freeze_pct(默认8%)
   → FROZEN_24H。冻结24小时;到期后下一次 check() 会自动把状态转回
   NORMAL(不需要人工介入)。
3. 否则 → NORMAL。

"单日跌幅"的定义(务必与 scorer.py 保持一致 —— 后者若独立推理 NAV 序列,
应采用同一定义):
    取"当前时刻"(check() 的 now_ts 参数;未显式提供时取 nav_series 最后
    一条记录的时间戳)所在的 UTC 自然日 D。在 D 当天、且时间戳 <= 当前时刻
    的全部 NAV 记录中:
        day_start_nav = D 当天时间戳最早的那条 NAV 记录的值
        day_low_nav   = D 当天(不看未来,只看到当前时刻为止)出现过的最低 NAV
        daily_decline_pct = (day_start_nav - day_low_nav) / day_start_nav * 100
    UTC 自然日边界用整数毫秒时间戳 `ts_ms // 86_400_000` 计算,不使用
    datetime 模块的时区相关API,避免时区歧义。

时间来源(铁律,保证可确定性单测,不依赖真实时钟):
    本模块任何地方都不调用 time.time() / datetime.now() / datetime.utcnow()
    或任何其它读取真实时钟的API。"现在"永远来自调用方传入的数据:
    check(nav_series, now_ts=...) 的 now_ts 参数,缺省时取 nav_series 最后
    一条记录的时间戳。
    is_frozen() 是零参数方法,不接受也不产生任何新的时间读数 —— 它只读取
    "上一次 check() 调用后留下的内部状态"。因此 FROZEN_24H 的到期只会在
    "下一次 check() 被调用、且传入的时间已跨过24小时窗口"时才会体现;
    is_frozen() 本身不会在两次 check() 之间凭空感知真实世界时间流逝。

调用约定(与 simulator.py 对齐 —— 后者按此实现,不要破坏):
    simulator.execute() 只调用 breaker.is_frozen()(零参数、不传 nav_series),
    把它当作"内部已跟踪好的最新状态"的只读查询。真正推动状态演进的是调度层
    (main.py)在每个周期结束时用最新 nav_series 调用一次 breaker.check(...)。

状态持久化:
    每一次"状态发生变化"(NORMAL<->FROZEN_24H<->FROZEN_FULL,以及人工
    manual_unfreeze())都会追加一条记录到 LOG/circuit_breaker_state.jsonl
    (通过 LOCKED.log_writer.append_jsonl,只追加,不提供修改/删除)。
    同一状态的重复 check() 调用不会重复写日志。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, Union

from LOCKED.log_writer import append_jsonl
from LOCKED.schemas import CircuitState

try:  # pandas 是项目既有依赖(data_pipeline.py 已使用),这里做可选导入
    import pandas as pd  # type: ignore
except ImportError:  # pragma: no cover - pandas 缺失属于环境问题，非本模块逻辑
    pd = None  # type: ignore

NavPoint = tuple  # (ts_ms: int, nav: float)
NavSeriesLike = Union["pd.Series", Sequence[Any]]

# UTC 自然日的毫秒数,用于 ts_ms // _DAY_MS 得到"UTC 日索引"(整除,无时区歧义)。
_DAY_MS = 24 * 60 * 60 * 1000
# FROZEN_24H 的冻结时长,与"单日"用同一个24小时常数,含义不同但数值相同。
_FREEZE_24H_WINDOW_MS = 24 * 60 * 60 * 1000

_DEFAULT_MAX_DRAWDOWN_PCT = 20.0
_DEFAULT_DAILY_LOSS_FREEZE_PCT = 8.0


def _extract_constraint(config: dict, key: str, default: float) -> float:
    """从 config.yaml 结构(constraints.<key>)取值,兼容测试传入的扁平 dict。"""
    constraints = config.get("constraints") if isinstance(config, dict) else None
    if isinstance(constraints, dict) and key in constraints:
        return float(constraints[key])
    if isinstance(config, dict) and key in config:
        return float(config[key])
    return float(default)


class CircuitBreaker:
    """§2.4 熔断器。只读评估 NAV 序列 → 汇报 CircuitState,不执行清仓/冻结动作。"""

    def __init__(
        self,
        config: dict,
        state_log_path: str = "circuit_breaker_state.jsonl",
        log_root: Optional[Union[str, Path]] = None,
    ) -> None:
        self.max_drawdown_pct = _extract_constraint(
            config, "max_drawdown_pct", _DEFAULT_MAX_DRAWDOWN_PCT
        )
        self.daily_loss_freeze_pct = _extract_constraint(
            config, "daily_loss_freeze_pct", _DEFAULT_DAILY_LOSS_FREEZE_PCT
        )
        self.state_log_path = state_log_path
        self.log_root = Path(log_root) if log_root is not None else None

        self._state: CircuitState = "NORMAL"
        # FROZEN_24H 触发时刻(ms);仅在 state == FROZEN_24H 时有意义。
        self._frozen_24h_trigger_ts: Optional[int] = None

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------
    @property
    def state(self) -> CircuitState:
        """上一次 check()(或 manual_unfreeze())之后留下的当前状态,只读。"""
        return self._state

    def is_frozen(self) -> bool:
        """simulator 的唯一调用点:零参数,不需要重新传入 nav_series。

        True 当且仅当当前内部状态是 FROZEN_FULL,或是一个尚未到期的
        FROZEN_24H(到期与否只由最近一次 check() 的判定结果决定 —— 本方法
        自己不做任何时间运算,见模块顶部"时间来源"说明)。
        """
        return self._state in ("FROZEN_FULL", "FROZEN_24H")

    def check(self, nav_series: NavSeriesLike, now_ts: Optional[int] = None) -> CircuitState:
        """用最新 NAV 历史重新评估熔断状态,返回并更新内部状态。

        Args:
            nav_series: 按时间正序排列的 NAV 历史,代表"截至当前时刻"的净值序列。
                支持:pandas.Series(DatetimeIndex 或数值索引均可)、
                [(ts_ms, nav), ...] 的二元组/列表序列、或 [{"ts":..,"nav":..}, ...]
                的字典序列(也兼容 "timestamp"/"value" 作为键名)。
            now_ts: 显式指定"现在"的 UTC 毫秒时间戳。缺省时取 nav_series 中
                最后一条记录的时间戳。不会读取真实时钟。

        Returns:
            评估后的 CircuitState("NORMAL" / "FROZEN_FULL" / "FROZEN_24H")。
        """
        points = self._normalize_nav_series(nav_series)
        if not points:
            raise ValueError("nav_series must contain at least one (ts, nav) point")

        if now_ts is not None:
            points = [p for p in points if p[0] <= now_ts]
            if not points:
                raise ValueError("no nav_series points at or before the supplied now_ts")
            effective_now_ts = now_ts
        else:
            effective_now_ts = points[-1][0]

        current_nav = points[-1][1]

        # FROZEN_FULL 是粘性状态:一旦触发,直接短路返回,不再评估任何东西,
        # 只有 manual_unfreeze() 能清除它。
        if self._state == "FROZEN_FULL":
            return "FROZEN_FULL"

        # 1) 总回撤(相对运行中最高NAV)判定 —— 优先级最高。
        peak_nav = max(nav for _, nav in points)
        drawdown_pct = self._pct_decline(peak_nav, current_nav)
        if drawdown_pct > self.max_drawdown_pct:
            self._transition(
                "FROZEN_FULL",
                effective_now_ts,
                reason=(
                    f"total drawdown {drawdown_pct:.4f}% from running peak "
                    f"{peak_nav} exceeds max_drawdown_pct={self.max_drawdown_pct}%"
                ),
                extra={"peak_nav": peak_nav, "current_nav": current_nav, "drawdown_pct": drawdown_pct},
            )
            self._frozen_24h_trigger_ts = None
            return "FROZEN_FULL"

        # 2) FROZEN_24H 到期检查(仅当当前状态本身就是 FROZEN_24H 时相关)。
        if self._state == "FROZEN_24H" and self._frozen_24h_trigger_ts is not None:
            elapsed_ms = effective_now_ts - self._frozen_24h_trigger_ts
            if elapsed_ms < _FREEZE_24H_WINDOW_MS:
                # 尚未到期,继续冻结,不重复写日志。
                return "FROZEN_24H"
            # 到期:自动解冻回 NORMAL,再往下用最新数据重新评估是否有新的单日跌幅触发。
            self._transition(
                "NORMAL",
                effective_now_ts,
                reason=f"FROZEN_24H expired after {elapsed_ms} ms >= 24h window",
            )
            self._frozen_24h_trigger_ts = None

        # 3) 单日跌幅判定。
        daily_decline_pct = self._single_day_decline_pct(points, effective_now_ts)
        if daily_decline_pct > self.daily_loss_freeze_pct:
            if self._state != "FROZEN_24H":
                self._transition(
                    "FROZEN_24H",
                    effective_now_ts,
                    reason=(
                        f"single-day decline {daily_decline_pct:.4f}% exceeds "
                        f"daily_loss_freeze_pct={self.daily_loss_freeze_pct}%"
                    ),
                    extra={"daily_decline_pct": daily_decline_pct},
                )
                self._frozen_24h_trigger_ts = effective_now_ts
            return "FROZEN_24H"

        # 4) 无触发 → NORMAL。
        if self._state != "NORMAL":
            self._transition("NORMAL", effective_now_ts, reason="no breach on reevaluation")
        return "NORMAL"

    def manual_unfreeze(self, now_ts: Optional[int] = None, note: str = "") -> None:
        """人工解冻。

        只应由人类操作员调用(例如通过一个人工确认过的运维命令),不应出现在
        任何 agent/调度自动化代码路径里 —— 方法名刻意叫 manual_unfreeze 而不是
        unfreeze,提醒调用方这是一次显式的人工动作。主要用于清除粘性的
        FROZEN_FULL,但对 FROZEN_24H / NORMAL 调用同样安全(直接重置为 NORMAL)。
        """
        previous = self._state
        self._frozen_24h_trigger_ts = None
        self._log_transition(
            previous_state=previous,
            new_state="NORMAL",
            ts=now_ts,
            reason=note or "manual_unfreeze() called by operator",
            extra=None,
            event="manual_unfreeze",
        )
        self._state = "NORMAL"

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    def _transition(
        self,
        new_state: CircuitState,
        ts: int,
        reason: str,
        extra: Optional[dict] = None,
    ) -> None:
        previous = self._state
        self._log_transition(
            previous_state=previous,
            new_state=new_state,
            ts=ts,
            reason=reason,
            extra=extra,
            event="state_transition",
        )
        self._state = new_state

    def _log_transition(
        self,
        previous_state: CircuitState,
        new_state: CircuitState,
        ts: Optional[int],
        reason: str,
        extra: Optional[dict],
        event: str,
    ) -> None:
        record = {
            "ts": ts,
            "event": event,
            "previous_state": previous_state,
            "new_state": new_state,
            "reason": reason,
        }
        if extra:
            record.update(extra)
        append_jsonl(self.state_log_path, record, root=self.log_root)

    @staticmethod
    def _pct_decline(base_nav: float, current_nav: float) -> float:
        if base_nav <= 0:
            return 0.0
        return (base_nav - current_nav) / base_nav * 100.0

    def _single_day_decline_pct(self, points: list, now_ts: int) -> float:
        day_key = now_ts // _DAY_MS
        day_navs = [nav for ts, nav in points if ts // _DAY_MS == day_key]
        if not day_navs:
            return 0.0
        day_start_nav = day_navs[0]  # points 已按时间正序排序,取到的是当天最早一条
        day_low_nav = min(day_navs)
        return self._pct_decline(day_start_nav, day_low_nav)

    @staticmethod
    def _normalize_nav_series(nav_series: NavSeriesLike) -> list:
        """把任意支持的 nav_series 形态归一化为按时间正序排列的 [(ts_ms, nav), ...]。"""
        points: list = []

        if pd is not None and isinstance(nav_series, pd.Series):
            index = nav_series.index
            if isinstance(index, pd.DatetimeIndex):
                ts_values = (index.view("int64") // 1_000_000).tolist()  # ns -> ms
            else:
                ts_values = [int(x) for x in index]
            for ts, nav in zip(ts_values, nav_series.values.tolist()):
                points.append((int(ts), float(nav)))
        else:
            for item in nav_series:
                if isinstance(item, dict):
                    ts = item.get("ts", item.get("timestamp"))
                    nav = item.get("nav", item.get("value"))
                elif isinstance(item, (tuple, list)) and len(item) >= 2:
                    ts, nav = item[0], item[1]
                else:
                    raise TypeError(f"unsupported nav_series item: {item!r}")
                if ts is None or nav is None:
                    raise ValueError(f"nav_series item missing ts/nav: {item!r}")
                points.append((int(ts), float(nav)))

        points.sort(key=lambda p: p[0])
        return points
