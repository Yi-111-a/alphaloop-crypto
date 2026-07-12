"""
simulator.py —— USDT 本位永续合约模拟撮合引擎(§2.2)。

铁律相关(§0):
  - 本模块是纯模拟盘:不下任何真实订单、不连接任何需要 apiKey 的交易所接口。
  - 决策必须先落盘 decisions.jsonl,execute() 才接受该决策(见 _decision_is_logged）。
  - 所有 LOG/ 写入一律走 LOCKED.log_writer.append_jsonl(),该 writer 不提供
    修改/删除接口——本文件里没有、也不会有任何"改历史记录"的代码路径。
  - liquidation_policy=branch_death 不可配置绕过:check_liquidation() 触发爆仓时
    硬编码地把 self.branch_dead 置 True,不读取任何"是否启用"开关。

关键依赖注入点(供并行开发的其它模块对接):
  - circuit_breaker: 任意对象,只要求有 `is_frozen() -> bool` 方法。
    未注入(None)时视为"从不冻结"，仅用于独立单测；生产环境必须注入真实
    circuit_breaker.py 的实例。
  - cold_start_gate: 任意零参 callable，返回 bool，True 表示"当前处于
    COLD_START，禁止一切交易"（§4.0）。作为 execute() 校验链的第 0 步，
    优先于第 1 步的"禁止偷看未来"检查——冷启动期间的拒绝不该跟任何具体
    K线时间戳挂钩，是一个更早、更硬的闸门。未注入(None)时视为"从不处于
    冷启动"，仅用于独立单测；生产环境必须注入真实冷启动状态机的
    is_cold_start 方法。
  - funding_rate_lookup: Callable[[symbol: str, ts_utc: int], float]，由调用方
    （生产环境里是 main.py，包一层 DataPipeline.fetch_funding_rate_history /
    fetch_funding_rate）注入给 settle_funding()，本模块不直接依赖 data_pipeline。
  - universe_symbols: list[str] | None，由 universe_filter.py 生成的
    universe_active.json 加载后传入。None/[] 时本模块视为"不限制"，
    但这只允许用于测试——生产环境 main.py 必须始终传入真实名单。
"""
from __future__ import annotations

import dataclasses
import sqlite3
from pathlib import Path
from typing import Any, Callable, Optional

from LOCKED import log_writer
from LOCKED.schemas import (
    Decision,
    FundingSettlement,
    LiquidationEvent,
    PerpPosition,
    Rejection,
    Trade,
)

# ---------------------------------------------------------------------------
# 常量(§1 config.yaml 之外、spec 里明确写死的数字)
# ---------------------------------------------------------------------------

MIN_THESIS_LEN = 20
# 滑点加倍的24h成交额门槛:5亿USDT（§1 fees.slippage_bps 注释）。
# 这个门槛本身不在 config.yaml 里，是 spec 写死的数字，不做成可配置项。
SLIPPAGE_DOUBLE_VOLUME_THRESHOLD_USDT = 500e6
# 维持保证金率两档（§2.2）。
MMR_BTC_ETH = 0.005
MMR_OTHER = 0.01

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class InvalidActionError(Exception):
    """内部异常:action 与当前持仓状态不匹配（如对不存在的仓位 close/adjust）。
    在 execute() 内被捕获并转换为 Rejection，不会抛出给调用方。"""


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """attribute-or-key fallback:next_bar / mark price bar 既可以是 dict 也可以是
    任意带属性的对象（namedtuple / dataclass / SimpleNamespace 等）。"""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


class Simulator:
    """永续合约模拟撮合引擎。每个 Simulator 实例对应一个策略分支(branch)的独立账本。"""

    def __init__(
        self,
        config: dict,
        circuit_breaker=None,
        cold_start_gate: Optional[Callable[[], bool]] = None,
        universe_symbols: Optional[list[str]] = None,
        decisions_log_path: str = "decisions.jsonl",
        db_path: Optional[str | Path] = None,
        branch: str = "main",
        log_root: Optional[str | Path] = None,
        resume: bool = False,
    ):
        self.config = config
        self.circuit_breaker = circuit_breaker
        self.cold_start_gate = cold_start_gate
        self.universe_symbols = list(universe_symbols) if universe_symbols else None
        self.decisions_log_path = decisions_log_path
        self.branch = branch
        self.branch_dead = False

        # log_root=None 时透传 None 给 log_writer，其内部默认落到项目 LOG/ 目录。
        self.log_root: Optional[Path] = Path(log_root) if log_root is not None else None

        fees_cfg = config.get("fees", {}) or {}
        self.taker_pct = float(fees_cfg.get("taker_pct", 0.0005))
        self.base_slippage_bps = float(fees_cfg.get("slippage_bps", 15))

        lev_cfg = config.get("leverage", {}) or {}
        self.max_leverage = int(lev_cfg.get("max", 10))

        self.constraints = config.get("constraints", {}) or {}

        safe_branch = branch.replace("/", "_").replace(":", "_").replace(" ", "_")
        if db_path is None:
            db_path = _PROJECT_ROOT / "state" / f"portfolio_{safe_branch}.db"
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = self._init_db()

        self.positions: dict[str, PerpPosition] = {}
        self.wallet_balance: float = float(config.get("capital_usdt", 100000))
        # 最近已知标记价，用于在没有实时快照时估算 NAV（见 _current_nav 文档）。
        self._last_prices: dict[str, float] = {}

        loaded = False
        if resume:
            loaded = self._load_state()
        if not loaded:
            # Only write a baseline row if this branch has no persisted state yet.
            # A non-resuming instance must not clobber another process's/instance's
            # already-persisted state for the same branch+db (it simply doesn't load
            # it into memory); it only starts writing once it changes something.
            existing_row = self._conn.execute(
                "SELECT 1 FROM wallet WHERE branch = ?", (self.branch,)
            ).fetchone()
            if existing_row is None:
                self._persist_state()

    # ------------------------------------------------------------------
    # sqlite 持久化(账本崩溃可恢复)
    # ------------------------------------------------------------------

    def _init_db(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.execute(
            """CREATE TABLE IF NOT EXISTS wallet (
                branch TEXT PRIMARY KEY,
                balance REAL NOT NULL,
                branch_dead INTEGER NOT NULL DEFAULT 0
            )"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS positions (
                branch TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                notional REAL NOT NULL,
                entry_price REAL NOT NULL,
                margin REAL NOT NULL,
                leverage INTEGER NOT NULL,
                PRIMARY KEY (branch, symbol)
            )"""
        )
        # M5 崩溃恢复幂等性:资金费率结算的"已生效"标记必须和 wallet_balance
        # 的扣减在同一个 sqlite 事务里提交,否则"结算写盘后、状态更新前杀进程"
        # 这类场景无法被安全去重(见 settle_funding() 的详细说明)。用 sqlite 的
        # 事务原子性做这个保证,而不是依赖 funding.jsonl(那只是尽力而为的
        # 审计记录,不是幂等性判定的依据)。
        conn.execute(
            """CREATE TABLE IF NOT EXISTS applied_funding_settlements (
                branch TEXT NOT NULL,
                symbol TEXT NOT NULL,
                ts INTEGER NOT NULL,
                PRIMARY KEY (branch, symbol, ts)
            )"""
        )
        conn.commit()
        return conn

    def _load_state(self) -> bool:
        row = self._conn.execute(
            "SELECT balance, branch_dead FROM wallet WHERE branch = ?", (self.branch,)
        ).fetchone()
        if row is None:
            return False
        self.wallet_balance = row[0]
        self.branch_dead = bool(row[1])
        self.positions = {}
        for r in self._conn.execute(
            "SELECT symbol, side, notional, entry_price, margin, leverage FROM positions WHERE branch = ?",
            (self.branch,),
        ):
            self.positions[r[0]] = PerpPosition(
                symbol=r[0], side=r[1], notional=r[2], entry_price=r[3], margin=r[4], leverage=r[5]
            )
        return True

    def _persist_state(self, newly_applied_settlement_ids: list[tuple[str, int]] | None = None) -> None:
        """落盘 wallet + positions,以及(可选)本次新生效的资金费率结算ID。

        `newly_applied_settlement_ids` 是 settle_funding() 用来在**同一个 sqlite
        事务**里把"这笔结算已经生效"的标记和"wallet_balance 已经反映这笔结算"
        绑在一起提交的入口——两者要么一起成功、要么一起失败,不存在"标记已生效
        但余额没扣"或者"余额扣了但标记没写"这种中间态,这正是保证结算幂等性
        所需要的原子性来源(sqlite 事务),而不是 funding.jsonl 这个纯 append-only
        审计文件本身。
        """
        self._conn.execute(
            """INSERT INTO wallet(branch, balance, branch_dead) VALUES (?, ?, ?)
               ON CONFLICT(branch) DO UPDATE SET balance = excluded.balance,
                                                  branch_dead = excluded.branch_dead""",
            (self.branch, self.wallet_balance, int(self.branch_dead)),
        )
        self._conn.execute("DELETE FROM positions WHERE branch = ?", (self.branch,))
        for pos in self.positions.values():
            self._conn.execute(
                """INSERT INTO positions(branch, symbol, side, notional, entry_price, margin, leverage)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (self.branch, pos.symbol, pos.side, pos.notional, pos.entry_price, pos.margin, pos.leverage),
            )
        for symbol, ts in newly_applied_settlement_ids or []:
            self._conn.execute(
                """INSERT OR IGNORE INTO applied_funding_settlements(branch, symbol, ts)
                   VALUES (?, ?, ?)""",
                (self.branch, symbol, ts),
            )
        self._conn.commit()

    def _funding_settlement_already_applied(self, symbol: str, ts_utc: int) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM applied_funding_settlements WHERE branch = ? AND symbol = ? AND ts = ?",
            (self.branch, symbol, ts_utc),
        ).fetchone()
        return row is not None

    # ------------------------------------------------------------------
    # 辅助:落盘决策(生产流程里由框架在 execute() 之前调用；测试也可直接用)
    # ------------------------------------------------------------------

    def log_decision(self, decision: Decision) -> None:
        log_writer.append_jsonl(self.decisions_log_path, decision, root=self.log_root)

    def _decision_is_logged(self, decision: Decision) -> bool:
        """§0 铁律3 / §2.2 校验链步骤2 的实现方式:
        读取 decisions_log_path（默认 decisions.jsonl），在其中查找一条
        (ts, symbol, action, branch) 完全匹配的记录。生产环境的真实流程是:
        框架在调用 execute() 之前先 log_writer.append_jsonl() 把 decision
        写入 decisions.jsonl，所以这里只需要读回来做存在性校验即可。
        """
        records = log_writer.read_jsonl(self.decisions_log_path, root=self.log_root)
        for r in records:
            if (
                r.get("ts") == decision.ts
                and r.get("symbol") == decision.symbol
                and r.get("action") == decision.action
                and r.get("branch", "main") == decision.branch
            ):
                return True
        return False

    def _reject(self, decision: Decision, reason: str) -> Rejection:
        rejection = Rejection(ts=decision.ts, symbol=decision.symbol, reason=reason, decision=decision)
        log_writer.append_jsonl("rejections.jsonl", rejection, root=self.log_root)
        return rejection

    # ------------------------------------------------------------------
    # 定价 / 手续费
    # ------------------------------------------------------------------

    def _slippage_bps_for(self, symbol: str, next_bar: Any) -> float:
        volume = _get(next_bar, "volume_24h_usdt", None)
        if volume is not None and volume < SLIPPAGE_DOUBLE_VOLUME_THRESHOLD_USDT:
            return self.base_slippage_bps * 2
        return self.base_slippage_bps

    def _fill_price(self, next_bar: Any, sign: int, symbol: str) -> float:
        open_price = float(_get(next_bar, "open"))
        bps = self._slippage_bps_for(symbol, next_bar)
        return open_price * (1 + bps / 1e4 * sign)

    @staticmethod
    def _unrealized_pnl(pos: PerpPosition, price: float) -> float:
        if pos.side == "long":
            return pos.notional / pos.entry_price * (price - pos.entry_price)
        return pos.notional / pos.entry_price * (pos.entry_price - price)

    def _nav_of(
        self,
        positions: dict[str, PerpPosition],
        wallet_balance: float,
        price_hint: Optional[dict[str, float]] = None,
    ) -> float:
        price_hint = price_hint or {}
        total_upnl = 0.0
        for sym, pos in positions.items():
            price = price_hint.get(sym, self._last_prices.get(sym, pos.entry_price))
            total_upnl += self._unrealized_pnl(pos, price)
        return wallet_balance + total_upnl

    def _current_nav(self) -> float:
        """决策落地时用于计算 target_notional_pct 的"当前净值"。
        没有实时快照时，用每个持仓最近一次已知标记价（否则退化为开仓价，
        即假设该仓位当前未实现盈亏为0）。这是一个简化假设，已在文档中注明。
        """
        return self._nav_of(self.positions, self.wallet_balance)

    @staticmethod
    def _maintenance_margin_rate(symbol: str) -> float:
        if "BTC" in symbol or "ETH" in symbol:
            return MMR_BTC_ETH
        return MMR_OTHER

    # ------------------------------------------------------------------
    # execute()
    # ------------------------------------------------------------------

    def _compute_fill(self, decision: Decision, next_bar: Any) -> dict:
        symbol = decision.symbol
        action = decision.action
        existing = self.positions.get(symbol)
        positions_copy = dict(self.positions)
        wallet = self.wallet_balance

        if action == "hold":
            fill_price = float(_get(next_bar, "open"))
            return dict(
                fill_price=fill_price,
                fee=0.0,
                notional_filled=0.0,
                side=(existing.side if existing else "long"),
                would_be_positions=positions_copy,
                would_be_wallet_balance=wallet,
                realized_pnl=0.0,
            )

        nav_pre = self._current_nav()

        if action in ("open_long", "open_short"):
            desired_side = "long" if action == "open_long" else "short"
            additional_notional = abs(decision.target_notional_pct) / 100.0 * nav_pre

            if existing is None or existing.side == desired_side:
                sign = 1 if desired_side == "long" else -1
                fill_price = self._fill_price(next_bar, sign, symbol)
                fee = additional_notional * self.taker_pct
                if existing is None:
                    new_pos = PerpPosition(
                        symbol=symbol,
                        side=desired_side,
                        notional=additional_notional,
                        entry_price=fill_price,
                        margin=additional_notional / decision.leverage,
                        leverage=decision.leverage,
                    )
                else:
                    new_notional = existing.notional + additional_notional
                    additional_margin = additional_notional / decision.leverage
                    new_margin = existing.margin + additional_margin
                    new_entry = (
                        existing.notional * existing.entry_price + additional_notional * fill_price
                    ) / new_notional
                    new_pos = PerpPosition(
                        symbol=symbol,
                        side=desired_side,
                        notional=new_notional,
                        entry_price=new_entry,
                        margin=new_margin,
                        leverage=decision.leverage,
                    )
                positions_copy[symbol] = new_pos
                return dict(
                    fill_price=fill_price,
                    fee=fee,
                    notional_filled=additional_notional,
                    side=desired_side,
                    would_be_positions=positions_copy,
                    would_be_wallet_balance=wallet - fee,
                    realized_pnl=0.0,
                )
            else:
                # 反向持仓存在:先平旧仓（结算盈亏），再开新仓。
                close_sign = -1 if existing.side == "long" else 1
                close_price = self._fill_price(next_bar, close_sign, symbol)
                realized_close = self._unrealized_pnl(existing, close_price)
                close_fee = existing.notional * self.taker_pct

                open_sign = 1 if desired_side == "long" else -1
                open_price = self._fill_price(next_bar, open_sign, symbol)
                open_fee = additional_notional * self.taker_pct

                new_pos = PerpPosition(
                    symbol=symbol,
                    side=desired_side,
                    notional=additional_notional,
                    entry_price=open_price,
                    margin=additional_notional / decision.leverage,
                    leverage=decision.leverage,
                )
                positions_copy[symbol] = new_pos
                fee = close_fee + open_fee
                return dict(
                    fill_price=open_price,
                    fee=fee,
                    notional_filled=additional_notional + existing.notional,
                    side=desired_side,
                    would_be_positions=positions_copy,
                    would_be_wallet_balance=wallet + realized_close - fee,
                    realized_pnl=realized_close,
                )

        elif action == "close":
            if existing is None:
                raise InvalidActionError(f"no_open_position: cannot close {symbol}, no open position")
            close_sign = -1 if existing.side == "long" else 1
            fill_price = self._fill_price(next_bar, close_sign, symbol)
            fee = existing.notional * self.taker_pct
            realized = self._unrealized_pnl(existing, fill_price)
            positions_copy.pop(symbol, None)
            return dict(
                fill_price=fill_price,
                fee=fee,
                notional_filled=existing.notional,
                side=existing.side,
                would_be_positions=positions_copy,
                would_be_wallet_balance=wallet + realized - fee,
                realized_pnl=realized,
            )

        elif action == "adjust":
            if existing is None:
                raise InvalidActionError(f"no_open_position: cannot adjust {symbol}, no open position")
            new_target_notional = abs(decision.target_notional_pct) / 100.0 * nav_pre
            delta = new_target_notional - existing.notional

            if delta > 0:
                sign = 1 if existing.side == "long" else -1
                fill_price = self._fill_price(next_bar, sign, symbol)
                fee = delta * self.taker_pct
                additional_margin = delta / decision.leverage
                new_notional = existing.notional + delta
                new_margin = existing.margin + additional_margin
                new_entry = (existing.notional * existing.entry_price + delta * fill_price) / new_notional
                new_pos = PerpPosition(
                    symbol=symbol,
                    side=existing.side,
                    notional=new_notional,
                    entry_price=new_entry,
                    margin=new_margin,
                    leverage=decision.leverage,
                )
                positions_copy[symbol] = new_pos
                return dict(
                    fill_price=fill_price,
                    fee=fee,
                    notional_filled=delta,
                    side=existing.side,
                    would_be_positions=positions_copy,
                    would_be_wallet_balance=wallet - fee,
                    realized_pnl=0.0,
                )
            elif delta < 0:
                closed_notional = abs(delta)
                sign = -1 if existing.side == "long" else 1
                fill_price = self._fill_price(next_bar, sign, symbol)
                fee = closed_notional * self.taker_pct
                closed_fraction = closed_notional / existing.notional
                total_upnl = self._unrealized_pnl(existing, fill_price)
                realized = total_upnl * closed_fraction
                remaining_notional = existing.notional - closed_notional
                if remaining_notional <= 1e-9:
                    positions_copy.pop(symbol, None)
                else:
                    remaining_margin = existing.margin * (1 - closed_fraction)
                    positions_copy[symbol] = PerpPosition(
                        symbol=symbol,
                        side=existing.side,
                        notional=remaining_notional,
                        entry_price=existing.entry_price,
                        margin=remaining_margin,
                        leverage=existing.leverage,
                    )
                return dict(
                    fill_price=fill_price,
                    fee=fee,
                    notional_filled=closed_notional,
                    side=existing.side,
                    would_be_positions=positions_copy,
                    would_be_wallet_balance=wallet + realized - fee,
                    realized_pnl=realized,
                )
            else:
                fill_price = float(_get(next_bar, "open"))
                return dict(
                    fill_price=fill_price,
                    fee=0.0,
                    notional_filled=0.0,
                    side=existing.side,
                    would_be_positions=positions_copy,
                    would_be_wallet_balance=wallet,
                    realized_pnl=0.0,
                )
        else:
            raise InvalidActionError(f"unknown_action: {action!r}")

    def execute(self, decision: Decision, next_bar: Any):
        """撮合一条决策。校验链顺序严格按 spec §2.2 的 9 步（见模块顶部注释）。
        第一个未通过的检查决定 Rejection.reason，返回 Rejection 而不是抛异常。
        """
        # 0. 冷启动期间禁止一切交易(§4.0),优先于其它一切检查。
        if self.cold_start_gate is not None and self.cold_start_gate():
            return self._reject(decision, "cold_start_active: trading is disabled during COLD_START")

        open_time = _get(next_bar, "open_time")

        # 1. 禁止偷看未来
        if open_time is None or not (decision.ts < open_time):
            return self._reject(
                decision, "future_peeking: decision.ts must be strictly earlier than next_bar.open_time"
            )

        # 2. 决策必须先落盘 decisions.jsonl
        if not self._decision_is_logged(decision):
            return self._reject(
                decision, "decision_not_logged: decision not found in decisions.jsonl prior to execution"
            )

        # 3. thesis / falsifier 非空且 >= 20 字符
        thesis = (decision.thesis or "").strip()
        falsifier = (decision.falsifier or "").strip()
        if len(thesis) < MIN_THESIS_LEN or len(falsifier) < MIN_THESIS_LEN:
            return self._reject(
                decision,
                f"thesis_or_falsifier_invalid: thesis and falsifier must both be non-empty "
                f"and >= {MIN_THESIS_LEN} characters",
            )

        # 4. symbol 必须在合格名单内(universe_symbols 为空仅测试环境允许"不限制")
        if self.universe_symbols and decision.symbol not in self.universe_symbols:
            return self._reject(
                decision, f"symbol_not_in_universe: {decision.symbol} not in active universe"
            )

        # 5. 杠杆硬上限
        if decision.leverage > self.max_leverage or decision.leverage < 1:
            return self._reject(
                decision,
                f"leverage_exceeds_max: leverage {decision.leverage} must be between 1 and {self.max_leverage}",
            )

        # NAV 基准取"本次成交前"(与 _compute_fill 内 target_notional_pct 定
        # 仓所用的 nav_pre 保持一致)。§2.2 的"成交后...<=上限"检查若改用扣除
        # 本笔手续费之后的 NAV 做分母,会导致 target_notional_pct 恰好等于上限
        # (例如 BTC_HOLD 基准的 100%)的决策必然因为自身手续费被拒——手续费是
        # 已发生的成本,不应反过来让当初按当前 NAV 合理定仓的决策失效。因此仓位
        # /总敞口上限检查用 nav_pre;可用保证金比例检查仍用 nav_post,因为那是
        # 真实成交后的保证金占用情况,理应反映手续费扣减。
        nav_pre = self._current_nav()

        try:
            fill = self._compute_fill(decision, next_bar)
        except InvalidActionError as exc:
            return self._reject(decision, str(exc))

        would_be_positions = fill["would_be_positions"]
        would_be_wallet_balance = fill["would_be_wallet_balance"]
        nav_post = self._nav_of(
            would_be_positions, would_be_wallet_balance, price_hint={decision.symbol: fill["fill_price"]}
        )

        max_pos_pct = self.constraints.get("max_position_notional_pct", 100)
        max_total_pct = self.constraints.get("max_total_notional_pct", 300)
        min_free_pct = self.constraints.get("min_free_margin_pct", 15)

        # 6. 单币名义 <= max_position_notional_pct% of NAV(基准:成交前 NAV)
        pos_after = would_be_positions.get(decision.symbol)
        nav_for_notional_caps = nav_pre if nav_pre > 0 else nav_post
        if pos_after is not None and nav_for_notional_caps > 0:
            limit = max_pos_pct / 100.0 * nav_for_notional_caps
            if abs(pos_after.notional) > limit + 1e-6:
                return self._reject(
                    decision,
                    f"position_notional_exceeds_limit: {abs(pos_after.notional):.4f} > {limit:.4f}",
                )

        # 7. 总名义 <= max_total_notional_pct% of NAV(基准:成交前 NAV)
        total_notional = sum(abs(p.notional) for p in would_be_positions.values())
        if nav_for_notional_caps > 0:
            total_limit = max_total_pct / 100.0 * nav_for_notional_caps
            if total_notional > total_limit + 1e-6:
                return self._reject(
                    decision, f"total_notional_exceeds_limit: {total_notional:.4f} > {total_limit:.4f}"
                )

        # 8. 成交后可用保证金比例 >= min_free_margin_pct%
        total_margin = sum(p.margin for p in would_be_positions.values())
        if nav_post <= 0:
            return self._reject(decision, "free_margin_below_minimum: non-positive NAV after fill")
        free_margin_ratio = (nav_post - total_margin) / nav_post
        if free_margin_ratio < min_free_pct / 100.0 - 1e-9:
            return self._reject(
                decision,
                f"free_margin_below_minimum: free margin ratio {free_margin_ratio * 100:.4f}% "
                f"< required {min_free_pct}%",
            )

        # 9. 熔断器未冻结
        if self.circuit_breaker is not None and self.circuit_breaker.is_frozen():
            return self._reject(decision, "circuit_breaker_frozen: trading is currently frozen")

        # ---- 全部校验通过,提交成交 ----
        self.positions = would_be_positions
        self.wallet_balance = would_be_wallet_balance
        self._last_prices[decision.symbol] = fill["fill_price"]
        self._persist_state()

        trade = Trade(
            ts=open_time,
            symbol=decision.symbol,
            side=fill["side"],
            action=decision.action,
            notional=fill["notional_filled"],
            price=fill["fill_price"],
            fee=fill["fee"],
            slippage_bps=self._slippage_bps_for(decision.symbol, next_bar),
            leverage=decision.leverage,
            branch=decision.branch,
        )
        log_writer.append_jsonl("trades.jsonl", trade, root=self.log_root)
        return trade

    # ------------------------------------------------------------------
    # settle_funding()
    # ------------------------------------------------------------------

    def settle_funding(
        self, ts_utc: int, funding_rate_lookup: Callable[[str, int], float]
    ) -> list[FundingSettlement]:
        """在 UTC 00:00/08:00/16:00 调用。funding_rate_lookup(symbol, ts_utc) 由调用方
        注入(生产环境包一层 DataPipeline.fetch_funding_rate_history)，本模块不直接
        依赖 data_pipeline，从而与并行开发的模块解耦。

        符号约定(与 FundingSettlement.amount 一致): amount 为正 = 账户被扣款,
        为负 = 账户收款。 rate>0 时多仓付款(amount=+notional*rate)、空仓收款
        (amount=-notional*rate)；rate<0 时反之。

        M5 崩溃恢复幂等性(硬要求1):每笔结算的确定性ID是 (branch, symbol, ts_utc)。
        LOG/funding.jsonl 是"结算意图"的 append-only 提交日志——一笔结算先写进
        这里,再把 wallet_balance 的扣减和"已生效"标记一起原子提交到 sqlite
        (applied_funding_settlements 表,见 _persist_state)。这个先后顺序刻意
        设计成能安全处理"结算写盘后、状态更新前杀进程"这个具体场景:

          - 若 (branch,symbol,ts_utc) 在 sqlite 的 applied_funding_settlements 里
            已存在 → 完全生效过了(日志写了、余额也扣了),本次调用对该symbol
            整体跳过,不重复扣款、不重复写日志。
          - 若 funding.jsonl 里已经有这条记录,但 sqlite 标记还不存在 → 正是
            "写盘后、状态更新前"崩溃的那个窗口。恢复时**复用日志里已经写死的
            rate/amount**去补做 wallet_balance 扣减 + 标记生效,不重新调用
            funding_rate_lookup 现算一次——重新现算可能因为费率数据源的时效性
            拿到一个不同的数字,那样结算金额就和已经落盘的审计记录对不上了,
            这本身就是另一种"重复/不一致结算"。
          - 若两边都没有 → 全新结算:先把它写进 funding.jsonl,再扣款+标记生效。

        崩溃恢复的完整性依赖 sqlite 事务的原子性(wallet_balance 扣减和
        applied_funding_settlements 标记在同一次 commit 里),funding.jsonl 单纯
        追加写入不保证原子性,但它的角色只是"结算意图的提交日志",不是幂等性
        判定本身的依据——sqlite 侧的标记才是。
        """
        settlements: list[FundingSettlement] = []
        newly_applied_ids: list[tuple[str, int]] = []

        existing_records = log_writer.read_jsonl("funding.jsonl", root=self.log_root)
        existing_by_symbol: dict[str, dict] = {}
        for r in existing_records:
            if r.get("branch") == self.branch and r.get("ts") == ts_utc:
                existing_by_symbol[r["symbol"]] = r

        for symbol, pos in list(self.positions.items()):
            if self._funding_settlement_already_applied(symbol, ts_utc):
                continue  # fully committed already (logged AND applied) -- no-op

            existing_record = existing_by_symbol.get(symbol)
            if existing_record is not None:
                # Crash-recovery reconciliation: reuse the rate/amount already
                # committed to the log, do not re-fetch a fresh rate.
                rate = existing_record["funding_rate"]
                amount = existing_record["amount"]
                settlement = FundingSettlement(
                    ts=ts_utc,
                    symbol=symbol,
                    branch=self.branch,
                    side=existing_record["side"],
                    notional=existing_record["notional"],
                    funding_rate=rate,
                    amount=amount,
                )
            else:
                rate = funding_rate_lookup(symbol, ts_utc)
                if pos.side == "long":
                    amount = pos.notional * rate
                else:
                    amount = -pos.notional * rate
                settlement = FundingSettlement(
                    ts=ts_utc,
                    symbol=symbol,
                    branch=self.branch,
                    side=pos.side,
                    notional=pos.notional,
                    funding_rate=rate,
                    amount=amount,
                )
                log_writer.append_jsonl("funding.jsonl", settlement, root=self.log_root)

            self.wallet_balance -= amount
            newly_applied_ids.append((symbol, ts_utc))
            settlements.append(settlement)

        if newly_applied_ids:
            self._persist_state(newly_applied_settlement_ids=newly_applied_ids)
        return settlements

    # ------------------------------------------------------------------
    # check_liquidation()
    # ------------------------------------------------------------------

    def check_liquidation(
        self, mark_prices: dict[str, dict], ts_utc: Optional[int] = None
    ) -> list[LiquidationEvent]:
        """每根K线调用一次。mark_prices[symbol] 至少包含 high/low/close，
        用 K 线内最不利价格（多仓看 low、空仓看 high）判断是否插针爆仓，
        即使收盘价已经回到爆仓价上方/下方也照样触发——这是 spec 明确要求的
        正确性保证，不能只看 close。

        ts_utc 为可选参数：优先使用调用方显式传入的时间戳；否则尝试从
        mark_prices[symbol] 里读取 "ts"/"open_time"/"close_time" 字段；
        都没有时退化为 0（仅测试场景可能出现，生产环境请始终传 ts_utc）。

        爆仓时的简化处理（模拟盘，非真实交易所）:仓位保证金全部损失并从
        wallet_balance 里扣除，亏损被封顶在 margin 这个数额（不会让浮亏
        进一步击穿账户），随后移除该仓位、把 self.branch_dead 硬编码置 True。
        """
        events: list[LiquidationEvent] = []
        for symbol in list(self.positions.keys()):
            if symbol not in mark_prices:
                continue
            pos = self.positions[symbol]
            bar = mark_prices[symbol]
            high = _get(bar, "high")
            low = _get(bar, "low")
            if pos.notional == 0:
                continue

            mmr = self._maintenance_margin_rate(symbol)
            margin_over_notional = pos.margin / pos.notional

            if pos.side == "long":
                p_liq = pos.entry_price * (1 + mmr - margin_over_notional)
                triggered = low is not None and low < p_liq
            else:
                p_liq = pos.entry_price * (1 - mmr + margin_over_notional)
                triggered = high is not None and high > p_liq

            if not triggered:
                continue

            event_ts = ts_utc
            if event_ts is None:
                event_ts = _get(bar, "ts", _get(bar, "open_time", _get(bar, "close_time", 0)))

            margin_lost = pos.margin
            event = LiquidationEvent(
                ts=event_ts,
                symbol=symbol,
                branch=self.branch,
                side=pos.side,
                notional=pos.notional,
                entry_price=pos.entry_price,
                liquidation_price=p_liq,
                margin_lost=margin_lost,
            )
            # 模拟爆仓:损失全部保证金(钱包不返还)，仓位清零。
            self.wallet_balance -= margin_lost
            del self.positions[symbol]
            self.branch_dead = True  # liquidation_policy=branch_death,硬编码不可配置
            log_writer.append_jsonl("liquidations.jsonl", event, root=self.log_root)
            events.append(event)

        if events:
            self._persist_state()
        return events

    # ------------------------------------------------------------------
    # mark_to_market() / get_portfolio()
    # ------------------------------------------------------------------

    def mark_to_market(self, snapshot: dict[str, float]) -> float:
        """NAV = wallet_balance + Σ unrealized_pnl，使用传入的最新标记价快照。"""
        self._last_prices.update(snapshot)
        nav = self._nav_of(self.positions, self.wallet_balance, price_hint=snapshot)
        return nav

    def get_portfolio(self, snapshot: Optional[dict[str, float]] = None) -> dict:
        if snapshot:
            nav = self.mark_to_market(snapshot)
        else:
            nav = self._nav_of(self.positions, self.wallet_balance)
        return {
            "wallet_balance": self.wallet_balance,
            "positions": list(self.positions.values()),
            "nav": nav,
            "branch": self.branch,
            "branch_dead": self.branch_dead,
        }
