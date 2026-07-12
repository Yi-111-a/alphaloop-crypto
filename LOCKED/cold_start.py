"""
cold_start.py —— 启动状态机(§4.0,两态,写死在LOCKED区)。

铁律相关(§0):
  - 状态机只有 COLD_START -> NORMAL 一个方向,一旦切到 NORMAL 就永久生效,
    check_and_transition() 之后任何调用都不会把状态改回 COLD_START,不
    读取任何"是否回滚"开关(与 simulator.py 的 branch_death 硬编码同一
    纪律)。
  - 状态转移记录只通过 LOCKED.log_writer.append_jsonl() 追加写入,不提供
    任何修改/删除历史记录的接口。
  - 核心决策逻辑(check_and_transition 本体)不调用任何墙钟函数
    (time.time()/datetime.now()/datetime.utcnow()),同一组
    (genesis_path 存在性, hypothesis_count) 输入必须在任何时候产生同样的
    转移判断,可离线、确定性地测试。调用方如果想在日志里记录"这次转移发生
    的时间",必须显式传入 ts 参数——本模块绝不会替调用方去问系统时钟。
  - genesis_path 的存在性检查用的是真实文件系统 Path.exists()。这不是一个
    "墙钟"依赖:检查一个文件在磁盘上是否存在,是调度器周期性调用本模块时
    的合法、幂等的一次性状态读取,和"检索时用 time.time() 做时间衰减"那种
    会导致同一份历史回放在不同运行时刻产生不同结果的信息泄漏完全是两回事。

依赖注入点(供 main.py / 调度器对接):
  - is_cold_start() 是零参 callable,签名与 simulator.Simulator 构造参数
    `cold_start_gate: Callable[[], bool] | None` 完全一致,可以直接
    `Simulator(..., cold_start_gate=gate.is_cold_start)` 注入。
  - is_cold_start() 只读取"上一次 check_and_transition() 的结果"，不会自己
    偷偷去重新检查 genesis.md/假设数——是否重新检查、何时重新检查,是调度器
    (§4.1 每日 09:00 Researcher 检索之后)显式驱动的动作，不是隐式轮询。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from LOCKED import log_writer
from LOCKED.schemas import ColdStartState

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_STATE_PATH = _PROJECT_ROOT / "state" / "cold_start_state.json"


class ColdStartGate:
    """§4.0 启动状态机。COLD_START -> NORMAL,单向,不可逆。"""

    def __init__(
        self,
        genesis_path: str | Path,
        min_hypothesis_count: int = 10,
        state_log_path: str = "cold_start_state.jsonl",
        log_root: Optional[str | Path] = None,
        state_path: Optional[str | Path] = None,
    ):
        self.genesis_path = Path(genesis_path)
        self.min_hypothesis_count = min_hypothesis_count
        self.state_log_path = state_log_path
        self.log_root: Optional[Path] = Path(log_root) if log_root is not None else None
        self.state_path: Path = Path(state_path) if state_path is not None else _DEFAULT_STATE_PATH

        self._state: ColdStartState = self._load_state()

    # ------------------------------------------------------------------
    # 状态持久化(供进程重启恢复,§4.1 崩溃恢复)
    # ------------------------------------------------------------------

    def _load_state(self) -> ColdStartState:
        if self.state_path.exists():
            try:
                data = json.loads(self.state_path.read_text(encoding="utf-8"))
                loaded = data.get("state")
                if loaded in ("COLD_START", "NORMAL"):
                    return loaded  # type: ignore[return-value]
            except (json.JSONDecodeError, OSError):
                pass
        return "COLD_START"

    def _persist_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(
            json.dumps({"state": self._state}, ensure_ascii=False), encoding="utf-8"
        )

    # ------------------------------------------------------------------
    # Simulator 注入契约
    # ------------------------------------------------------------------

    def is_cold_start(self) -> bool:
        """零参 callable,匹配 Simulator(cold_start_gate=...) 的既定契约。
        反映"上一次 check_and_transition() 之后"的状态,不做隐式重新检查。"""
        return self._state == "COLD_START"

    @property
    def state(self) -> ColdStartState:
        return self._state

    # ------------------------------------------------------------------
    # 状态转移(由调度器显式驱动)
    # ------------------------------------------------------------------

    def check_and_transition(self, hypothesis_count: int, ts: Optional[int] = None) -> ColdStartState:
        """检查 genesis_path 是否存在 且 hypothesis_count >= min_hypothesis_count。
        两者都满足且当前仍是 COLD_START -> 切换到 NORMAL,记录一条转移日志,
        并持久化到 state_path。一旦已经是 NORMAL,本方法就是永久性 no-op,
        直接返回 "NORMAL",不再理会传入的 hypothesis_count/genesis_path 现状
        (单向状态机,不回滚)。

        注意:本方法核心判断逻辑不调用任何墙钟函数;ts 只用于日志记录，
        None 时不在日志记录里写时间戳字段(不用 time.time()/datetime.now()
        兜底)。
        """
        if self._state == "NORMAL":
            return "NORMAL"

        genesis_exists = self.genesis_path.exists()
        if genesis_exists and hypothesis_count >= self.min_hypothesis_count:
            self._state = "NORMAL"
            self._persist_state()

            record = {
                "from_state": "COLD_START",
                "to_state": "NORMAL",
                "genesis_path": str(self.genesis_path),
                "hypothesis_count": hypothesis_count,
            }
            if ts is not None:
                record["ts"] = ts
            log_writer.append_jsonl(self.state_log_path, record, root=self.log_root)

        return self._state
