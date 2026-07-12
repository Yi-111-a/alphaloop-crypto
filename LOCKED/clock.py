"""
clock.py —— 全项目唯一的墙钟时间来源(M5 main.py 硬要求3)。

铁律:从 M2(ASSET/memory/engine.py)开始,项目里所有消费时间的模块
(memory/cold_start/reflector/researcher/evolution_orchestrator/simulator...)
都被设计成"时间由调用方显式传参,模块自身零墙钟调用"——这条纪律本身没有
问题,但它把"到底谁去调 time.time()/datetime.utcnow()"这个问题推到了
调用者身上,而调用者最终只有一个:main.py 的调度循环。如果调度循环里到处
散落着 datetime.now() / datetime.utcnow() / time.time(),就会重新引入
本地时区混入、以及"同一次调度里前后两次取到的'现在'不一致"这类隐蔽 bug。

本模块把"取现在几点"收敛成唯一一个接口,全项目(main.py 及其调度的一切)
只应该通过这里取时间,不再各自调用标准库的墙钟函数。

- Clock: 协议/抽象基类,只有一个方法 now_ms() -> int(UTC毫秒)。
- SystemClock: 唯一被允许调用 time.time() 的地方——全项目 grep
  `time.time()`/`datetime.utcnow()`/`datetime.now()`,除了这个类的实现和
  这段说明性注释本身,不应该再出现第二处。
- FakeClock: 测试用,显式设置/推进时间,不接触真实时钟,保证调度逻辑的单测
  是完全确定性的、可重放的。
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod


class Clock(ABC):
    """时间提供者协议:唯一方法 now_ms() -> int(UTC 毫秒时间戳)。"""

    @abstractmethod
    def now_ms(self) -> int:
        raise NotImplementedError


class SystemClock(Clock):
    """唯一允许调用真实墙钟的实现。生产环境 main.py 用这个。"""

    def now_ms(self) -> int:
        return int(time.time() * 1000)


class FakeClock(Clock):
    """测试/回放用:时间完全由调用方控制,不接触 time.time()。

    支持两种推进方式:直接 set_ms(ts) 跳到某个时刻,或 advance_ms(delta) 相对
    当前值前进 —— 都不允许时间倒退(与本项目"决策/结算/评分只吃前向数据"的
    整体纪律一致),倒退会抛 ValueError,而不是静默接受一个可能破坏幂等性/
    去重逻辑假设的时间戳。
    """

    def __init__(self, start_ms: int):
        self._now_ms = int(start_ms)

    def now_ms(self) -> int:
        return self._now_ms

    def set_ms(self, ts_ms: int) -> None:
        ts_ms = int(ts_ms)
        if ts_ms < self._now_ms:
            raise ValueError(f"FakeClock cannot move backwards: {ts_ms} < current {self._now_ms}")
        self._now_ms = ts_ms

    def advance_ms(self, delta_ms: int) -> int:
        if delta_ms < 0:
            raise ValueError(f"FakeClock cannot advance by a negative delta: {delta_ms}")
        self._now_ms += int(delta_ms)
        return self._now_ms
