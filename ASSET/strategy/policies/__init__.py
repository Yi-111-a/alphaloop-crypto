"""
ASSET/strategy/policies —— M7 确定性策略代码协议(spec M7)。

背景:M1-M6 的 Trader 是"LLM 每周期现场生成 Decision"。M7 引入另一条路径——
把战术预先编译成**确定性 Python 代码**(本包下的 policy 模块),同一个输入
ctx 必须产出逐字段相同的 Decision 列表,不依赖 LLM 的现场采样,也就没有
"同一份历史回放,今天跑和明天跑结果不一样"这类不确定性来源。这不是要取代
Trader/LLM 路径,而是为"锦标赛"里的战术分支提供一种可回归测试、可静态审查
的替代实现方式。

本文件定义:
  1. StrategyContext —— policy 模块的唯一输入,一个不可变的轻量 dataclass。
     刻意做成"平铺的几个字段",而不是把 Simulator/MemoryStore 等对象整个
     传进去——policy 代码只应该看到它执行决策所需要的只读快照,不应该有
     机会调用某个对象上意料之外的方法(比如不小心调用了 memory_store.write,
     或者拿到 Simulator 引用后读到未来的账本状态)。这与 LOCKED/reflector.py
     "reflect() 参数表严格封死"是同一种设计考虑。
  2. StrategyFn —— policy 模块必须导出的 decide 可调用对象的类型别名。
  3. load_policy(policy_id) —— 按文件名动态加载 ASSET/strategy/policies/
     {policy_id}.py,校验其导出契约,返回加载好的模块对象。

关于"无信号/数据不足时策略应该返回什么"这一点,spec 允许二选一,这里明确
选定:**返回空列表 [] 表示本周期无操作**(而不是显式的 hold Decision)。
理由:
  - 回测/实盘引擎对"这个分支这个周期没有产出任何决策"和"产出了一条
    action=hold 的决策"应该等价处理(都是"维持现状,不改变任何仓位"),
    用空列表更省事,不需要为每个 hold 决策都编出一句真实但没有信息量的
    thesis/falsifier 文本来凑够 MIN_THESIS_LEN。
  - 5个种子策略(aggressive/conservative/momentum/carry/diversified)在
    "没有可信信号"的分支里统一 `return []`,不构造 hold Decision。
  - 如果调用方(回测引擎/main.py 未来的 policy 执行器)确实需要显式记录
    "这个周期本策略选择不动",可以在拿到空列表后自行合成一条 hold 记录,
    那是调用方的职责,不是 policy 模块的职责。
"""
from __future__ import annotations

import importlib.util
import itertools
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from LOCKED.schemas import Decision, PerpPosition

POLICIES_DIR = Path(__file__).resolve().parent

# load_policy 每次都要生成互不冲突的模块名,不能直接用 policy_id 本身当
# sys.modules 的 key——内环会反复重写同一个 {policy_id}.py 文件、要求
# "重新加载最新版本",如果沿用旧的模块名,importlib 在某些 Python 实现下
# 可能命中旧的 .pyc/属性缓存。用一个单调递增计数器拼进内部模块名,保证
# 每次 load_policy() 调用都是一次真正从磁盘重新读取源码的全新 import。
_load_counter = itertools.count()

REQUIRED_ATTRS: dict[str, type | tuple[type, ...]] = {
    "decide": None,  # 可调用对象,类型检查用 callable() 而不是 isinstance
    "REQUIRED_HISTORY_BARS": int,
    "DESCRIPTION": str,
}


@dataclass(frozen=True)
class StrategyContext:
    """policy 模块 decide() 的唯一入参(§M7)。

    字段含义:
      ts             当前 bar 的时间戳,UTC 毫秒。policy 代码不得读墙钟,
                     任何"现在是什么时候"的判断都必须以这个字段为准
                     (与 LOCKED/reflector.py、ASSET/memory/engine.py 的
                     时间边界纪律同源)。
      positions      symbol -> 当前持仓(PerpPosition),没有持仓的symbol
                     不出现在dict里(而不是显式存一个None/空持仓占位)。
      snapshot       symbol -> {"last": float, ...}(与
                     DataPipeline.fetch_latest_snapshot 同构,至少含
                     "last" 键)。
      recent_bars    symbol -> 最近N根K线的DataFrame,列为
                     ["timestamp","open","high","low","close","volume"]
                     (与 LOCKED.data_pipeline.OHLCV_COLUMNS 同构),按
                     timestamp 升序排列,最后一行是最新的已收盘K线。
      memory_context 记忆检索结果的文本列表,回测场景下可以是空列表。
      recent_funding symbol -> 资金费率历史DataFrame(列 ["timestamp",
                     "funding_rate"],与 LOCKED.data_pipeline.
                     FUNDING_COLUMNS 同构),按timestamp升序排列,只包含
                     <= ctx.ts 的记录——这是与 recent_bars 同源的时间边界
                     纪律(回放/前向决策都不可窥见未来资金费率)。带默认值
                     {}(field(default_factory=dict)),因为这是M7上线后
                     新增的字段:所有既有构造点(测试/LOCKED/backtest_engine.py
                     的鸭子类型StrategyContext/ASSET/strategy/policy_trader.py
                     之前版本)不传它也不能报错。策略代码读取这个字段时应该
                     用 `getattr(ctx, 'recent_funding', {})` 而不是
                     `ctx.recent_funding` 直接访问——理由:
                       1. 兼容"喂给policy.decide()的ctx对象不是本类实例"
                          这种鸭子类型场景(LOCKED/backtest_engine.py自己
                          定义了一个同名但字段集不完全相同的StrategyContext,
                          module.decide作为strategy_fn被直接传进
                          BacktestEngine.run()——如果某次改动暂时让那份
                          ctx落后于本类的字段集,getattr兜底能让policy代码
                          优雅降级到"当作没有funding数据"，而不是直接
                          AttributeError崩掉整个回测)。
                       2. 即使本类实例本身已经有default_factory兜底、
                          "不传参也不炸"，policy模块仍然应该显式写
                          getattr(ctx, 'recent_funding', {}) 这种防御性
                          读法，把"字段可能不存在"这件事在调用点就地
                          文档化，而不是依赖调用方类型恰好是本类这一
                          隐式假设。carry_v1.py 就是这么写的，示例:
                              funding_map = getattr(ctx, "recent_funding", {})
                              fdf = funding_map.get(symbol)
      recent_spot    symbol -> 现货K线DataFrame(列同recent_bars:
                     ["timestamp","open","high","low","close","volume"]),
                     按timestamp升序排列,只包含<=ctx.ts的记录(与
                     recent_funding同源的时间边界纪律)。basis(现货溢价,
                     basis = perp_close/spot_close - 1)由策略代码自己算,
                     本字段只提供原料——ctx本身不预先算好这个衍生指标,
                     理由与"snapshot只给last价格、不预先算收益率"是同一种
                     设计原则。带默认值{}(field(default_factory=dict)),
                     是M9新增字段,既有构造点不传它也不能报错;策略代码
                     读取时同样应该用
                     `getattr(ctx, 'recent_spot', {})` 这种防御性写法
                     (理由与recent_funding完全一致,见上文)。
      recent_oi      symbol -> 持仓量(OI)历史DataFrame(列
                     ["timestamp","open_interest"],与
                     LOCKED.data_pipeline.OI_COLUMNS同构),按timestamp升序
                     排列,只包含<=ctx.ts的记录。带默认值{}
                     (field(default_factory=dict)),是M9新增字段,读取
                     方式与recent_spot/recent_funding同一套防御性约定。
    """

    ts: int
    positions: dict[str, PerpPosition]
    snapshot: dict[str, dict]
    recent_bars: dict[str, "pd.DataFrame"]
    memory_context: list[str] = field(default_factory=list)
    recent_funding: dict[str, "pd.DataFrame"] = field(default_factory=dict)
    recent_spot: dict[str, "pd.DataFrame"] = field(default_factory=dict)
    recent_oi: dict[str, "pd.DataFrame"] = field(default_factory=dict)


StrategyFn = Callable[[StrategyContext], list[Decision]]


class PolicyLoadError(Exception):
    """load_policy() 校验失败时抛出,消息里必须清楚指出缺了哪个契约项。"""


def load_policy(policy_id: str):
    """加载 ASSET/strategy/policies/{policy_id}.py,返回校验通过的模块对象。

    每次调用都从磁盘重新读取源码(不复用 sys.modules 里的旧模块),因为内环
    (agent 自我进化战术代码)会反复重写同一个 policy_id 对应的文件,调用方
    需要"总是拿到磁盘上最新的版本",而不是进程内第一次 import 时的缓存。

    校验:模块必须导出
      - decide: Callable[[StrategyContext], list[Decision]]
      - REQUIRED_HISTORY_BARS: int
      - DESCRIPTION: str
    缺任何一项都抛 PolicyLoadError,消息里点名缺的是哪个属性。
    """
    file_path = POLICIES_DIR / f"{policy_id}.py"
    if not file_path.exists():
        raise PolicyLoadError(
            f"policy file not found: {file_path} (policy_id={policy_id!r})"
        )

    internal_name = f"_alphaloop_policy_{policy_id}_{next(_load_counter)}"

    # 有意不用 importlib.util.spec_from_file_location()+exec_module() 的常规
    # 路径:那条路径会经过标准 SourceFileLoader,后者会在 __pycache__ 里按
    # (mtime, size) 校验并复用已编译的 .pyc——如果内环两次重写同一个文件
    # 发生得足够快(同一个文件系统时间戳粒度内)且新旧源码字节长度恰好相同,
    # 就可能读到"看起来没变"的陈旧字节码,产出旧版本的模块(已在本文件配套
    # 测试 test_load_policy_reloads_latest_version_from_disk 里实测复现过)。
    # 手动 read_text + compile + exec 完全绕开这层字节码缓存,每次调用都是
    # 一次真正从磁盘重新读取源码字符串、重新编译执行,消除这类"重新加载最新
    # 版本"失败的可能性。
    source = file_path.read_text(encoding="utf-8")
    try:
        code = compile(source, str(file_path), "exec")
    except SyntaxError as exc:
        raise PolicyLoadError(
            f"failed to compile policy module {file_path}: {exc!r}"
        ) from exc

    module = importlib.util.module_from_spec(
        importlib.util.spec_from_loader(internal_name, loader=None)
    )
    module.__file__ = str(file_path)
    # 加载失败(源码执行期抛异常等)时不留一个半初始化的模块在 sys.modules 里。
    sys.modules[internal_name] = module
    try:
        exec(code, module.__dict__)
    except Exception as exc:
        sys.modules.pop(internal_name, None)
        raise PolicyLoadError(
            f"failed to execute policy module {file_path}: {exc!r}"
        ) from exc

    missing: list[str] = []
    if not hasattr(module, "decide") or not callable(getattr(module, "decide")):
        missing.append("decide (callable)")
    if not hasattr(module, "REQUIRED_HISTORY_BARS") or not isinstance(
        getattr(module, "REQUIRED_HISTORY_BARS"), int
    ) or isinstance(getattr(module, "REQUIRED_HISTORY_BARS"), bool):
        missing.append("REQUIRED_HISTORY_BARS (int)")
    if not hasattr(module, "DESCRIPTION") or not isinstance(
        getattr(module, "DESCRIPTION"), str
    ):
        missing.append("DESCRIPTION (str)")

    if missing:
        sys.modules.pop(internal_name, None)
        raise PolicyLoadError(
            f"policy module {file_path} is missing required export(s): "
            f"{', '.join(missing)}"
        )

    return module


__all__ = [
    "StrategyContext",
    "StrategyFn",
    "load_policy",
    "PolicyLoadError",
    "POLICIES_DIR",
]
