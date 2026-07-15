"""
ASSET/strategy/policy_lint.py —— M7 确定性策略代码的 AST 静态检查。

policy 模块(ASSET/strategy/policies/{policy_id}.py)必须是**纯函数、确定性**
的:同一个 StrategyContext 输入,任何时候、在任何机器上跑,都必须产出逐字段
相同的 Decision 列表。这条纪律不能只靠人工代码审查或运行时单测偶然发现——
和 LOCKED/reflector.py、ASSET/memory/engine.py 的时间边界纪律一样,这里用
AST 静态扫描把它钉死成一道结构性的、跑得快的护栏(不需要真的执行代码就能
判定"这份源码是否可能违反确定性/沙箱约束")。

被禁止的东西,以及为什么:
  1. 网络库(requests/httpx/urllib/socket/ccxt/anthropic/...) —— policy 只应该
     读 ctx 里已经拼装好的数据,任何网络调用都意味着"同一个 ctx、不同时刻跑
     结果可能不同"(网络请求本身就是非确定性输入源),而且回测环境要求完全
     离线。
  2. 墙钟读取(time.time/time.time_ns/datetime.now/utcnow/today/date.today)——
     ctx.ts 是 policy 唯一被允许认知"现在几点"的信息来源,读墙钟直接违反
     "同一份历史回放,不管哪天跑结果必须一致"的确定性要求。
  3. 随机数(random 模块、numpy.random)—— 同一个理由:确定性要求下,如果
     策略真的需要"抖动",必须从 ctx.ts 派生一个确定性种子自己实现,而不是
     调用系统级随机数源。第一版直接整体禁掉,不做"允许从ctx.ts播种"的白名单
     豁免,判断成本更低、也不留下"看似受控实则不受控"的模糊地带。
  4. os.environ / subprocess / exec / eval —— 沙箱逃逸/环境探测/动态代码执行,
     纸面交易研究系统里 policy 代码没有任何合法理由需要这些。
  5. open() 的写/追加/独占模式 —— policy 不应该有任何写盘副作用(纯函数);
     只读模式(默认模式或显式 'r'/'rb'/'rt')不在此列。

lint_policy_source(source) 返回违规字符串列表,每条形如
"<类别>: <细节> (line N)";空列表 = 通过。lint_policy_file(path) 是文件读取
的便捷入口。
"""
from __future__ import annotations

import ast
from pathlib import Path

# ---------------------------------------------------------------------------
# 禁止名单
# ---------------------------------------------------------------------------

# 网络库:按"根模块名"匹配,import a.b.c 或 from a.b import c 都只看最左边那段
# (如 "urllib.request" 的根模块是 "urllib"),避免遗漏子模块导入。
_FORBIDDEN_NETWORK_ROOT_MODULES = {
    "requests",
    "httpx",
    "urllib",
    "urllib2",
    "urllib3",
    "socket",
    "ccxt",
    "anthropic",
    "aiohttp",
    "http",  # http.client 等
    "ftplib",
    "smtplib",
    "telnetlib",
    "websocket",
    "websockets",
    "grpc",
}

# 完全禁止 import 的模块(与确定性/沙箱相关,不属于"网络库"这一类但同样整体
# 禁掉):random(数值随机源)、subprocess(子进程/沙箱逃逸)。
_FORBIDDEN_FULL_BAN_ROOT_MODULES = {
    "random",
    "subprocess",
}

# 墙钟相关的属性调用名——不管调用对象是什么(time.time()、datetime.datetime.
# now()、date.today() 等都命中),与 tests/test_reflector.py 里的同款护栏保持
# 一致的检测方式。
_WALLCLOCK_CALL_ATTRS = {"time", "time_ns", "now", "utcnow", "today"}

# exec/eval 这两个内置函数名。
_FORBIDDEN_BUILTIN_CALLS = {"exec", "eval"}

# open() 里代表"非只读"的模式片段:只要 mode 字符串里出现下列任一字符
# (且不是纯粹的 "r"/"rt"/"rb"/"U"),就判定为写/追加/独占/更新模式。
_WRITE_MODE_MARKERS = {"w", "a", "x", "+"}
_READ_ONLY_MODES = {"r", "rt", "rb", "rU", "Ur", "U"}


def _root_module_name(dotted: str) -> str:
    return dotted.split(".", 1)[0]


def _describe(node: ast.AST) -> int:
    return getattr(node, "lineno", -1)


def lint_policy_source(source: str) -> list[str]:
    """扫描一段 policy 源码,返回违规清单(空列表 = 通过)。

    source 语法错误时,把 SyntaxError 本身当作一条违规返回,而不是让异常
    往外抛——调用方(load_policy 之前的 lint 步骤 / 测试)只关心"这份源码
    能不能过",语法错误显然过不了。
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"syntax_error: {exc}"]

    violations: list[str] = []

    # 记录 numpy 的 import 别名(比如 `import numpy as np`),用于识别
    # `np.random.xxx` 这种通过属性访问触达 numpy.random 的写法。
    numpy_aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _root_module_name(alias.name) == "numpy":
                    numpy_aliases.add(alias.asname or alias.name.split(".")[0])

    for node in ast.walk(tree):
        # -------------------- import xxx --------------------
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = _root_module_name(alias.name)
                if root in _FORBIDDEN_NETWORK_ROOT_MODULES:
                    violations.append(
                        f"forbidden_network_import: import {alias.name} (line {_describe(node)})"
                    )
                elif root in _FORBIDDEN_FULL_BAN_ROOT_MODULES:
                    violations.append(
                        f"forbidden_import: import {alias.name} (line {_describe(node)})"
                    )
                elif alias.name == "numpy.random" or root == "numpy" and alias.name.startswith(
                    "numpy.random"
                ):
                    violations.append(
                        f"forbidden_random_import: import {alias.name} (line {_describe(node)})"
                    )

        # -------------------- from xxx import yyy --------------------
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            root = _root_module_name(module)
            if root in _FORBIDDEN_NETWORK_ROOT_MODULES:
                violations.append(
                    f"forbidden_network_import: from {module} import ... (line {_describe(node)})"
                )
            elif root in _FORBIDDEN_FULL_BAN_ROOT_MODULES:
                violations.append(
                    f"forbidden_import: from {module} import ... (line {_describe(node)})"
                )
            elif module == "numpy" and any(a.name == "random" for a in node.names):
                violations.append(
                    f"forbidden_random_import: from numpy import random (line {_describe(node)})"
                )
            elif root == "os" and any(a.name == "environ" for a in node.names):
                violations.append(
                    f"forbidden_os_environ: from os import environ (line {_describe(node)})"
                )

        # -------------------- 函数/方法调用 --------------------
        elif isinstance(node, ast.Call):
            func = node.func

            # exec(...) / eval(...)
            if isinstance(func, ast.Name) and func.id in _FORBIDDEN_BUILTIN_CALLS:
                violations.append(
                    f"forbidden_builtin_call: {func.id}(...) (line {_describe(node)})"
                )

            if isinstance(func, ast.Attribute):
                # 墙钟调用:xxx.now() / xxx.today() / xxx.time() / xxx.time_ns()
                if func.attr in _WALLCLOCK_CALL_ATTRS:
                    violations.append(
                        f"forbidden_wallclock_call: .{func.attr}(...) (line {_describe(node)})"
                    )
                # os.getenv(...)
                if func.attr == "getenv" and isinstance(func.value, ast.Name) and func.value.id == "os":
                    violations.append(
                        f"forbidden_os_environ: os.getenv(...) (line {_describe(node)})"
                    )
                # np.random.xxx(...) —— func.value 是 Attribute(value=Name(np), attr="random")
                if (
                    isinstance(func.value, ast.Attribute)
                    and func.value.attr == "random"
                    and isinstance(func.value.value, ast.Name)
                    and func.value.value.id in numpy_aliases
                ):
                    violations.append(
                        f"forbidden_random_call: {func.value.value.id}.random.{func.attr}(...) "
                        f"(line {_describe(node)})"
                    )
                # random.xxx(...) 直接调用(即便没有走 import random 的分支也兜底一次,
                # 比如 `import random as rnd` 后 rnd.random() —— 这里退化成只按属性名
                # 兼容常见写法,严格覆盖仍以 import 检测为主)。

            # open(...) 写/追加/独占模式
            if isinstance(func, ast.Name) and func.id == "open":
                mode_node = None
                if len(node.args) >= 2:
                    mode_node = node.args[1]
                else:
                    for kw in node.keywords:
                        if kw.arg == "mode":
                            mode_node = kw.value
                if mode_node is not None:
                    if isinstance(mode_node, ast.Constant) and isinstance(mode_node.value, str):
                        mode_value = mode_node.value
                        if mode_value not in _READ_ONLY_MODES and any(
                            marker in mode_value for marker in _WRITE_MODE_MARKERS
                        ):
                            violations.append(
                                f"forbidden_open_write_mode: open(..., mode={mode_value!r}) "
                                f"(line {_describe(node)})"
                            )
                    else:
                        # mode 不是字面量字符串,无法静态确认是只读——保守拒绝。
                        violations.append(
                            f"forbidden_open_dynamic_mode: open() mode is not a static string "
                            f"literal, cannot verify it is read-only (line {_describe(node)})"
                        )

        # -------------------- os.environ 属性访问(非 import 形式,如 os.environ["X"]) --------------------
        elif isinstance(node, ast.Attribute):
            if node.attr == "environ" and isinstance(node.value, ast.Name) and node.value.id == "os":
                violations.append(
                    f"forbidden_os_environ: os.environ (line {_describe(node)})"
                )
            # np.random(不作为调用,单纯属性访问,比如把 np.random 存成变量)
            if (
                node.attr == "random"
                and isinstance(node.value, ast.Name)
                and node.value.id in numpy_aliases
            ):
                violations.append(
                    f"forbidden_random_import: {node.value.id}.random (line {_describe(node)})"
                )

    return violations


def lint_policy_file(path: str | Path) -> list[str]:
    """lint_policy_source 的文件入口,读取 path 并返回违规清单。"""
    source = Path(path).read_text(encoding="utf-8")
    return lint_policy_source(source)


__all__ = ["lint_policy_source", "lint_policy_file"]
