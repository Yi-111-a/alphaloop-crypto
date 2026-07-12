"""
tests/test_git_merge_executor.py -- M5 git-merge 执行器验收测试。

覆盖复审对 M5 提出的两条硬性要求(见 LOCKED/git_merge_executor.py 模块文档
字符串对这两条要求的逐字引用与实现说明):
  1. "git merge由LOCKED区编排器通过subprocess执行,策略agent的运行时对main
     分支零写权限" -- 结构性零调用能力测试(test 6)是这个项目诚实能给出的
     那一层保证;权限锁定冒烟测试(test 7)覆盖尽力而为的文件系统层面纵深
     防御,并如实验证/记录它在本 Windows 沙箱下的真实局限。
  2. "merge前跑一次完整测试套件,红了就拒绝晋升并记LOG" -- test 2 是这条
     要求里最核心、优先级最高的验收场景:一个"棘轮判定会赢但代码会炸"的
     分支必须被正确拒绝,且 main 分支必须保持不变。

所有涉及 git 操作的测试都在 tmp_path 下构造真实的、一次性的 git 仓库
(subprocess `git init`/`git commit`/`git checkout -b` 等),完全不触碰
alphaloop 项目自身的仓库状态 -- 与 tests/test_evolution_orchestrator.py
里"在 tmp_path 下构造真实 Simulator 做跨分支账本隔离测试"是同一套纪律。
"""
from __future__ import annotations

import ast
import os
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from LOCKED import log_writer
from LOCKED.git_merge_executor import GitMergeExecutor, MergeResult

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSET_DIR = PROJECT_ROOT / "ASSET"


# ---------------------------------------------------------------------------
# 小工具:在 tmp_path 下构造真实的一次性 git 仓库
# ---------------------------------------------------------------------------


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    result = subprocess.run(
        ["git", "-C", str(cwd), *args],
        capture_output=True,
        text=True,
        shell=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git -C {cwd} {' '.join(args)} failed:\n{result.stdout}\n{result.stderr}")
    return result


def _write_pkg(repo: Path, compute_return: int, expected_in_test: int) -> None:
    (repo / "pkg.py").write_text(f"def compute():\n    return {compute_return}\n", encoding="utf-8")
    (repo / "test_pkg.py").write_text(
        "from pkg import compute\n\n\ndef test_compute():\n"
        f"    assert compute() == {expected_in_test}\n",
        encoding="utf-8",
    )


@pytest.fixture
def log_root(tmp_path):
    d = tmp_path / "LOG"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def repo_with_branches(tmp_path):
    """main(通过) + evo/winner(修改后测试依然通过) + evo/loser(修改后自身
    测试套件会失败)的一次性真实仓库。"""
    repo = tmp_path / "candidate_repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test User")

    _write_pkg(repo, compute_return=2, expected_in_test=2)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "initial commit on main")

    _git(repo, "checkout", "-b", "evo/winner")
    _write_pkg(repo, compute_return=3, expected_in_test=3)  # still passes
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "evo/winner: bump compute() to 3, tests still pass")
    _git(repo, "checkout", "main")

    _git(repo, "checkout", "-b", "evo/loser")
    _write_pkg(repo, compute_return=5, expected_in_test=999)  # deliberately broken
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "evo/loser: broken -- its own test suite fails")
    _git(repo, "checkout", "main")

    return repo


def _make_executor(repo: Path, log_root: Path) -> GitMergeExecutor:
    return GitMergeExecutor(repo_path=repo, log_root=log_root, merge_log_path="merge_attempts.jsonl")


# ---------------------------------------------------------------------------
# 1. Happy path
# ---------------------------------------------------------------------------


def test_happy_path_merges_passing_branch(repo_with_branches, log_root):
    executor = _make_executor(repo_with_branches, log_root)
    result = executor.attempt_merge("evo/winner")

    assert isinstance(result, MergeResult)
    assert result.merged is True
    assert result.test_suite_passed is True
    assert result.branch == "evo/winner"

    # main 分支现在真的包含了 evo/winner 的改动。
    head_branch = _git(repo_with_branches, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    assert head_branch == "main"

    pkg_content = (repo_with_branches / "pkg.py").read_text(encoding="utf-8")
    assert "return 3" in pkg_content

    log_output = _git(repo_with_branches, "log", "main", "--oneline").stdout
    assert "evo/winner" in log_output or "bump compute" in log_output


# ---------------------------------------------------------------------------
# 2. THE CORE REJECTION TEST -- 评分会赢但代码会炸的分支必须被拒绝
# ---------------------------------------------------------------------------


def test_failing_test_suite_rejects_merge_and_leaves_main_untouched(repo_with_branches, log_root):
    executor = _make_executor(repo_with_branches, log_root)

    main_pkg_before = (repo_with_branches / "pkg.py").read_text(encoding="utf-8")
    main_log_before = _git(repo_with_branches, "log", "main", "--oneline").stdout

    result = executor.attempt_merge("evo/loser")

    assert result.merged is False
    assert result.test_suite_passed is False
    assert "evo/loser" in result.reason or "test" in result.reason.lower()

    # main 完全没有变化:working tree 内容不变,main 分支的提交历史不变。
    main_pkg_after = (repo_with_branches / "pkg.py").read_text(encoding="utf-8")
    assert main_pkg_after == main_pkg_before
    assert "return 5" not in main_pkg_after

    main_log_after = _git(repo_with_branches, "log", "main", "--oneline").stdout
    assert main_log_after == main_log_before
    assert "broken" not in main_log_after

    head_branch = _git(repo_with_branches, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    assert head_branch == "main"


# ---------------------------------------------------------------------------
# 3. Invalid / nonexistent branch name
# ---------------------------------------------------------------------------


def test_nonexistent_branch_refused_cleanly(repo_with_branches, log_root):
    executor = _make_executor(repo_with_branches, log_root)
    result = executor.attempt_merge("does/not/exist")

    assert result.merged is False
    assert result.test_suite_passed is None
    assert "does/not/exist" in result.reason


def test_structurally_unsafe_branch_name_refused_without_exception(repo_with_branches, log_root):
    executor = _make_executor(repo_with_branches, log_root)
    # 含有 shell 元字符 / 危险片段的分支名,绝不能让原始字符串路径到达任何
    # subprocess 调用并引发异常或注入。
    for bad_name in ["; rm -rf /", "--upload-pack=evil", "../../etc/passwd", ""]:
        result = executor.attempt_merge(bad_name)
        assert result.merged is False
        assert result.test_suite_passed is None


# ---------------------------------------------------------------------------
# 4. Every outcome logged, append-only, accumulating across calls
# ---------------------------------------------------------------------------


def test_every_outcome_is_logged_append_only(repo_with_branches, log_root):
    executor = _make_executor(repo_with_branches, log_root)

    r1 = executor.attempt_merge("does/not/exist")  # invalid
    r2 = executor.attempt_merge("evo/loser")  # rejected by tests
    r3 = executor.attempt_merge("evo/winner")  # merged

    records = log_writer.read_jsonl("merge_attempts.jsonl", root=log_root)
    assert len(records) == 3

    by_branch = {rec["branch"]: rec for rec in records}
    assert by_branch["does/not/exist"]["merged"] is False
    assert by_branch["does/not/exist"]["test_suite_passed"] is None

    assert by_branch["evo/loser"]["merged"] is False
    assert by_branch["evo/loser"]["test_suite_passed"] is False

    assert by_branch["evo/winner"]["merged"] is True
    assert by_branch["evo/winner"]["test_suite_passed"] is True

    # 结果对象与落盘记录一致。
    assert r1.merged is False and r2.merged is False and r3.merged is True

    # 再打一次,确认是追加而不是覆盖。
    executor2 = _make_executor(repo_with_branches, log_root)
    executor2.attempt_merge("does/not/exist")
    records_after = log_writer.read_jsonl("merge_attempts.jsonl", root=log_root)
    assert len(records_after) == 4


# ---------------------------------------------------------------------------
# 5. Worktree cleanup -- no leftovers after success OR failure
# ---------------------------------------------------------------------------


def test_worktree_cleaned_up_after_success_and_failure(repo_with_branches, log_root):
    tmp_root = Path(tempfile.gettempdir())
    leftovers_before = set(tmp_root.glob("alphaloop_merge_worktree_*"))

    executor = _make_executor(repo_with_branches, log_root)
    executor.attempt_merge("evo/loser")  # failure path
    executor2 = _make_executor(repo_with_branches, log_root)
    executor2.attempt_merge("does/not/exist")  # never even creates a worktree

    # need a fresh winner-equivalent branch for a second executor call to
    # exercise the success path without re-merging an already-merged branch
    _git(repo_with_branches, "checkout", "-b", "evo/winner2", "evo/winner")
    _git(repo_with_branches, "checkout", "main")
    executor3 = _make_executor(repo_with_branches, log_root)
    executor3.attempt_merge("evo/winner2")  # success path

    leftovers_after = set(tmp_root.glob("alphaloop_merge_worktree_*"))
    assert leftovers_after == leftovers_before, f"leaked temp worktree dirs: {leftovers_after - leftovers_before}"

    worktree_list = _git(repo_with_branches, "worktree", "list").stdout
    # 只应该剩下主工作树这一条(porcelain 输出每个 worktree 一行)。
    assert len(worktree_list.strip().splitlines()) == 1


# ---------------------------------------------------------------------------
# 6. Structural enforcement -- ASSET has zero git/subprocess capability
# ---------------------------------------------------------------------------

_GIT_RELATED_MODULE_NAMES = {"subprocess", "git", "pygit2", "dulwich"}
_FORBIDDEN_GIT_LITERALS = ("git merge", "git commit", "git push")


def _imported_module_roots(tree: ast.AST) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                roots.add(node.module.split(".")[0])
    return roots


def _string_literals(tree: ast.AST) -> list[str]:
    literals = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            literals.append(node.value)
    return literals


def test_asset_region_has_zero_git_subprocess_capability():
    """M5 复审要求1(不是约定层面):ASSET/ 是唯一 agent 运行时可修改/可执行
    的区域。这个测试断言里面没有任何一个 .py 文件 import subprocess 或任何
    git 相关模块,也没有任何字符串字面量匹配 'git merge'/'git commit'/
    'git push' -- 也就是说,agent 的运行时代码路径里不存在任何一条能触发
    真实 git 操作的执行路径。这是本项目在单一 OS 用户开发沙箱下能给出的、
    可回归测试守住的结构性保证 -- 不是文档字符串里的一句"agent不应该调用
    这个"约定,而是"根本没有代码能这么做"。"""
    assert ASSET_DIR.exists(), "ASSET/ directory should exist in this project"

    offenders: dict[str, list[str]] = {}
    py_files = [p for p in ASSET_DIR.rglob("*.py") if "__pycache__" not in p.parts]
    assert py_files, "expected at least some .py files under ASSET/ to scan"

    for path in py_files:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))

        hits: list[str] = []
        forbidden_imports = _imported_module_roots(tree) & _GIT_RELATED_MODULE_NAMES
        for name in forbidden_imports:
            hits.append(f"import of forbidden module {name!r}")

        for literal in _string_literals(tree):
            for forbidden in _FORBIDDEN_GIT_LITERALS:
                if forbidden in literal:
                    hits.append(f"string literal containing {forbidden!r}")

        # os.system/os.popen 是绕开 subprocess 模块本身、但同样能执行任意
        # shell 命令的旁路,顺带查一下(比要求更严格一点,属于额外的纵深检查)。
        if "os.system" in source or "os.popen" in source:
            hits.append("use of os.system/os.popen")

        if hits:
            offenders[str(path.relative_to(PROJECT_ROOT))] = hits

    assert not offenders, (
        "ASSET/ must have zero structural capability to invoke git/subprocess -- "
        f"found: {offenders}"
    )


def test_git_merge_executor_is_the_only_subprocess_git_caller_in_the_project():
    """更进一步的项目级证据(超出要求的最小范围,作为额外交叉验证):扫描
    LOCKED/ 下除 git_merge_executor.py 自身之外的所有模块,确认没有第二个
    地方也在调用 subprocess 执行 git 操作。"""
    locked_dir = PROJECT_ROOT / "LOCKED"
    offenders: dict[str, list[str]] = {}
    for path in locked_dir.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        if path.name == "git_merge_executor.py":
            continue
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        forbidden_imports = _imported_module_roots(tree) & _GIT_RELATED_MODULE_NAMES
        if forbidden_imports:
            offenders[str(path.relative_to(PROJECT_ROOT))] = sorted(forbidden_imports)
    assert not offenders, f"unexpected subprocess/git-capable modules outside git_merge_executor.py: {offenders}"


# ---------------------------------------------------------------------------
# 7. Permission lockdown smoke test
# ---------------------------------------------------------------------------


@pytest.fixture
def lockdown_repo(tmp_path):
    """一个带 .git 和 ASSET/ 子目录的最小仓库,专门用于权限锁定测试(不需要
    真实提交历史,只需要真实存在的路径供 chmod 操作)。"""
    repo = tmp_path / "lockdown_repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test User")
    (repo / "README.md").write_text("placeholder\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "initial")

    asset_dir = repo / "ASSET"
    asset_dir.mkdir()
    (asset_dir / "strategy.py").write_text("# placeholder strategy file\n", encoding="utf-8")

    return repo


def test_lock_down_and_restore_permissions_observable_change(lockdown_repo, log_root):
    """在这个 Windows 沙箱上,lock_down_permissions() 真正能保证的东西只是:
    .git/ 与 ASSET/ 下每个文件的只读属性位(stat.S_IWRITE / `attrib +r` 语义)
    被翻转,os.access(path, os.W_OK) 会因此从 True 变成 False;
    restore_permissions() 把它翻回去。这不是 POSIX 权限位那种内核强制的
    多用户 ACL,只是一个可观察、可断言的文件属性状态 -- 详见
    LOCKED/git_merge_executor.py 里 lock_down_permissions() 的文档字符串。
    """
    executor = _make_executor(lockdown_repo, log_root)

    git_config = lockdown_repo / ".git" / "config"
    asset_file = lockdown_repo / "ASSET" / "strategy.py"

    assert os.access(git_config, os.W_OK) is True
    assert os.access(asset_file, os.W_OK) is True

    executor.lock_down_permissions()
    try:
        assert os.access(git_config, os.W_OK) is False
        assert os.access(asset_file, os.W_OK) is False
        # 直观校验:read-only 属性位确实被清掉了写权限位。
        assert not (git_config.stat().st_mode & stat.S_IWRITE)
        assert not (asset_file.stat().st_mode & stat.S_IWRITE)
    finally:
        executor.restore_permissions()

    assert os.access(git_config, os.W_OK) is True
    assert os.access(asset_file, os.W_OK) is True


def test_attempt_merge_restores_permissions_even_when_tests_fail(repo_with_branches, log_root):
    """attempt_merge() 内部在"候选分支测试评估期"这一段套了
    lock_down_permissions()/restore_permissions(),用 try/finally 保证不管
    测试红不红、甚至测试子进程本身抛异常,都会在返回前把权限还原 -- 一次被
    拒绝的 merge 绝不能把 repo 永久锁死。"""
    executor = _make_executor(repo_with_branches, log_root)
    git_config = repo_with_branches / ".git" / "config"

    assert os.access(git_config, os.W_OK) is True

    result = executor.attempt_merge("evo/loser")  # tests fail -> rejected
    assert result.merged is False
    assert result.test_suite_passed is False

    # 权限必须已经被还原,不残留锁定状态。
    assert os.access(git_config, os.W_OK) is True
    assert executor._locked_down is False
