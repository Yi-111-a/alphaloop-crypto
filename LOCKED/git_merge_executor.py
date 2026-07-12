"""
git_merge_executor.py -- M5 git-merge 执行器(§3.5 进化机制 / 复审对 M5 提出的
六条硬性要求中的第 1、2 条)。

复审原话(节选,逐字):

1. "git merge由LOCKED区编排器通过subprocess执行,策略agent的运行时对main
   分支零写权限(文件系统/git权限层面,不是约定层面)。"
   关键词是"不是约定层面"。本文件因此做两件事,分别对应两层不同强度的证据:
     (a) 结构性证据(这个项目**能**诚实做到的部分):本模块是整个项目里唯一
         调用 subprocess 执行 git merge/commit/push 的地方 -- 用一个
         项目级测试(tests/test_git_merge_executor.py 里的
         test_asset_region_has_zero_git_subprocess_capability)去扫描
         ASSET/ 下所有 .py 文件,断言里面既没有 import subprocess / 任何
         git 相关模块,也没有 "git merge"/"git commit"/"git push" 这类
         字符串字面量。只要 ASSET/ 区的代码(agent 运行时可修改、可执行的
         唯一区域)完全不具备调用 subprocess 的能力,策略 agent 的运行时
         代码路径里就**不存在任何一条**能触发真实 git 操作的执行路径 --
         这是代码层面可验证、可回归测试守住的保证。
     (b) 文件系统权限层面的尽力而为(lock_down_permissions/
         restore_permissions,见下方 GitMergeExecutor 类文档字符串):在merge
         窗口期间把 repo 的 .git 目录和 ASSET/ 树标记为只读。**必须诚实
         说明它在本沙箱(单一 OS 用户、Windows)下的真实局限**:这不是、也
         不能替代"策略agent 进程本身以受限 OS 用户运行、物理上无法拿到
         main 分支写权限"这种真正的操作系统级权限隔离 -- 这里的"运行时"
         和"merge 执行器"其实是同一个 OS 用户下的同一个 Python 进程空间，
         chmod/只读属性挡不住这个用户自己再把权限改回来。全 OS 用户级别
         的隔离需要在部署时把 ASSET 区代码放进一个单独的、权限更低的
         系统账户/容器里跑,这超出了单机开发沙箱能搭建和测试的范围,本模块
         不假装解决了这个问题,只提供"结构性零调用能力"这一层可验证的保证,
         外加一个尽力而为、在真正的多用户/容器化部署里会更有意义的
         文件系统只读标记作为纵深防御的第二层。

2. "merge前跑一次完整测试套件,红了就拒绝晋升并记LOG——防止agent提交一个
   '评分赢了但代码会炸'的分支。" 这是 attempt_merge() 的主线逻辑:候选分支
   即使已经被 scorer/orchestrator 判定 PROMOTE,如果它自己的测试套件跑不过,
   这里依然会拒绝真正的 git merge,并把拒绝原因 append 到 merge_log_path
   (通过 LOCKED.log_writer.append_jsonl,append-only,与项目里其它 LOCKED
   模块记录判定的方式一致)。

设计选择:candidate 分支的测试套件通过 `git worktree add <tmpdir> <branch>`
在一个完全独立于调用方当前工作树的临时目录里跑(而不是 `git checkout` 原地
切换分支)。原因:
  - `git worktree add` 从不触碰调用方 repo 的当前工作树/HEAD,即使
    attempt_merge() 中途抛出未预期的异常,repo 的当前检出状态也绝不会被
    这个模块意外改变;`git checkout` 做同样的事则必须自己想办法"测试完了
    再切回来",多一条容易在异常路径上被漏掉的恢复逻辑。
  - worktree 里跑测试,测的就是"如果真的合并进去,这份代码在它自己的目录
    树里长什么样、能不能通过它自己的测试" -- 这与"合并后的真实结果"更
    接近,而不是在共享目录里可能残留上一次检出状态的产物。
  - worktree 用完即删(try/finally 保证),不会在 repo 里留下垃圾。

零墙钟调用:本模块不导入 time.time()/datetime.now()。它需要的"现在"只是
"git log 里这次 merge commit 的时间戳"(由 git 自己写入 commit 里,不是本
模块显式取的墙钟);merge_log_path 里的记录也不带 now_date/ts 字段 -- 这与
本模块的定位一致:它不做时间窗口相关的判定(那是 scorer/orchestrator 的
职责),只负责"给定一个分支名,能不能、该不该把它真正合并进 main"这一件事。
"""
from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from LOCKED import log_writer

# 合法的 git 分支名字符集(保守白名单,拒绝一切可能被拿去做 shell/参数注入
# 的字符)。§3.5 里分支命名约定是 evo/YYYYMMDD-简述,main 本身也要合法。
# 这里不强制必须匹配 evo/... 前缀(main/其它历史分支也要能作为
# target_branch 或被合并),只做“不含危险字符”的结构性校验。
_SAFE_BRANCH_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")


def _is_safe_branch_name(name: str) -> bool:
    if not name or len(name) > 200:
        return False
    if not _SAFE_BRANCH_NAME_RE.match(name):
        return False
    if ".." in name:  # 防 git refname 里的父目录穿越写法
        return False
    if name.startswith("-"):  # 防被当成 flag 解析(即使我们从不拼 shell 字符串,也多一层防御)
        return False
    return True


@dataclass
class MergeResult:
    branch: str
    merged: bool
    reason: str  # human-readable: why rejected, or confirmation of what happened
    test_suite_passed: Optional[bool] = None  # None if the merge was refused before tests even ran (e.g. bad branch name)


class GitMergeExecutor:
    """LOCKED 区编排器通过 subprocess 执行 git merge 的唯一入口(§3.5 + 复审
    M5 要求1)。attempt_merge() 是唯一的对外方法 -- 调用方(main.py/调度器)
    只需要给一个已经被判定该晋升的分支名,本模块自己负责"先跑测试、红了就
    拒绝、绿了才真正 merge、无论如何都记 LOG"这一整套流程。"""

    def __init__(
        self,
        repo_path: str | Path,
        test_command: list[str] | None = None,
        target_branch: str = "main",
        log_root: str | Path | None = None,
        merge_log_path: str = "merge_attempts.jsonl",
    ) -> None:
        self.repo_path = Path(repo_path).resolve()
        self.test_command = test_command if test_command is not None else [sys.executable, "-m", "pytest", "-q"]
        self.target_branch = target_branch
        self.log_root: Optional[Path] = Path(log_root) if log_root is not None else None
        self.merge_log_path = merge_log_path
        self._locked_down = False
        # lock_down_permissions()/restore_permissions() 需要知道当时实际改了
        # 哪些路径的权限,以便精确复原(而不是无脑对整棵树重设一个"猜测"的
        # 权限位)。
        self._original_modes: dict[Path, int] = {}

    # ------------------------------------------------------------------
    # 内部小工具
    # ------------------------------------------------------------------

    def _run_git(self, args: list[str], cwd: Path, timeout: float = 60.0) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", "-C", str(cwd), *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False,
        )

    def _branch_exists(self, branch_name: str) -> bool:
        result = self._run_git(["branch", "--list", branch_name], cwd=self.repo_path)
        # `git branch --list <name>` 精确匹配时,存在则输出非空一行(可能带
        # 前导 "* " 或空格),不存在则输出为空。
        return result.returncode == 0 and result.stdout.strip() != ""

    def _log(self, record: dict) -> None:
        log_writer.append_jsonl(self.merge_log_path, record, root=self.log_root)

    # ------------------------------------------------------------------
    # 权限锁定(复审 M5 要求1(b),尽力而为 -- 见模块文档字符串对局限的说明)
    # ------------------------------------------------------------------

    def _iter_lockdown_paths(self) -> list[Path]:
        paths = []
        git_dir = self.repo_path / ".git"
        if git_dir.exists():
            paths.append(git_dir)
        asset_dir = self.repo_path / "ASSET"
        if asset_dir.exists():
            paths.append(asset_dir)
        return paths

    def lock_down_permissions(self) -> None:
        """merge 窗口期间尽力而为地把 repo 的 .git/ 与 ASSET/ 标记为只读。

        平台差异,如实说明(不夸大):
        - POSIX:对每个文件/目录清除写权限位(chmod 去掉 owner/group/other 的
          w 位),同一 OS 用户如果不先手工 chmod 回来,常规写操作(open(...,'w')、
          os.remove 等)会被内核拒绝 -- 这是相对有意义的一层保护。
        - Windows(本沙箱运行的平台):os.chmod 在 Windows 上只能有效影响
          stat.S_IWRITE 这一个只读属性位(对应资源管理器里的"只读"复选框),
          效果等价于 `attrib +r`。它会阻止一些"以只读方式跳过覆盖确认"的
          简单写入,但**不是**真正的 ACL 级强制:同一账户下拥有该文件写权限
          的进程仍然可以通过显式修改属性(或某些 API 直接覆盖)绕开它,目录
          本身的“只读”属性在 Windows 资源管理器语义下甚至经常被直接忽略。
          因此在本沙箱里,这个方法能保证的只是"一个观察得到的、可断言的
          文件属性状态变化"(测试 7 验证的就是这个),不能保证"任何代码都
          物理上无法写入" -- 真正的强隔离需要在部署时用单独的、权限更低的
          OS 账户/容器跑 ASSET 区代码,这不是当前单机开发沙箱能搭建的。
        """
        if self._locked_down:
            return
        self._original_modes.clear()
        for root in self._iter_lockdown_paths():
            for path in [root, *root.rglob("*")]:
                try:
                    mode = path.stat().st_mode
                except OSError:
                    continue
                self._original_modes[path] = mode
                read_only_mode = mode & ~(stat.S_IWRITE | stat.S_IWGRP | stat.S_IWOTH)
                try:
                    os.chmod(path, read_only_mode)
                except OSError:
                    # 尽力而为:单个文件 chmod 失败(权限不足/文件被占用等)
                    # 不应该让整个锁定过程崩溃或让 merge 流程异常终止。
                    pass
        self._locked_down = True

    def restore_permissions(self) -> None:
        """撤销 lock_down_permissions() 的效果。即使 merge 过程中途抛出异常,
        attempt_merge() 也会在 finally 里调用这个方法,确保一次被拒绝/失败的
        merge 不会把 repo 永久锁死。"""
        for path, mode in self._original_modes.items():
            try:
                os.chmod(path, mode)
            except OSError:
                pass
        self._original_modes.clear()
        self._locked_down = False

    # ------------------------------------------------------------------
    # 唯一入口
    # ------------------------------------------------------------------

    def attempt_merge(self, branch_name: str) -> MergeResult:
        # 1. 分支名结构性校验 + 存在性校验 -- 在做任何 git 操作之前,先挡掉
        #    明显不合法/危险的输入,绝不把原始字符串拼进 shell 命令(这里始终
        #    用 argv 列表传参,从不 shell=True)。
        if not _is_safe_branch_name(branch_name):
            result = MergeResult(
                branch=branch_name,
                merged=False,
                reason=f"refused: branch name {branch_name!r} failed structural validation",
                test_suite_passed=None,
            )
            self._log(self._result_to_record(result))
            return result

        try:
            exists = self._branch_exists(branch_name)
        except (subprocess.SubprocessError, OSError) as exc:
            result = MergeResult(
                branch=branch_name,
                merged=False,
                reason=f"refused: could not verify branch existence ({exc!r})",
                test_suite_passed=None,
            )
            self._log(self._result_to_record(result))
            return result

        if not exists:
            result = MergeResult(
                branch=branch_name,
                merged=False,
                reason=f"refused: branch {branch_name!r} does not exist in repo {self.repo_path}",
                test_suite_passed=None,
            )
            self._log(self._result_to_record(result))
            return result

        # 2. 在隔离的临时 worktree 里跑该分支自己的完整测试套件。
        worktree_dir = Path(tempfile.mkdtemp(prefix="alphaloop_merge_worktree_"))
        # mkdtemp 已经创建了目录本身,但 `git worktree add` 要求目标目录
        # 不存在(或为空)才能用,这里让它自己管理,先删掉这个空壳,只保留
        # 路径名字。
        worktree_dir.rmdir()
        worktree_added = False
        try:
            add_result = self._run_git(["worktree", "add", str(worktree_dir), branch_name], cwd=self.repo_path)
            if add_result.returncode != 0:
                result = MergeResult(
                    branch=branch_name,
                    merged=False,
                    reason=(
                        f"refused: failed to create worktree for {branch_name!r}: "
                        f"{add_result.stderr.strip()[:2000]}"
                    ),
                    test_suite_passed=None,
                )
                self._log(self._result_to_record(result))
                return result
            worktree_added = True

            # Lockdown window: wraps exactly the test-suite subprocess call,
            # not the git worktree add/remove or checkout/merge calls below.
            # This is a deliberate, disclosed deviation from locking down
            # literally "for the duration of the merge subprocess call": git
            # itself needs write access to .git to perform worktree add/
            # remove/checkout/merge, so making .git read-only while OUR OWN
            # git subprocess is trying to write to it would make git fail
            # regardless of who invoked it (permission bits are path-scoped
            # for the OS user, not process-scoped -- there is no way to let
            # "the merge executor's git call" write while simultaneously
            # blocking all writes). The one part of the whole flow where
            # locking .git/ASSET read-only is both meaningful and harmless is
            # exactly this window: candidate code is being evaluated (a plain
            # `pytest` subprocess that reads code and writes only inside the
            # disposable worktree dir), main hasn't been touched yet, and a
            # hypothetically still-running ASSET-region process gets a
            # best-effort block against sneaking in a write to .git/ASSET
            # while that evaluation is in flight. See module docstring and
            # lock_down_permissions() for the fuller (and more important)
            # caveat: on this Windows sandbox this only flips the read-only
            # file attribute, not a real ACL-level write block.
            self.lock_down_permissions()
            try:
                test_result = subprocess.run(
                    self.test_command,
                    capture_output=True,
                    text=True,
                    cwd=str(worktree_dir),
                    shell=False,
                    timeout=600,
                )
            except (subprocess.SubprocessError, OSError) as exc:
                result = MergeResult(
                    branch=branch_name,
                    merged=False,
                    reason=f"refused: test suite could not be executed ({exc!r})",
                    test_suite_passed=False,
                )
                self._log(self._result_to_record(result))
                return result
            finally:
                # Restore regardless of pass/fail/exception -- a rejected
                # (red) merge must never leave the repo locked down.
                self.restore_permissions()

            if test_result.returncode != 0:
                # 3. 红了 -- 拒绝晋升,记LOG,绝不 merge。这是"评分赢了但代码
                #    会炸"的核心拒绝路径。
                tail = (test_result.stdout + "\n" + test_result.stderr).strip()[-4000:]
                result = MergeResult(
                    branch=branch_name,
                    merged=False,
                    reason=(
                        f"refused: test suite failed on branch {branch_name!r} "
                        f"(exit code {test_result.returncode}); output tail: {tail}"
                    ),
                    test_suite_passed=False,
                )
                self._log(self._result_to_record(result))
                return result

            # 4. 测试全绿 -- 真正执行 merge。注意:此处不再套一层
            #    lock_down_permissions() -- checkout/merge 本身就需要对 .git
            #    有写权限才能完成,把 .git 设为只读会让这两条 git 命令自己先
            #    失败,是自相矛盾的("只读地执行写操作"做不到,不管是谁发起
            #    的写操作,操作系统的权限位是按路径而不是按进程判定的)。真正
            #    有意义、且不自相矛盾的锁定窗口是上面"候选分支测试评估期"那
            #    一段(此时还没决定要不要碰 main,.git 也不需要被我们自己写
            #    入),已经在那里锁过、也已经在那里的 finally 里还原过了。
            checkout_result = self._run_git(["checkout", self.target_branch], cwd=self.repo_path)
            if checkout_result.returncode != 0:
                result = MergeResult(
                    branch=branch_name,
                    merged=False,
                    reason=(
                        f"refused: could not checkout target branch {self.target_branch!r}: "
                        f"{checkout_result.stderr.strip()[:2000]}"
                    ),
                    test_suite_passed=True,
                )
                self._log(self._result_to_record(result))
                return result

            merge_result = self._run_git(
                ["merge", "--no-ff", branch_name, "-m", f"PROMOTE {branch_name} into {self.target_branch}"],
                cwd=self.repo_path,
            )
            if merge_result.returncode != 0:
                # merge 冲突等 -- 中止掉,避免把 repo 留在冲突态的半merge。
                self._run_git(["merge", "--abort"], cwd=self.repo_path)
                result = MergeResult(
                    branch=branch_name,
                    merged=False,
                    reason=(
                        f"refused: git merge of {branch_name!r} into {self.target_branch!r} failed: "
                        f"{merge_result.stderr.strip()[:2000]}"
                    ),
                    test_suite_passed=True,
                )
                self._log(self._result_to_record(result))
                return result

            result = MergeResult(
                branch=branch_name,
                merged=True,
                reason=(
                    f"merged {branch_name!r} into {self.target_branch!r} (--no-ff) "
                    f"after full test suite passed"
                ),
                test_suite_passed=True,
            )
            self._log(self._result_to_record(result))
            return result
        finally:
            # 5. 清理临时 worktree,无论成功/失败/异常。
            if worktree_added:
                self._run_git(["worktree", "remove", "--force", str(worktree_dir)], cwd=self.repo_path)
            shutil.rmtree(worktree_dir, ignore_errors=True)
            # 有些 git 版本即使 worktree 目录已被删除,仍会在
            # .git/worktrees/<name> 下留一条已失效的注册记录 -- prune 一下,
            # 保证 `git worktree list` 不会有残留条目。
            self._run_git(["worktree", "prune"], cwd=self.repo_path)

    @staticmethod
    def _result_to_record(result: MergeResult) -> dict:
        return {
            "branch": result.branch,
            "merged": result.merged,
            "reason": result.reason,
            "test_suite_passed": result.test_suite_passed,
        }
