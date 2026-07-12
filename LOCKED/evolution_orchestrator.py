"""
evolution_orchestrator.py -- M4 进化编排器(§3.5 进化机制 / §6 M4 验收标准)。

铁律相关(§0 + 复审对 M4 提出的五条硬性要求):

1. 窗口对齐:本模块自己不做任何净值切片计算(那是 scorer.ratchet_score 的
   职责),但它是唯一负责"记住每个候选分支自己的创建日期,并在裁决时把它
   原样喂给 scorer"的一环。judge() 里传给 scorer.ratchet_score 的
   branch_created_dates 字典严格来自 BranchMeta.created_date(register_branch()
   时由调用方一次性写死,judge() 不接受任何覆盖它的参数)——没有任何代码
   路径能让某次裁决使用一个"不是该分支真实创建日期"的窗口起点。

2. 判定权归属:register_branch() 是唯一的 agent-facing 入口,只负责"准入"，
   不产生、不参与、也无法影响任何 Verdict。judge() 是唯一的裁决入口，其签名
   里没有任何参数可以让调用方直接指定/覆盖某个分支的判定结果——Verdict 100%
   由 (branch_navs, benchmark_navs, branch_dead_flags) 经
   scorer.ratchet_score() + 本模块的清算强制FAIL短路逻辑推导得出。每条
   Verdict(含清算强制FAIL)都通过 LOCKED.log_writer.append_jsonl 落盘，
   append-only，不提供、也不会提供任何"改写历史判定"的接口。

3. 账本隔离:本模块不触碰 Simulator 的账本(positions/wallet_balance/sqlite)。
   一个分支只是这里的一条 BranchMeta 记录 + scorer 消费的一段 NAV 序列，互相
   之间没有共享可变状态——账本级别的隔离测试见
   tests/test_evolution_orchestrator.py 里直接构造两个真实 Simulator 实例的
   那个测试，不在本文件的职责范围内。

4. 最小优势门槛:本模块从不给 scorer.ratchet_score 传一个比
   config.evolution.min_promote_edge_pct 更低的 min_promote_edge_pct，也不
   提供任何绕过它的独立 PROMOTE 路径——judge() 里对 PROMOTE/ARCHIVE/FAIL 的
   处理完全是"照抄 scorer 的判定结果"，没有自己的第二套阈值逻辑。

5. 晋升后追踪:promotion_records() 维护"曾经被 PROMOTE 过的分支"的权威列表
   (PromotionRecord(branch, created_date, promoted_date))，可直接喂给
   scorer.monthly_report(promotions=...)。

零墙钟调用:本文件不导入/调用 time.time()/datetime.now() 等接口——"现在是
哪天"(now_date)、"分支是哪天创建的"(created_date)一律由调用方显式传入，
保证同一段历史可以在任何时候重放出完全相同的裁决结果。

M5 崩溃恢复(硬要求1):分支登记表(_branches/_promotion_order/
_current_main_branch)不是一份独立维护、需要另外原子落盘的可变快照——那样
会重新引入"快照和日志谁先写、崩溃在中间怎么办"这个 funding.jsonl 曾经踩过
的坑。这里选择更简单也更稳的做法:_branches 等内存状态是**完全可以从两份
append-only 日志重放出来的派生视图**:
  1. branch_registrations.jsonl —— register_branch() 每次成功登记都追加一条
     (branch, created_date) 记录,这是"这个分支何时被提交"的权威事实来源。
  2. verdicts_log_path(ratchet_verdicts.jsonl) —— judge() 已经会把每条
     Verdict 落这里,这是"这个分支后来被怎么判"的权威事实来源。
构造函数在 __init__ 里把这两份日志按顺序重放一遍,重新推导出当前应该有哪些
active/promoted/archived/failed 分支——不管进程在任何时间点被杀掉重启，
下次启动时只要这两份日志文件本身完好(它们是普通的 append-only 写入，没有
"半条记录"的中间态问题),内存状态就能被精确重建,不存在"日志和快照不一致"
这类问题,因为压根没有独立快照。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

from LOCKED import log_writer
from LOCKED.schemas import PromotionRecord, Verdict

BranchStatus = Literal["active", "promoted", "archived", "failed"]

DEFAULT_MAX_CONCURRENT_BRANCHES = 3


@dataclass
class BranchMeta:
    name: str
    created_date: str  # ISO日期, scorer.ratchet_score 会用它作为该分支的评分窗口起点
    status: BranchStatus
    promoted_date: Optional[str] = None


class EvolutionOrchestrator:
    """M4 棘轮进化的编排器:准入(register_branch)与裁决(judge)职责分离，
    judge() 是唯一能产出 Verdict 的入口，且是 (NAV数据, branch_dead标志) 的
    纯确定性函数。"""

    def __init__(
        self,
        config: dict,
        scorer,
        log_root: str | Path | None = None,
        verdicts_log_path: str = "ratchet_verdicts.jsonl",
        registrations_log_path: str = "branch_registrations.jsonl",
    ) -> None:
        self.config = config
        self.scorer = scorer
        self.log_root: Optional[Path] = Path(log_root) if log_root is not None else None
        self.verdicts_log_path = verdicts_log_path
        self.registrations_log_path = registrations_log_path
        evolution_cfg = config.get("evolution", {}) or {}
        self.max_concurrent_branches = int(
            evolution_cfg.get("max_concurrent_branches", DEFAULT_MAX_CONCURRENT_BRANCHES)
        )

        self._branches: dict[str, BranchMeta] = {}
        self._promotion_order: list[str] = []  # 分支名, 按晋升发生的先后顺序
        self._current_main_branch: str = "main"
        self._replay_state()

    # ------------------------------------------------------------------
    # M5 崩溃恢复:从两份 append-only 日志重放状态(见模块 docstring)
    # ------------------------------------------------------------------

    def _replay_state(self) -> None:
        for record in log_writer.read_jsonl(self.registrations_log_path, root=self.log_root):
            name, created_date = record["branch"], record["created_date"]
            if name in self._branches:
                continue  # 幂等:同一条注册记录不会因为重放而出现两次分支
            self._branches[name] = BranchMeta(name=name, created_date=created_date, status="active")

        for record in log_writer.read_jsonl(self.verdicts_log_path, root=self.log_root):
            name = record["branch"]
            meta = self._branches.get(name)
            if meta is None or meta.status != "active":
                # 防御性处理:理论上不应发生(每条 verdict 必然对应一次此前的
                # register_branch),但重放是"信任日志、不信任内存"的场景,
                # 遇到无法对应的记录选择跳过而不是崩溃。
                continue
            self._apply_verdict_to_state(meta, record["decision"], record["now_date"])

    def _apply_verdict_to_state(self, meta: "BranchMeta", decision: str, now_date: str) -> None:
        if decision == "PROMOTE":
            meta.status = "promoted"
            meta.promoted_date = now_date
            self._current_main_branch = meta.name
            self._promotion_order.append(meta.name)
        elif decision == "ARCHIVE":
            meta.status = "archived"
        elif decision == "FAIL":
            meta.status = "failed"
        else:  # pragma: no cover -- VerdictDecision 是封闭的 Literal
            raise ValueError(f"unknown verdict decision: {decision!r}")

    # ------------------------------------------------------------------
    # 准入(agent-facing,无裁决权)
    # ------------------------------------------------------------------

    def register_branch(self, name: str, created_date: str) -> bool:
        """提交一个新的 evo 分支进入并行模拟池。

        没有任何裁决权 —— 只是把一条 BranchMeta(status="active") 放进池子，
        供未来某次 judge() 调用去评估。

        - 名字已被注册过(不论当前状态是 active/promoted/archived/failed，
          名字不重用)→ 抛 ValueError。
        - 当前 ACTIVE 分支数已达到 max_concurrent_branches → 直接返回
          False，不注册(§3.5"同时并行分支数上限"在这里强制，agent 代码
          无法绕过)。
        - 否则注册成功，返回 True。
        """
        if name in self._branches:
            raise ValueError(f"branch {name!r} is already registered (names are never reused)")
        if len(self.active_branches()) >= self.max_concurrent_branches:
            return False
        # 先写 append-only 登记日志,再更新内存状态——如果进程在这两步之间被杀
        # 掉,下次启动时 _replay_state() 会从日志里重新看到这条注册记录,不会
        # 丢失(唯一的风险窗口是"日志写完、还没来得及返回 True 给调用方"时崩溃,
        # 这种情况下调用方会认为注册失败而重试,而 _branches 里其实已经有这个
        # 名字了——下次调用方重试 register_branch(同名) 会命中上面的 ValueError
        # 分支,这是调用方需要处理的正常竞态,不是本方法的职责)。
        log_writer.append_jsonl(
            self.registrations_log_path, {"branch": name, "created_date": created_date}, root=self.log_root
        )
        self._branches[name] = BranchMeta(name=name, created_date=created_date, status="active")
        return True

    def active_branches(self) -> list[BranchMeta]:
        return [b for b in self._branches.values() if b.status == "active"]

    def branch_meta(self, name: str) -> Optional[BranchMeta]:
        return self._branches.get(name)

    # ------------------------------------------------------------------
    # 裁决(唯一入口,无 agent 覆盖面)
    # ------------------------------------------------------------------

    def judge(
        self,
        now_date: str,
        branch_navs: dict[str, list[tuple[str, float]]],
        benchmark_navs: list[tuple[str, float]],
        branch_dead_flags: dict[str, bool] | None = None,
    ) -> dict[str, Verdict]:
        """对当前所有 ACTIVE 分支做一次裁决。

        没有任何参数可以让调用方直接断言"分支X的判定是PROMOTE"——判定结果
        100%由 (branch_navs, benchmark_navs, branch_dead_flags) 经
        scorer.ratchet_score() + 清算强制FAIL短路逻辑推导得出。

        步骤:
          1. 清算强制FAIL(§0 铁律 liquidation_policy=branch_death,不可配置):
             branch_dead_flags.get(name) is True 的 ACTIVE 分支直接判 FAIL，
             完全不咨询 scorer.ratchet_score。这一步只把"被爆仓的分支"从
             接下来要送进 ratchet_score 的候选集里摘掉，不影响同一次 judge()
             调用里其它 ACTIVE 分支的正常评分。
          2. 剩下的(未被摘掉的)ACTIVE 分支,一次性调用
             scorer.ratchet_score(...),用它们各自的 BranchMeta.created_date
             作为 branch_created_dates -- 这正是窗口对齐机制的落地点。
          3. 每条 Verdict 更新对应 BranchMeta.status;PROMOTE 额外记录
             promoted_date=now_date 并把 current_main_branch 指向该分支(“模拟
             合并”——真正的代码/资产合并与生产流量切换是 main.py/部署层的
             职责，本编排器的范围到"这个分支的账本/代码从此就是 main 的含义"
             这一事实为止)。
          4. 每条 Verdict(含清算强制FAIL)都 append 到 verdicts_log_path。
          5. 已裁决(PROMOTE/ARCHIVE/FAIL)的分支离开 active 集合，腾出名额。

        返回本次调用实际裁决过的 dict[branch_name -> Verdict](没有 ACTIVE
        分支时返回空 dict)。
        """
        branch_dead_flags = branch_dead_flags or {}
        active = self.active_branches()
        if not active:
            return {}

        verdicts: dict[str, Verdict] = {}
        liquidated_names: set[str] = set()

        # 1. 清算强制FAIL —— 无条件、不咨询 scorer,且不影响其它分支的评分。
        for meta in active:
            if branch_dead_flags.get(meta.name) is True:
                liquidated_names.add(meta.name)
                verdicts[meta.name] = Verdict(
                    branch=meta.name,
                    decision="FAIL",
                    score=0.0,
                    max_drawdown_pct=0.0,
                    edge_vs_main_pct=0.0,
                    reason=(
                        "liquidation_policy=branch_death (iron law, not configurable): this branch "
                        "was forcibly liquidated, so it FAILs unconditionally regardless of NAV "
                        "returns -- scorer.ratchet_score was never consulted for this branch"
                    ),
                )

        # 2. 剩余(未被清算摘掉的)ACTIVE分支,一次性送进 scorer.ratchet_score。
        remaining = [m for m in active if m.name not in liquidated_names]
        if remaining:
            branch_created_dates = {m.name: m.created_date for m in remaining}
            # 只把 main + 仍待裁决的分支的 NAV 序列递给 scorer -- 不把本轮被
            # 清算摘掉的分支、或 branch_navs 里可能残留的其它(已裁决过的)
            # 分支数据一并递过去,避免 scorer.ratchet_score 因为"branch_navs
            # 里有一个不在 branch_created_dates 里的候选分支"而报错，也避免
            # 无关分支的数据被意外送进这次裁决。
            filtered_navs: dict[str, list[tuple[str, float]]] = {}
            if "main" in branch_navs:
                filtered_navs["main"] = branch_navs["main"]
            for m in remaining:
                if m.name in branch_navs:
                    filtered_navs[m.name] = branch_navs[m.name]

            scored = self.scorer.ratchet_score(filtered_navs, branch_created_dates, benchmark_navs)
            verdicts.update(scored)

        # 3 & 4. 先落LOG(append-only,权威事实来源),再更新内存状态——顺序与
        # register_branch() 一致,如果进程在"日志写完、内存状态还没更新"之间
        # 被杀掉,下次启动 _replay_state() 会重新从日志推导出正确状态,不会
        # 出现"判定已经发生但状态没反映"这种不一致(状态本身就是日志的派生
        # 视图,不是需要额外保证一致性的独立副本)。
        for name, verdict in verdicts.items():
            log_writer.append_jsonl(
                self.verdicts_log_path,
                {
                    "branch": verdict.branch,
                    "decision": verdict.decision,
                    "score": verdict.score,
                    "edge_vs_main_pct": verdict.edge_vs_main_pct,
                    "max_drawdown_pct": verdict.max_drawdown_pct,
                    "reason": verdict.reason,
                    "now_date": now_date,
                },
                root=self.log_root,
            )
            meta = self._branches[name]
            self._apply_verdict_to_state(meta, verdict.decision, now_date)

        return verdicts

    # ------------------------------------------------------------------
    # 晋升追踪
    # ------------------------------------------------------------------

    def promotion_records(self) -> list[PromotionRecord]:
        """每个"曾经被 PROMOTE 过"的分支一条 PromotionRecord，按晋升发生的
        先后顺序排列。直接喂给 scorer.monthly_report(promotions=...)。"""
        records: list[PromotionRecord] = []
        for name in self._promotion_order:
            meta = self._branches[name]
            assert meta.promoted_date is not None  # 内部不变量:进入 _promotion_order 时必已设置
            records.append(
                PromotionRecord(branch=meta.name, created_date=meta.created_date, promoted_date=meta.promoted_date)
            )
        return records

    @property
    def current_main_branch(self) -> str:
        """"main" 直到第一次 PROMOTE 发生，此后是最近一次被 PROMOTE 的分支名。"""
        return self._current_main_branch
