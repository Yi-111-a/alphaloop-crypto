"""
scorer.py —— 唯一裁判(§2.3)。

铁律:
- 策略agent运行时无权读取本文件源码(访问控制在部署/权限层强制,不在本文件内实现)。
- 本模块只消费"前向"净值数据(daily_mark 写入的 nav.tsv / 调用方传入的
  branch_navs)。没有任何历史回测数据源接入 ratchet_score,不要新增一条
  "在回测数据上跑PROMOTE"的代码路径 —— 棘轮判定只吃前向数据(§0 铁律、§7）。
- 本文件不提供任何让 ASSET 区代码内省或修改评分逻辑的钩子(no callback/plugin
  hooks that let strategy code alter scoring behaviour)。
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from LOCKED.log_writer import LOG_ROOT, append_tsv_row
from LOCKED.schemas import PromotionRecord, ThesisMark, Verdict

NAV_TSV_RELATIVE_PATH = "nav.tsv"
NAV_TSV_HEADER = ["date", "nav_agent", "nav_benchmark", "nav_random"]

DEFAULT_MIN_PROMOTE_EDGE_PCT = 0.5  # 50bps,可被 config.evolution.min_promote_edge_pct 覆盖


def _compounded_return_pct(navs: list[tuple[str, float]]) -> float:
    """窗口内扣费收益率(百分比):(end/start - 1) * 100。"""
    if len(navs) < 2:
        return 0.0
    start_nav = navs[0][1]
    end_nav = navs[-1][1]
    if start_nav == 0:
        return 0.0
    return (end_nav / start_nav - 1.0) * 100.0


def _max_drawdown_pct(navs: list[tuple[str, float]]) -> float:
    """窗口内最大回撤(百分比),相对窗口内自身的滚动高点(而非仅窗口起点)。"""
    if not navs:
        return 0.0
    peak = navs[0][1]
    max_dd = 0.0
    for _, nav in navs:
        if nav > peak:
            peak = nav
        if peak > 0:
            dd = (peak - nav) / peak * 100.0
            if dd > max_dd:
                max_dd = dd
    return max_dd


def _slice_from_date(navs: list[tuple[str, float]], start_date: str) -> list[tuple[str, float]]:
    """M4 窗口对齐:取序列中 date >= start_date 的所有点,不是"末尾N个"。

    ISO 日期字符串("YYYY-MM-DD")按字典序比较等价于按时间比较,所以这里直接用
    字符串 >= 比较,不需要解析成 datetime。

    这是修复"错位比较"bug的核心:候选分支只能用自己创建时刻之后的数据评分,
    且 main/benchmark 必须用同一个 start_date 切片,而不是各自取"最近N天"——
    如果分支是在一个固定判定周期中途创建的(比如判定周期是3天一次,分支在
    周期第2天才创建),用"最近3天"切main/benchmark会包含分支根本不存在的
    那1天,把候选分支2天的真实窗口和主线3天的窗口放在一起比较,是系统性偏向
    "运气好活得久"或"运气差活得短"的分支,而不是比较真实的同期表现。
    """
    return [(d, nav) for d, nav in navs if d >= start_date]


class Scorer:
    """唯一裁判:记录三线净值(daily_mark)、棘轮判定(ratchet_score)、月度报告。"""

    def __init__(self, config: dict, log_root: str | Path | None = None) -> None:
        self.config = config
        self.log_root = Path(log_root) if log_root is not None else None
        self.max_drawdown_pct = float(config["constraints"]["max_drawdown_pct"])
        self.min_promote_edge_pct = float(
            (config.get("evolution", {}) or {}).get("min_promote_edge_pct", DEFAULT_MIN_PROMOTE_EDGE_PCT)
        )

    # ------------------------------------------------------------------
    def daily_mark(
        self,
        nav_agent: float,
        nav_benchmark: float,
        nav_random: float,
        date: str | None = None,
    ) -> None:
        """向 LOG/nav.tsv 追加一行三线净值。

        列顺序严格为: date, nav_agent, nav_benchmark, nav_random。
        `date` 必须由调用方以 ISO 日期字符串传入 —— 本方法故意不调用
        datetime.now(),以保证测试可复现、调度时机由 main.py 控制。
        """
        if date is None:
            raise ValueError(
                "daily_mark(date=...) is required; scorer.py never calls datetime.now() "
                "internally so callers must supply a deterministic ISO date string."
            )
        append_tsv_row(
            NAV_TSV_RELATIVE_PATH,
            [date, nav_agent, nav_benchmark, nav_random],
            header=NAV_TSV_HEADER,
            root=self.log_root,
        )

    # ------------------------------------------------------------------
    def ratchet_score(
        self,
        branch_navs: dict[str, list[tuple[str, float]]],
        branch_created_dates: dict[str, str],
        benchmark_navs: list[tuple[str, float]],
        min_promote_edge_pct: float | None = None,
    ) -> dict[str, Verdict]:
        """棘轮判定:每个候选分支 vs 主线 main,输出 dict[branch -> Verdict]。

        M4 修订(窗口对齐 + 最小优势门槛,取代 M1 的"末尾window_days个点"设计):

        - `branch_created_dates[branch]` 是该候选分支的创建日期(ISO字符串),是
          它评分窗口的起点。main 和 benchmark 的序列都用**这同一个日期**切片
          (见 _slice_from_date),不是各自独立取"最近N天"——这就是防止"分支
          挑自己最好的一段 vs 主线整段"这类错位比较的机制本身。main 没有自己
          的"创建日期"概念(它从系统诞生就存在),每次判定时都是"借用"候选分支
          的创建日期作为窗口起点。
        - edge_vs_main_pct = 候选分支窗口内收益率(%) − main同窗口收益率(%)。
          (这与"各自减去同期benchmark收益率再比较"代数等价,benchmark项相互
          抵消,这里直接算更直白,也让 Verdict.edge_vs_main_pct 这个字段名副
          其实。)
        - score 字段保留 M1 语义(候选收益率 − 同窗口benchmark收益率),供报告
          展示用,不参与PROMOTE门槛判定。
        - 死刑条款(优先于分数):窗口内最大回撤(相对窗口内自身滚动高点) >
          max_drawdown_pct → 该分支 verdict.decision 强制为 "FAIL"，无论score
          或edge多好。
        - 最小优势门槛(M4新增,防"多重比较被运气填满"):edge_vs_main_pct >=
          min_promote_edge_pct(默认 config.evolution.min_promote_edge_pct,
          未配置时 0.5 即50bps)→ PROMOTE；否则(含负数、恰好0、或正但不足
          门槛的情况)→ ARCHIVE。平局/微弱优势一律维持主线是故意的现状偏置,
          不是bug。
        - 只返回候选分支(branch_navs 中除 "main" 外、且必须出现在
          branch_created_dates 里的每个 key 一条 Verdict)。
        """
        if "main" not in branch_navs:
            raise ValueError("branch_navs must include a 'main' entry to compute the baseline score")

        edge_threshold = (
            float(min_promote_edge_pct) if min_promote_edge_pct is not None else self.min_promote_edge_pct
        )

        verdicts: dict[str, Verdict] = {}
        for branch, navs in branch_navs.items():
            if branch == "main":
                continue
            if branch not in branch_created_dates:
                raise ValueError(
                    f"branch {branch!r} has no entry in branch_created_dates -- every candidate "
                    "branch's scoring window must be anchored to its own creation date"
                )
            window_start = branch_created_dates[branch]

            branch_window = _slice_from_date(navs, window_start)
            main_window = _slice_from_date(branch_navs["main"], window_start)
            benchmark_window = _slice_from_date(benchmark_navs, window_start)

            branch_return = _compounded_return_pct(branch_window)
            main_return = _compounded_return_pct(main_window)
            benchmark_return = _compounded_return_pct(benchmark_window)

            score = branch_return - benchmark_return  # 报告用,M1遗留语义
            edge_vs_main = branch_return - main_return  # PROMOTE门槛实际判定的数
            drawdown = _max_drawdown_pct(branch_window)

            if drawdown > self.max_drawdown_pct:
                verdicts[branch] = Verdict(
                    branch=branch,
                    decision="FAIL",
                    score=score,
                    max_drawdown_pct=drawdown,
                    edge_vs_main_pct=edge_vs_main,
                    reason=(
                        f"death clause: intra-window (from {window_start}) max drawdown "
                        f"{drawdown:.2f}% > max_drawdown_pct {self.max_drawdown_pct:.2f}%"
                    ),
                )
                continue

            if edge_vs_main >= edge_threshold - 1e-9:
                verdicts[branch] = Verdict(
                    branch=branch,
                    decision="PROMOTE",
                    score=score,
                    max_drawdown_pct=drawdown,
                    edge_vs_main_pct=edge_vs_main,
                    reason=(
                        f"edge vs main {edge_vs_main:.4f}% >= min_promote_edge_pct "
                        f"{edge_threshold:.4f}% (window since {window_start})"
                    ),
                )
            else:
                verdicts[branch] = Verdict(
                    branch=branch,
                    decision="ARCHIVE",
                    score=score,
                    max_drawdown_pct=drawdown,
                    edge_vs_main_pct=edge_vs_main,
                    reason=(
                        f"edge vs main {edge_vs_main:.4f}% < min_promote_edge_pct "
                        f"{edge_threshold:.4f}% (window since {window_start}) -- ties/marginal "
                        "edges default to keeping main"
                    ),
                )
        return verdicts

    # ------------------------------------------------------------------
    def monthly_report(
        self,
        nav_root: str | Path | None = None,
        thesis_marks: Iterable[ThesisMark] | None = None,
        promotions: Iterable[PromotionRecord] | None = None,
        branch_navs: dict[str, list[tuple[str, float]]] | None = None,
        backtest_forward_pairs: Iterable[dict] | None = None,
    ) -> str:
        """读取 LOG/nav.tsv,输出月度 Markdown 报告字符串(不写盘,main.py 负责落地)。

        M4 新增"晋升前后表现对比"栏(promotions + branch_navs 都提供时才输出):
        对每条 PromotionRecord,用 branch_navs[branch] 分别算
        [created_date, promoted_date) 的晋升前收益率和 [promoted_date, 序列末尾]
        的晋升后收益率。这是系统级体检指标,不是决策逻辑的一部分——如果晋升后
        普遍大幅跳水,说明棘轮在系统性地筛选运气而不是真实alpha,需要人类介入
        复查 min_promote_edge_pct 是否设置过低,而不是本方法自动做任何事。

        M8 新增"回测vs前向一致性"栏(backtest_forward_pairs 提供且非空时才
        输出),与上面"晋升前后表现对比"栏同一哲学——纯体检指标,不触发任何
        自动动作。backtest_forward_pairs 里每个元素是一个从"内环回测(§5 M6
        holdout窗口)进入前向池、最终真的被PROMOTE"的分支的一条记录,形如
        {"branch": str, "backtest_holdout_edge_pct": float,
        "forward_edge_pct": float}(调用方——scripts/ignite.py 或未来的M6
        内环收尾脚本——负责从各自的记录里拼出这个列表,本方法不关心它们的
        来源,只按这个约定的dict形状消费)。展示 holdout回测edge 与 前向实际
        edge 的差值(forward - backtest):如果前向edge系统性大幅低于holdout
        回测edge,说明内环回测存在过拟合/数据泄漏风险,需要人类介入复查,而
        不是本方法自动做任何事——没有任何一条记录时优雅跳过整个栏目,不报错。
        """
        root = Path(nav_root) if nav_root is not None else (self.log_root or LOG_ROOT)
        path = root / NAV_TSV_RELATIVE_PATH
        df = pd.read_csv(path, sep="\t")

        first, last = df.iloc[0], df.iloc[-1]
        agent_return_pct = (last["nav_agent"] / first["nav_agent"] - 1.0) * 100.0
        benchmark_return_pct = (last["nav_benchmark"] / first["nav_benchmark"] - 1.0) * 100.0
        random_return_pct = (last["nav_random"] / first["nav_random"] - 1.0) * 100.0
        cumulative_excess_return_pct = agent_return_pct - benchmark_return_pct
        gap_vs_random_pct = agent_return_pct - random_return_pct

        marks = list(thesis_marks) if thesis_marks else []
        if marks:
            hit_count = sum(1 for m in marks if m.thesis_status == "应验")
            hit_rate_str = f"{hit_count}/{len(marks)} ({hit_count / len(marks) * 100:.1f}%)"
        else:
            hit_rate_str = "N/A"

        lines = [
            "# Monthly Report",
            "",
            f"- Period: {first['date']} ~ {last['date']}",
            f"- Cumulative excess return (agent vs benchmark): {cumulative_excess_return_pct:.2f}%",
            f"- Gap vs random agent: {gap_vs_random_pct:.2f}%",
            f"- Thesis hit-rate: {hit_rate_str}",
            "",
            "## Raw returns",
            f"- Agent: {agent_return_pct:.2f}%",
            f"- Benchmark (BTC_HOLD): {benchmark_return_pct:.2f}%",
            f"- Random agent: {random_return_pct:.2f}%",
        ]

        promo_list = list(promotions) if promotions else []
        if promo_list and branch_navs:
            lines.append("")
            lines.append("## Promoted Branches: Before vs After Promotion")
            lines.append(
                "(系统级体检指标,非决策依据。健康系统里两列应量级相当;"
                "晋升后普遍大幅低于晋升前,是棘轮在筛选运气而非真实alpha的直接证据。)"
            )
            lines.append("")
            lines.append("| branch | promoted_date | before % | after % |")
            lines.append("|---|---|---|---|")
            before_returns: list[float] = []
            after_returns: list[float] = []
            for promo in promo_list:
                navs = branch_navs.get(promo.branch, [])
                before_window = [
                    (d, nav) for d, nav in navs if promo.created_date <= d < promo.promoted_date
                ]
                after_window = [(d, nav) for d, nav in navs if d >= promo.promoted_date]
                before_pct = _compounded_return_pct(before_window)
                after_pct = _compounded_return_pct(after_window)
                before_returns.append(before_pct)
                after_returns.append(after_pct)
                lines.append(f"| {promo.branch} | {promo.promoted_date} | {before_pct:.2f}% | {after_pct:.2f}% |")

            if before_returns:
                avg_before = sum(before_returns) / len(before_returns)
                avg_after = sum(after_returns) / len(after_returns)
                lines.append("")
                lines.append(
                    f"- Average before: {avg_before:.2f}% / Average after: {avg_after:.2f}% "
                    f"(n={len(before_returns)} promotions)"
                )

        pairs = list(backtest_forward_pairs) if backtest_forward_pairs else []
        if pairs:
            lines.append("")
            lines.append("## Backtest vs Forward Consistency (M8)")
            lines.append(
                "(体检指标,非决策依据,不触发任何自动动作——与上面"
                "\"晋升前后表现对比\"栏同一哲学。前向edge系统性大幅低于holdout"
                "回测edge,是内环回测过拟合/数据泄漏的直接证据,需要人类介入"
                "复查,而不是本方法自动做任何事。)"
            )
            lines.append("")
            lines.append("| branch | holdout backtest edge % | forward edge % | gap (forward - backtest) % |")
            lines.append("|---|---|---|---|")
            gaps: list[float] = []
            for pair in pairs:
                branch = pair["branch"]
                backtest_edge = float(pair["backtest_holdout_edge_pct"])
                forward_edge = float(pair["forward_edge_pct"])
                gap = forward_edge - backtest_edge
                gaps.append(gap)
                lines.append(f"| {branch} | {backtest_edge:.2f}% | {forward_edge:.2f}% | {gap:.2f}% |")

            avg_gap = sum(gaps) / len(gaps)
            lines.append("")
            lines.append(f"- Average gap: {avg_gap:.2f}% (n={len(gaps)} branches)")

        return "\n".join(lines)
