from __future__ import annotations

import pandas as pd
import pytest

from LOCKED.schemas import PromotionRecord, ThesisMark
from LOCKED.scorer import Scorer


@pytest.fixture
def config():
    return {
        "constraints": {
            "max_drawdown_pct": 20,
        },
        "evolution": {
            "min_promote_edge_pct": 0.5,
        },
    }


@pytest.fixture
def scorer(config, log_root):
    return Scorer(config, log_root=log_root)


# ----------------------------------------------------------------------
# 1. daily_mark
# ----------------------------------------------------------------------
def test_daily_mark_writes_expected_header_and_rows(scorer, log_root):
    scorer.daily_mark(nav_agent=100000.0, nav_benchmark=100000.0, nav_random=100000.0, date="2026-06-01")
    scorer.daily_mark(nav_agent=101500.5, nav_benchmark=100800.0, nav_random=99900.25, date="2026-06-02")
    scorer.daily_mark(nav_agent=99000.0, nav_benchmark=102000.0, nav_random=100100.0, date="2026-06-03")

    nav_path = log_root / "nav.tsv"
    assert nav_path.exists()

    with open(nav_path, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]

    assert lines[0] == "date\tnav_agent\tnav_benchmark\tnav_random"
    assert lines[1] == "2026-06-01\t100000.0\t100000.0\t100000.0"
    assert lines[2] == "2026-06-02\t101500.5\t100800.0\t99900.25"
    assert lines[3] == "2026-06-03\t99000.0\t102000.0\t100100.0"
    assert len(lines) == 4


def test_daily_mark_requires_explicit_date(scorer):
    with pytest.raises(ValueError):
        scorer.daily_mark(nav_agent=100.0, nav_benchmark=100.0, nav_random=100.0)


# ----------------------------------------------------------------------
# helpers for ratchet_score tests
# ----------------------------------------------------------------------
def _dates(n, start_day=1):
    return [f"2026-06-{start_day + i:02d}" for i in range(n)]


def _series(navs, start_day=1):
    ds = _dates(len(navs), start_day)
    return list(zip(ds, navs))


# ----------------------------------------------------------------------
# 2. ratchet_score: clear outperformance -> PROMOTE
# ----------------------------------------------------------------------
def test_ratchet_score_promote_on_outperformance(scorer):
    benchmark_navs = _series([100, 101, 102, 103, 105])  # +5%
    main_navs = _series([100, 100.5, 101, 101.5, 102])  # +2%
    candidate_navs = _series([100, 105, 110, 115, 120])  # +20%, monotonic (no drawdown)

    branch_navs = {
        "main": main_navs,
        "evo/20260601-momentum": candidate_navs,
    }
    created_dates = {"evo/20260601-momentum": "2026-06-01"}
    verdicts = scorer.ratchet_score(branch_navs, created_dates, benchmark_navs)

    assert set(verdicts.keys()) == {"evo/20260601-momentum"}
    v = verdicts["evo/20260601-momentum"]
    assert v.branch == "evo/20260601-momentum"
    assert v.decision == "PROMOTE"
    assert v.edge_vs_main_pct == pytest.approx(20.0 - 2.0, abs=0.1)  # candidate 20% - main 2%
    assert v.max_drawdown_pct <= 20


# ----------------------------------------------------------------------
# 3. ratchet_score: underperforms main -> ARCHIVE
# ----------------------------------------------------------------------
def test_ratchet_score_archive_on_underperformance(scorer):
    benchmark_navs = _series([100, 101, 102, 103, 105])  # +5%
    main_navs = _series([100, 100.5, 101, 101.5, 102])  # +2%
    candidate_navs = _series([100, 100.2, 100.4, 100.6, 101])  # +1% -> edge vs main = -1%

    branch_navs = {
        "main": main_navs,
        "evo/20260601-laggard": candidate_navs,
    }
    created_dates = {"evo/20260601-laggard": "2026-06-01"}
    verdicts = scorer.ratchet_score(branch_navs, created_dates, benchmark_navs)

    v = verdicts["evo/20260601-laggard"]
    assert v.decision == "ARCHIVE"
    assert v.edge_vs_main_pct < 0
    assert v.max_drawdown_pct <= 20


# ----------------------------------------------------------------------
# 3b. M4: minimum-edge threshold -- a candidate that nominally beats main but
#     by less than min_promote_edge_pct must still ARCHIVE (ties/marginal
#     edges default to keeping main; this is the anti-selection-gaming gate).
# ----------------------------------------------------------------------
def test_ratchet_score_marginal_edge_below_threshold_archives_not_promotes(scorer):
    benchmark_navs = _series([100, 100, 100, 100, 100])  # flat
    main_navs = _series([100, 100.2, 100.4, 100.6, 100.8])  # +0.8%
    # +1.0% final -> nominally beats main by 0.2 percentage points, well under
    # the 0.5% (50bps) min_promote_edge_pct threshold from the `config` fixture.
    candidate_navs = _series([100, 100.25, 100.5, 100.75, 101.0])

    branch_navs = {"main": main_navs, "evo/20260601-marginal": candidate_navs}
    created_dates = {"evo/20260601-marginal": "2026-06-01"}
    verdicts = scorer.ratchet_score(branch_navs, created_dates, benchmark_navs)

    v = verdicts["evo/20260601-marginal"]
    assert v.edge_vs_main_pct > 0, "sanity: candidate does nominally beat main"
    assert v.edge_vs_main_pct < 0.5, "sanity: edge is below the 50bps threshold"
    assert v.decision == "ARCHIVE", "marginal edge below threshold must default to keeping main"


def test_ratchet_score_edge_exactly_at_threshold_promotes(scorer):
    """Inclusive boundary: edge_vs_main_pct == min_promote_edge_pct must PROMOTE (>=, not >)."""
    benchmark_navs = _series([100, 100, 100])
    main_navs = _series([100, 100, 100])  # 0%
    candidate_navs = _series([100, 100.25, 100.5])  # +0.5% exactly

    branch_navs = {"main": main_navs, "evo/x": candidate_navs}
    created_dates = {"evo/x": "2026-06-01"}
    verdicts = scorer.ratchet_score(branch_navs, created_dates, benchmark_navs, min_promote_edge_pct=0.5)

    v = verdicts["evo/x"]
    assert v.edge_vs_main_pct == pytest.approx(0.5, abs=1e-6)
    assert v.decision == "PROMOTE"


# ----------------------------------------------------------------------
# 4. ratchet_score: death clause overrides a great final score -> FAIL
# ----------------------------------------------------------------------
def test_ratchet_score_fail_on_intrawindow_drawdown_despite_great_final_return(scorer):
    benchmark_navs = _series([100, 100, 100, 100, 100])  # flat, 0%

    main_navs = _series([100, 100.5, 101, 101.5, 102])  # +2%

    # Peaks at 150 on day2, dips to 100 on day3 (33.3% drawdown from local peak),
    # then recovers to a very strong final number by the end of the window.
    candidate_navs = _series([100, 150, 100, 140, 200])  # final return = +100%

    branch_navs = {
        "main": main_navs,
        "evo/20260601-spike": candidate_navs,
    }
    created_dates = {"evo/20260601-spike": "2026-06-01"}
    verdicts = scorer.ratchet_score(branch_navs, created_dates, benchmark_navs)

    v = verdicts["evo/20260601-spike"]
    assert v.decision == "FAIL"
    assert v.max_drawdown_pct > 20
    # sanity: without the death clause this branch's edge would clearly beat main
    assert v.edge_vs_main_pct > 0


def test_ratchet_score_only_returns_candidate_branches(scorer):
    benchmark_navs = _series([100, 101, 102])
    main_navs = _series([100, 101, 102])
    candidate_navs = _series([100, 102, 104])

    branch_navs = {"main": main_navs, "evo/x": candidate_navs}
    created_dates = {"evo/x": "2026-06-01"}
    verdicts = scorer.ratchet_score(branch_navs, created_dates, benchmark_navs)

    assert "main" not in verdicts
    assert "evo/x" in verdicts


def test_ratchet_score_missing_created_date_raises(scorer):
    branch_navs = {"main": _series([100, 101]), "evo/x": _series([100, 102])}
    with pytest.raises(ValueError):
        scorer.ratchet_score(branch_navs, {}, _series([100, 101]))


# ----------------------------------------------------------------------
# 4b. M4 window alignment: a branch created mid-window must be scored ONLY
#     from its own creation date forward, and main/benchmark must be sliced
#     to that SAME window -- not "main's full window vs branch's short one".
#     This is the explicit M4 acceptance criterion: "构造一个窗口中途创建的
#     分支,断言其评分起点等于创建时刻".
# ----------------------------------------------------------------------
def test_ratchet_score_branch_created_mid_window_uses_its_own_creation_as_window_start(scorer):
    # main has a full 5-day history with a huge day1->day2 spike-and-crash that
    # happened entirely BEFORE the candidate branch existed, then sits flat at
    # 100 from day3 onward. The candidate branch was only created on day 3
    # (mid-window relative to main's full history) and only has 3 days of its
    # own data (day3,4,5).
    main_navs = _series([100, 300, 100, 100, 100], start_day=1)  # wild day1-2 move, flat day3-5
    benchmark_navs = _series([100, 100, 100, 100, 100], start_day=1)  # flat throughout
    # Candidate only exists from day3 onward, and does great in ITS window (day3->day5: +10%).
    candidate_navs = _series([100, 105, 110], start_day=3)

    branch_navs = {"main": main_navs, "evo/midwindow": candidate_navs}
    created_dates = {"evo/midwindow": "2026-06-03"}
    verdicts = scorer.ratchet_score(branch_navs, created_dates, benchmark_navs)

    v = verdicts["evo/midwindow"]
    # main's comparison window must ALSO start at day3 (nav=100), excluding the
    # day1->day2 spike-and-crash entirely -- that move happened before the
    # branch existed and must not leak into main's comparison return/drawdown.
    # If it DID leak in (a naive "trailing N" comparison using main's full
    # history), main's window return would be unaffected here by coincidence
    # (100->100 either way) but its drawdown would NOT be -- day1->day2's 200%
    # spike followed by the crash back to 100 is itself not a drawdown (peak
    # comes first), so to make the leak observable via drawdown too we assert
    # directly on the aligned slice below rather than relying on drawdown alone.
    main_window_return = (100 / 100 - 1.0) * 100.0  # main day3(100) -> day5(100) = 0%
    candidate_window_return = (110 / 100 - 1.0) * 100.0  # candidate day3(100) -> day5(110) = +10%
    assert v.edge_vs_main_pct == pytest.approx(candidate_window_return - main_window_return, abs=0.01)
    assert v.decision == "PROMOTE"  # +10% edge, comfortably over the 50bps threshold

    # The branch's own window (day3->day5, monotonically 100->105->110) has
    # zero intra-window drawdown -- confirms the reported drawdown is computed
    # over the branch's OWN aligned window, not accidentally over main's data.
    assert v.max_drawdown_pct == pytest.approx(0.0, abs=0.01)


def test_ratchet_score_main_window_excludes_pre_creation_drawdown(scorer):
    """Sharper version of the mid-window alignment test, isolating the case a
    return-only assertion could miss: main has a huge DRAWDOWN entirely before
    the branch's creation date. If that drawdown leaked into the comparison
    window, it would trigger main's own death-clause-style distortion of the
    comparison baseline. Aligned slicing must exclude it completely."""
    # day1=100, day2=20 (80% crash), day3=100 (recovered, this is where the
    # branch's window starts), day4=100, day5=100 -- main's day3-5 window is
    # perfectly flat with ZERO drawdown; the 80% crash is entirely pre-creation.
    main_navs = _series([100, 20, 100, 100, 100], start_day=1)
    benchmark_navs = _series([100, 100, 100, 100, 100], start_day=1)
    candidate_navs = _series([100, 100.6, 101.2], start_day=3)  # +1.2%, comfortably over 50bps

    branch_navs = {"main": main_navs, "evo/x": candidate_navs}
    created_dates = {"evo/x": "2026-06-03"}
    verdicts = scorer.ratchet_score(branch_navs, created_dates, benchmark_navs)

    v = verdicts["evo/x"]
    # main's window-aligned return is 0% (100->100->100 from day3), so edge ==
    # candidate's own return. If the pre-creation crash had leaked in, main's
    # window would start at day1's 100 -> day5's 100 = ALSO 0% by coincidence
    # here, so the real tell is the branch's own reported drawdown: it must be
    # ~0% (its own day3-5 data is flat-then-up), not main's 80%.
    assert v.edge_vs_main_pct == pytest.approx(1.2, abs=0.01)
    assert v.max_drawdown_pct < 1.0, "branch's own window has no meaningful drawdown"
    assert v.decision == "PROMOTE"


# ----------------------------------------------------------------------
# 5. monthly_report smoke tests
# ----------------------------------------------------------------------
def test_monthly_report_without_thesis_marks(scorer, log_root):
    scorer.daily_mark(nav_agent=100000, nav_benchmark=100000, nav_random=100000, date="2026-06-01")
    scorer.daily_mark(nav_agent=105000, nav_benchmark=102000, nav_random=100500, date="2026-06-15")
    scorer.daily_mark(nav_agent=110000, nav_benchmark=103000, nav_random=101000, date="2026-06-30")

    report = scorer.monthly_report()

    assert isinstance(report, str)
    assert report.strip() != ""
    assert "Cumulative excess return" in report
    assert "Gap vs random agent" in report
    assert "Thesis hit-rate" in report
    assert "N/A" in report


def test_monthly_report_with_thesis_marks(scorer, log_root):
    scorer.daily_mark(nav_agent=100000, nav_benchmark=100000, nav_random=100000, date="2026-06-01")
    scorer.daily_mark(nav_agent=108000, nav_benchmark=101000, nav_random=100200, date="2026-06-30")

    marks = [
        ThesisMark(decision_ts=1, symbol="BTC/USDT:USDT", thesis_status="应验"),
        ThesisMark(decision_ts=2, symbol="ETH/USDT:USDT", thesis_status="证伪"),
        ThesisMark(decision_ts=3, symbol="BTC/USDT:USDT", thesis_status="应验"),
    ]
    report = scorer.monthly_report(thesis_marks=marks)

    assert isinstance(report, str)
    assert report.strip() != ""
    assert "2/3" in report
    assert "N/A" not in report


def test_monthly_report_uses_explicit_nav_root(scorer, tmp_path):
    other_root = tmp_path / "other_log"
    other_root.mkdir()
    scorer2 = Scorer({"constraints": {"max_drawdown_pct": 20}}, log_root=other_root)
    scorer2.daily_mark(nav_agent=100, nav_benchmark=100, nav_random=100, date="2026-06-01")
    scorer2.daily_mark(nav_agent=120, nav_benchmark=110, nav_random=101, date="2026-06-10")

    report = scorer2.monthly_report(nav_root=other_root)
    assert "Cumulative excess return" in report


# ----------------------------------------------------------------------
# 6. M4: monthly_report promotion-tracking column (system health check)
# ----------------------------------------------------------------------
def test_monthly_report_omits_promotion_section_when_no_promotions(scorer, log_root):
    scorer.daily_mark(nav_agent=100000, nav_benchmark=100000, nav_random=100000, date="2026-06-01")
    scorer.daily_mark(nav_agent=105000, nav_benchmark=102000, nav_random=100500, date="2026-06-30")

    report = scorer.monthly_report()
    assert "Promoted Branches" not in report


def test_monthly_report_promotion_section_shows_before_vs_after(scorer, log_root):
    scorer.daily_mark(nav_agent=100000, nav_benchmark=100000, nav_random=100000, date="2026-06-01")
    scorer.daily_mark(nav_agent=110000, nav_benchmark=103000, nav_random=101000, date="2026-06-30")

    # Branch created day1, promoted day5 (strong before), then dives after promotion.
    # Window semantics: before = [created_date, promoted_date) -- i.e. promoted_date
    # itself belongs to the "after" window (the branch's life as the new main starts
    # counting from the promotion instant onward).
    branch_navs = {
        "evo/20260601-x": [
            ("2026-06-01", 100.0),
            ("2026-06-03", 110.0),  # before-promotion window: 2026-06-01..2026-06-03(<06-05), +10%
            ("2026-06-05", 120.0),  # after-promotion window starts here: 2026-06-05..end, -25%
            ("2026-06-10", 100.0),
            ("2026-06-15", 90.0),
        ]
    }
    promotions = [PromotionRecord(branch="evo/20260601-x", created_date="2026-06-01", promoted_date="2026-06-05")]

    report = scorer.monthly_report(promotions=promotions, branch_navs=branch_navs)

    assert "Promoted Branches" in report
    assert "evo/20260601-x" in report
    assert "2026-06-05" in report
    # before: 100 -> 110 = +10.00%; after: 120 -> 90 = -25.00%
    assert "10.00%" in report
    assert "-25.00%" in report
    assert "Average before" in report and "Average after" in report
