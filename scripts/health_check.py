"""
scripts/health_check.py —— M5 阶段三:只读每日体检脚本。

铁律(与 webui/app.py 同一纪律):不 import 任何 LOCKED/ASSET 业务模块,只
直接读 LOG/ 下的文件和 state/ 下的心跳文件,零写入。适合放进 cron 每天跑
一次;任一红灯时以非零退出码退出,方便外部监控/告警系统据此触发通知。

跑法:
    cd alphaloop
    python scripts/health_check.py

检查项(见 spec 用户原话):
    - decisions.jsonl 无周期缺口
    - funding.jsonl 覆盖 UTC 0/8/16
    - nav.tsv 三线更新
    - 进程存活(读 ignite.py 写的心跳文件)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_ROOT = PROJECT_ROOT / "LOG"
STATE_ROOT = PROJECT_ROOT / "state"

DECISION_INTERVAL_MS = 4 * 3_600_000
SETTLE_HOURS_UTC = (0, 8, 16)
STALE_MULTIPLE = 2.5
DAY_MS = 86_400_000


def _now_ms() -> int:
    return int(time.time() * 1000)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def check_decision_gaps(now_ms: int) -> tuple[bool, str]:
    records = [r for r in _read_jsonl(LOG_ROOT / "decisions.jsonl") if r.get("branch", "main") == "main"]
    if not records:
        return False, "decisions.jsonl 无任何记录(系统可能尚未点火,或决策周期完全停摆)"
    last_ts = max(r["ts"] for r in records if "ts" in r)
    gap_ms = now_ms - last_ts
    if gap_ms > DECISION_INTERVAL_MS * STALE_MULTIPLE:
        return False, f"距上一条决策已 {gap_ms // 60000} 分钟,超过 {DECISION_INTERVAL_MS * STALE_MULTIPLE // 60000} 分钟阈值"
    return True, f"最近一条决策 {gap_ms // 60000} 分钟前,正常"


def check_funding_coverage(now_ms: int) -> tuple[bool, str]:
    records = _read_jsonl(LOG_ROOT / "funding.jsonl")
    if not records:
        return False, "funding.jsonl 无任何记录"
    seen_hours_today = set()
    today_start = (now_ms // DAY_MS) * DAY_MS
    for r in records:
        ts = r.get("ts")
        if ts is None or ts < today_start:
            continue
        hour = (ts // 3_600_000) % 24
        if hour in SETTLE_HOURS_UTC:
            seen_hours_today.add(hour)
    expected_hours = {h for h in SETTLE_HOURS_UTC if today_start + h * 3_600_000 <= now_ms}
    missing = expected_hours - seen_hours_today
    if missing:
        return False, f"今日UTC结算点缺失: {sorted(missing)}(已覆盖: {sorted(seen_hours_today)})"
    return True, f"今日UTC结算点已覆盖: {sorted(seen_hours_today) or '(今日尚未到任何结算点)'}"


def check_nav_updated(now_ms: int) -> tuple[bool, str]:
    path = LOG_ROOT / "nav.tsv"
    if not path.exists():
        return False, "nav.tsv 不存在"
    lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if len(lines) < 2:
        return False, "nav.tsv 只有表头,没有任何净值记录"
    last_row = lines[-1].split("\t")
    header = lines[0].split("\t")
    if len(last_row) != len(header):
        return False, "nav.tsv 最后一行列数与表头不一致(文件可能损坏)"
    return True, f"nav.tsv 最新一行: {dict(zip(header, last_row))}"


def check_process_heartbeat(now_ms: int) -> tuple[bool, str]:
    path = STATE_ROOT / "ignite_heartbeat.json"
    if not path.exists():
        return False, "ignite_heartbeat.json 不存在(ignite.py 从未启动过,或从未成功写过一次心跳)"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return False, f"心跳文件读取/解析失败: {exc!r}"
    ts = data.get("ts_ms")
    if ts is None:
        return False, "心跳文件缺少 ts_ms 字段"
    gap_ms = now_ms - ts
    # 主循环轮询间隔是60秒,给足够宽松的容忍度(5分钟)避免偶发调度延迟被误判红灯。
    if gap_ms > 5 * 60_000:
        return False, f"心跳已 {gap_ms // 60000} 分钟未更新,进程可能已死"
    return True, f"心跳 {gap_ms // 1000} 秒前,进程存活"


def main() -> int:
    now_ms = _now_ms()
    checks = [
        ("decisions_no_gap", check_decision_gaps),
        ("funding_coverage", check_funding_coverage),
        ("nav_updated", check_nav_updated),
        ("process_heartbeat", check_process_heartbeat),
    ]
    all_green = True
    print(f"=== AlphaLoop health_check @ {now_ms} ===")
    for name, fn in checks:
        ok, detail = fn(now_ms)
        status = "GREEN" if ok else "RED"
        print(f"[{status}] {name}: {detail}")
        if not ok:
            all_green = False

    if not all_green:
        print("\n>>> 至少一项红灯,需要人工介入排查 <<<")
        return 1
    print("\n全部绿灯。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
