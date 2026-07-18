"""
scripts/brain_researcher.py —— 第五代架构"大脑转岗量化研究员"模块。

背景(第五代架构决定):废除大模型直接交易——所有交易由 ASSET/strategy/
policies/{policy_id}.py 里的确定性策略代码执行(5分钟一次,零LLM介入)。
大模型(每个"大脑德比"分支绑定的 llm_provider)转岗为量化研究员:每小时
检验自己那条确定性策略代码在实盘(纸面)上的真实表现、写研究日志、必要时
提交一版新策略代码——新代码必须先过 policy_lint 静态审查、再在与
scripts/research_loop.py 完全同一套多窗口回测机制下"凭成绩"跑赢现任
policy_id,才允许上线替换。

本模块只提供一个模块级入口 run_brain_review(),不做任何越权副作用:
  - 唯一允许的磁盘写入落在 LOG/(经 LOCKED.log_writer.append_jsonl)、
    ASSET/strategy/policies/(新策略候选文件)、以及调用方传入的
    memory_store。
  - 不读任何全局状态、不读墙钟——LLM client / 当前时刻(now_ms)/ 各种目录
    路径全部由调用方通过参数注入,时钟纪律与 ASSET/memory/engine.py、
    LOCKED/reflector.py 同源(§0 铁律)。
  - 不修改 scripts/ignite.py / config.yaml / webui/(另一位代理并行改造这些
    文件,负责接线),也不修改锦标赛名册 meta 本身——meta 是只读输入,
    roster 里 policy_id/policy_history 的写回由调用方(主代理接线处)在
    拿到本函数返回值之后完成。

设计上几条"规格书未明确、本次实现自行拍板"的决定,如实标注:

  1. run_brain_review() 签名里没有 project_root/data_pipeline/symbols/
     data_end_ts 这些参数(契约由任务书直接给定,不能加参数)。回测关卡
     需要的这几样东西,这里从 policies_dir 反推:policies_dir 在生产环境
     里固定是 PROJECT_ROOT/ASSET/strategy/policies(与
     ASSET.strategy.policies.POLICIES_DIR 的构造方式同源),往上退3层
     (parents[2])就是 project_root。这不是新增自由度,只是把"policies_dir
     的相对位置是项目里到处硬编码的既有约定"这件事显式利用起来。
  2. direction_memo.md 摘要路径同理:experiments_dir =
     project_root / config.backtest.scratch_root(默认
     "ASSET/strategy/experiments"),与 scripts/research_loop.py
     build_loop_context() 里 scratch_root 的算法完全一致。
  3. 复用 scripts/research_loop.py 的:build_default_windows(切窗口)、
     determine_data_end_ts(推数据终点)、load_universe_symbols(读universe)、
     load_policy_from_dir(可参数化目录加载策略模块)、compute_val_stats/
     compute_val_trade_count(验证窗口统计口径)、decide_keep_or_revert
     (棘轮判定,"平手不算赢"这条纪律与内环研究循环一字不差)。
     BacktestEngine/BacktestWindow 直接从 LOCKED.backtest_engine 导入到
     本模块的模块级命名空间(而不是经由 research_loop 转手),让测试能像
     tests/test_research_loop.py 一样直接
     `monkeypatch.setattr(brain_researcher, "BacktestEngine", Fake)`。
     同理 DataPipeline/determine_data_end_ts 也在模块级命名空间下,方便
     测试整体打桩掉"需要真实历史数据"的部分,不需要在 tmp_path 下伪造
     parquet 缓存文件。
  4. gate_result 的 verdict 取四个值之一:"accepted"(换码)/"rejected"
     (回测分数不如现任,或验证窗口交易笔数不够 min_val_trades)/
     "lint_failed"(静态审查不过)/"error"(回测过程本身抛异常)。这是
     本次新增的词汇表,不是从别处照搬的既有约定。
  5. 冷却期(quant_derby.proposal_cooldown_hours,默认4小时)挡下的
     propose 请求,brain_journals.jsonl 里记录的 action 字段落成
     "keep"(因为确实没有发生任何策略变更尝试),原始意图保留在
     journal 文本与返回的 state 里,不额外污染 schema。
"""
from __future__ import annotations

import json
import re
import traceback
from pathlib import Path
from typing import Any, Callable, Optional

from ASSET.strategy.policy_lint import lint_policy_source
from LOCKED.backtest_engine import BacktestEngine, BacktestWindow  # noqa: F401 (BacktestWindow 供测试/类型引用)
from LOCKED.data_pipeline import DataPipeline
from LOCKED.log_writer import append_jsonl, read_jsonl
from scripts.llm_bridge import _strip_code_fences as strip_code_fences
from scripts.research_loop import (
    build_default_windows,
    compute_val_stats,
    compute_val_trade_count,
    decide_keep_or_revert,
    determine_data_end_ts,
    load_policy_from_dir,
    load_universe_symbols,
    safe_policy_path,
    validate_policy_id,
)

_MS_PER_HOUR = 3_600_000
_JSON_PARSE_MAX_ATTEMPTS = 3  # 首次 + 最多2次重试(spec:"解析失败重试最多2次")
_JOURNAL_LOG_PATH = "brain_journals.jsonl"
_GATE_LOG_PATH = "policy_gate.jsonl"
_RECENT_JOURNAL_LIMIT = 5
_DIRECTION_MEMO_SUMMARY_CHARS = 1500  # 摘要口径:直接截断前N字符,不额外调LLM二次摘要
_DEFAULT_PROPOSAL_COOLDOWN_HOURS = 4
_MEMORY_CONTENT_MAX_CHARS = 4000  # 防止极端超长journal把memory_store撑爆,防御性上限


# ---------------------------------------------------------------------------
# 1. 小工具:路径推导 / 净值统计 / 供应商slug
# ---------------------------------------------------------------------------


def _infer_project_root(policies_dir: Path) -> Path:
    """见模块docstring设计决定1。"""
    return Path(policies_dir).resolve().parents[2]


def _infer_experiments_dir(project_root: Path, config: dict) -> Path:
    """见模块docstring设计决定2,与 research_loop.build_loop_context 同一算法。"""
    scratch_root = (config.get("backtest", {}) or {}).get("scratch_root", "ASSET/strategy/experiments")
    return project_root / scratch_root


def _slice_since(series: list[tuple[int, float]], since_ms: int, until_ms: int) -> list[tuple[int, float]]:
    return [(ts, nav) for ts, nav in series if since_ms <= ts <= until_ms]


def _return_pct(series: list[tuple[int, float]]) -> Optional[float]:
    """独立小实现,不导入 scripts/ignite.py 的同名私有函数(下划线前缀代表
    那是它自己的内部实现细节,不是稳定对外API,不应该被本模块跨文件依赖)。"""
    if len(series) < 2:
        return None
    first, last = series[0][1], series[-1][1]
    if first == 0:
        return None
    return (last - first) / first * 100.0


def _max_drawdown_pct(series: list[tuple[int, float]]) -> Optional[float]:
    if not series:
        return None
    peak = series[0][1]
    max_dd = 0.0
    for _, nav in series:
        peak = max(peak, nav)
        if peak > 0:
            max_dd = max(max_dd, (peak - nav) / peak * 100.0)
    return max_dd


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _provider_slug(provider: Optional[str]) -> str:
    """把 llm_provider(如 "ARK-Kimi"/"deepseek-v3")规整成满足
    ^[a-z][a-z0-9_]{2,40}$ 白名单前缀的slug片段(还差"_v{N}"后缀)。"""
    raw = (provider or "brain").strip().lower()
    slug = _SLUG_RE.sub("_", raw).strip("_")
    if not slug:
        slug = "brain"
    if not slug[0].isalpha():
        slug = f"b_{slug}"
    return slug[:24]  # 留出"_v{N}"后缀与40字符上限之间的余量


def _next_candidate_policy_id(provider_slug: str, version_counter: int, policies_dir: Path) -> tuple[str, int]:
    """从 state.version_counter 递增找一个尚未存在于 policies_dir 下、且满足
    policy_id 白名单的候选名。返回 (policy_id, 用掉的新version_counter值)。"""
    n = int(version_counter) + 1
    while True:
        candidate = f"{provider_slug}_v{n}"
        try:
            validate_policy_id(candidate)
        except ValueError:
            n += 1
            continue
        if not (Path(policies_dir) / f"{candidate}.py").exists():
            return candidate, n
        n += 1


# ---------------------------------------------------------------------------
# 2. 上下文材料收集(日志尾部 / 方向备忘录摘要)
# ---------------------------------------------------------------------------


def _read_recent_journals(log_root: Path, branch: str, limit: int = _RECENT_JOURNAL_LIMIT) -> list[dict]:
    records = read_jsonl(_JOURNAL_LOG_PATH, root=Path(log_root))
    own = [r for r in records if r.get("branch") == branch]
    own.sort(key=lambda r: r.get("ts", 0))
    return own[-limit:]


def _read_direction_memo_summary(project_root: Path, config: dict) -> Optional[str]:
    """读 ASSET/strategy/experiments/direction_memo.md;不存在/读取失败一律
    返回None(冷启动/总监从未运行过是正常状态,不是错误)。"""
    path = _infer_experiments_dir(project_root, config) / "direction_memo.md"
    if not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    text = text.strip()
    if not text:
        return None
    if len(text) > _DIRECTION_MEMO_SUMMARY_CHARS:
        text = text[:_DIRECTION_MEMO_SUMMARY_CHARS] + "\n...(截断)"
    return text


# ---------------------------------------------------------------------------
# 3. review prompt 构造 + LLM JSON 解析(带重试)
# ---------------------------------------------------------------------------

_DERBY_RULES_TEXT = (
    "你是{provider}大脑,你写的策略代码({policy_desc})正在管理100U真实模拟盘"
    "资金,每5分钟由确定性代码(零LLM介入)自动执行交易。你本人不再直接下单,"
    "你的职责是量化研究员:每小时复盘一次自己策略的实盘表现,写研究日志,"
    "必要时提交新版策略代码。\n\n"
    "德比规则(关系到你的生死):\n"
    "- 爆仓(该分支模拟账户被强平/清零)即死;\n"
    "- 72小时滚动窗口末位斩杀:表现持续垫底的分支会被替换;\n"
    "- 死亡分支的位置会被原脑复活(同一个大脑再给一次100U从头再来),死亡"
    "计数对全体公示;\n"
    "- 你的策略一旦跑赢main主账户超过0.5个百分点,就会被提升接管主账户。\n"
)


def _build_identity_section(provider: str, policy_description: str) -> str:
    return _DERBY_RULES_TEXT.format(provider=provider or "未知", policy_desc=policy_description or "(无运行中策略)")


def _build_performance_section(
    since_ms: Optional[int],
    now_ms: int,
    return_since_last_review_pct: Optional[float],
    current_drawdown_pct: Optional[float],
) -> str:
    since_text = "首次检验(无上一次检验记录)" if since_ms is None else f"上次检验(ts={since_ms})以来"
    return_text = "无法计算(数据点不足)" if return_since_last_review_pct is None else f"{return_since_last_review_pct:.3f}%"
    dd_text = "无法计算(暂无净值数据)" if current_drawdown_pct is None else f"{current_drawdown_pct:.3f}%"
    return (
        f"## 净值表现(截至ts={now_ms})\n"
        f"- {since_text}的净值变化:{return_text}\n"
        f"- 当前相对历史高点的回撤:{dd_text}\n"
    )


def build_review_prompt(
    *,
    branch: str,
    meta: dict,
    current_policy_id: Optional[str],
    current_policy_source: str,
    policy_description: str,
    performance_section: str,
    recent_journals: list[dict],
    market_context: str,
    direction_memo_summary: Optional[str],
    in_cooldown: bool,
    cooldown_remaining_hours: float,
) -> str:
    provider = str(meta.get("llm_provider") or "未知")
    journal_lines = (
        "\n".join(
            f"- ts={j.get('ts')} action={j.get('action')} journal={(j.get('journal') or '')[:120]!r}"
            for j in recent_journals
        )
        if recent_journals
        else "(暂无历史日志,这是第一次检验)"
    )
    cooldown_text = (
        f"当前处于提案冷却期,距离可以再次提交新策略代码还有约{cooldown_remaining_hours:.1f}小时——"
        "这段时间内即使你输出action=\"propose\"也会被直接搁置、不会进入回测关卡,不要浪费精力写代码,"
        "把这次的journal专注在观察与假设积累上。"
        if in_cooldown
        else "当前不处于提案冷却期,如果你有充分把握的改进,可以输出action=\"propose\"提交新策略代码。"
    )
    memo_section = (
        f"## 公共研发部方向备忘录摘要\n{direction_memo_summary}\n" if direction_memo_summary else ""
    )
    return (
        f"{_build_identity_section(provider, policy_description)}\n"
        f"## 你当前的策略代码(policy_id={current_policy_id or '(无)'})\n"
        f"```python\n{current_policy_source}\n```\n\n"
        f"{performance_section}\n"
        f"## 最近日志(最多{_RECENT_JOURNAL_LIMIT}条)\n{journal_lines}\n\n"
        f"## 市场背景\n{market_context}\n\n"
        f"{memo_section}"
        f"## 提案冷却状态\n{cooldown_text}\n\n"
        "## 本次任务\n"
        "复盘上面的表现数据、日志与市场背景,写一段中文研究日志(journal):"
        "本小时的观察、教训、假设。然后决定 action:\"keep\"(继续用当前策略代码,"
        "不提交变更)或\"propose\"(提交一版完整的新策略代码替换当前实现)。\n\n"
        "## StrategyContext 协议(decide(ctx)的唯一输入,action=\"propose\"时必须遵守)\n"
        "- ctx.ts: int,当前bar时间戳(UTC毫秒),策略判断\"现在几点\"只能用这个字段\n"
        "- ctx.positions: dict[str, PerpPosition],当前持仓快照\n"
        "- ctx.snapshot: dict[str, dict],{symbol: {'last': 最新收盘价}}\n"
        "- ctx.recent_bars: dict[str, pd.DataFrame],{symbol: 最近K线,列为"
        "[timestamp,open,high,low,close,volume],按timestamp升序,最后一行是最新已收盘K线}\n"
        "- ctx.memory_context: list[str],记忆检索文本,可以是空列表\n"
        "模块必须导出 decide(ctx) -> list[Decision]、REQUIRED_HISTORY_BARS: int、"
        "DESCRIPTION: str。无信号时返回空列表[],不要构造hold Decision。\n\n"
        "## 硬性确定性/沙箱禁止事项(action=\"propose\"时违反任何一条都会被静态检查直接判废)\n"
        "- 禁止任何网络调用/网络库导入(requests/httpx/urllib/socket/ccxt/anthropic/aiohttp/http/websocket/grpc等)\n"
        "- 禁止读墙钟(time.time()/time.time_ns()/datetime.now()/utcnow()/date.today()等)——"
        "\"现在几点\"只能来自 ctx.ts\n"
        "- 禁止随机数(random模块、numpy.random)\n"
        "- 禁止 os.environ / os.getenv / subprocess / exec / eval\n"
        "- open() 只允许只读模式,禁止写/追加/独占模式\n\n"
        "## 输出要求\n"
        "严格输出一个JSON对象(不要markdown代码块,不要任何JSON之外的文字):\n"
        '{"journal": "本小时的观察/教训/假设(中文)", "action": "keep 或 propose", '
        '"policy_code": "action=propose时为完整python源码字符串,action=keep时可以是空字符串"}'
    )


def build_repair_prompt(original_prompt: str, raw_response: str, error: str) -> str:
    return (
        f"{original_prompt}\n\n"
        "---\n"
        f"你上一次的输出未通过解析,原因:{error}。\n"
        f"你上一次的原始输出:\n{raw_response}\n\n"
        "请严格只输出一个合法JSON对象(不要markdown代码块,不要任何JSON之外的文字),"
        '字段为 {"journal": str, "action": "keep"或"propose", "policy_code": str}。'
    )


def _parse_review_response(raw: str) -> tuple[Optional[dict], Optional[str]]:
    try:
        parsed = json.loads(strip_code_fences(raw))
    except (json.JSONDecodeError, TypeError, ValueError):
        return None, "response_not_valid_json"
    if not isinstance(parsed, dict):
        return None, "response_not_a_json_object"
    journal = parsed.get("journal")
    if not isinstance(journal, str) or not journal.strip():
        return None, "missing_non_empty_journal"
    action = parsed.get("action")
    if action not in ("keep", "propose"):
        return None, 'action_must_be_"keep"_or_"propose"'
    if action == "propose":
        policy_code = parsed.get("policy_code")
        if not isinstance(policy_code, str) or not policy_code.strip():
            return None, "propose_requires_non_empty_policy_code"
    return parsed, None


def _call_review_llm_with_retries(
    deep_llm: Callable[[str], str], prompt: str, max_attempts: int = _JSON_PARSE_MAX_ATTEMPTS
) -> tuple[Optional[dict], list[str]]:
    """调用 deep_llm 解析出 journal/action/policy_code,解析失败就把错误原因
    喂回去重试,最多 max_attempts 次(首次+最多2次重试)。返回
    (parsed_or_None, error_log)。"""
    errors: list[str] = []
    current_prompt = prompt
    for _ in range(max_attempts):
        raw = deep_llm(current_prompt)
        parsed, error = _parse_review_response(raw)
        if error is None:
            return parsed, errors
        errors.append(error)
        current_prompt = build_repair_prompt(prompt, raw, error)
    return None, errors


# ---------------------------------------------------------------------------
# 4. 回测关卡(复用 research_loop 的窗口切分/评分口径)
# ---------------------------------------------------------------------------


def run_policy_gate(
    *,
    candidate_policy_id: str,
    policy_code: str,
    incumbent_policy_id: Optional[str],
    policies_dir: Path,
    config: dict,
) -> dict:
    """把 policy_code 落盘为 {policies_dir}/{candidate_policy_id}.py,过
    policy_lint,再跑与 scripts/research_loop.py 完全同一套多窗口回测,和
    incumbent_policy_id(现任policy_id,可以是None——代表这个分支之前还没有
    正式代码,首次亮码直接与0分基线比)在同一批窗口上对比打分。

    通过标准:candidate的score严格优于incumbent的score(平手=不换,与
    decide_keep_or_revert棘轮判定同一条纪律)。验证窗口总交易笔数低于
    config.backtest.min_val_trades 时,不管分数多高一律rejected(与
    research_loop.run_single_experiment的min_val_trades门槛同一口径)。

    单次review的回测最多跑一个候选(本函数每次只处理一个candidate,加上
    最多一次incumbent对照跑,不做多候选竞赛)。BacktestEngine/DataPipeline/
    determine_data_end_ts 均在模块级命名空间下,测试可以整体
    monkeypatch.setattr(brain_researcher, "BacktestEngine"/"DataPipeline"/
    "determine_data_end_ts", ...) 打桩掉真实历史数据依赖。
    """
    policies_dir = Path(policies_dir)
    policies_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = safe_policy_path(policies_dir, candidate_policy_id)
    # 先落盘(spec顺序:先存文件,再lint)——不管lint/回测结果如何都"留档",
    # 不做的是"revert删除文件"这件事(与research_loop不同,那里revert会删
    # 从未提交过的新文件;这里研究员产出的候选文件本身就是研究记录,予以保留)。
    candidate_path.write_text(policy_code, encoding="utf-8")

    violations = lint_policy_source(policy_code)
    if violations:
        return {
            "verdict": "lint_failed",
            "candidate_policy_id": candidate_policy_id,
            "incumbent_policy_id": incumbent_policy_id,
            "candidate_score": None,
            "incumbent_score": None,
            "violations": violations,
        }

    try:
        project_root = _infer_project_root(policies_dir)
        data_end_ts = determine_data_end_ts(config, project_root / "data_cache")
        symbols = load_universe_symbols(project_root)
        scratch_root = _infer_experiments_dir(project_root, config)
        windows = build_default_windows(config, data_end_ts)
        data_pipeline = DataPipeline(exchange_id=(config.get("data", {}) or {}).get("exchange", "okx"))
        engine = BacktestEngine(config=config, data_pipeline=data_pipeline, scratch_root=scratch_root)

        candidate_module = load_policy_from_dir(candidate_policy_id, policies_dir)
        candidate_results = engine.run(
            candidate_module.decide, symbols, windows, experiment_id=f"gate_{candidate_policy_id}"
        )
        candidate_scoring = {label: r for label, r in candidate_results.items() if label != "holdout"}
        candidate_score = engine.score(candidate_scoring)
        candidate_val_trades = compute_val_trade_count(candidate_results)
        candidate_val_edge, candidate_val_dd = compute_val_stats(candidate_results)

        incumbent_score = 0.0
        incumbent_metrics: dict[str, Any] = {}
        if incumbent_policy_id and (policies_dir / f"{incumbent_policy_id}.py").exists():
            incumbent_module = load_policy_from_dir(incumbent_policy_id, policies_dir)
            incumbent_results = engine.run(
                incumbent_module.decide, symbols, windows, experiment_id=f"gate_incumbent_{incumbent_policy_id}"
            )
            incumbent_scoring = {label: r for label, r in incumbent_results.items() if label != "holdout"}
            incumbent_score = engine.score(incumbent_scoring)
            inc_edge, inc_dd = compute_val_stats(incumbent_results)
            incumbent_metrics = {"val_edge_vs_benchmark_pct": inc_edge, "val_max_drawdown_pct": inc_dd}

        min_val_trades = int((config.get("backtest", {}) or {}).get("min_val_trades", 10))
        metrics = {
            "candidate_val_trade_count": candidate_val_trades,
            "candidate_val_edge_vs_benchmark_pct": candidate_val_edge,
            "candidate_val_max_drawdown_pct": candidate_val_dd,
            "min_val_trades": min_val_trades,
            **({f"incumbent_{k}": v for k, v in incumbent_metrics.items()} if incumbent_metrics else {}),
        }

        if candidate_val_trades < min_val_trades:
            verdict = "rejected"
            reason = "insufficient_trades"
        else:
            verdict = "accepted" if decide_keep_or_revert(candidate_score, incumbent_score) == "kept" else "rejected"
            reason = None if verdict == "accepted" else "score_not_better"

        return {
            "verdict": verdict,
            "reason": reason,
            "candidate_policy_id": candidate_policy_id,
            "incumbent_policy_id": incumbent_policy_id,
            "candidate_score": candidate_score,
            "incumbent_score": incumbent_score,
            "metrics": metrics,
        }
    except Exception as exc:  # noqa: BLE001 - 回测异常不能让调用方崩,候选文件依然留档
        return {
            "verdict": "error",
            "candidate_policy_id": candidate_policy_id,
            "incumbent_policy_id": incumbent_policy_id,
            "candidate_score": None,
            "incumbent_score": None,
            "error": f"{exc!r}\n{traceback.format_exc(limit=5)}",
        }


# ---------------------------------------------------------------------------
# 5. 日志落盘
# ---------------------------------------------------------------------------


def _append_brain_journal(
    log_root: Path, *, ts: int, branch: str, provider: Optional[str], journal: str, action: str, policy_id: Optional[str]
) -> None:
    append_jsonl(
        _JOURNAL_LOG_PATH,
        {"ts": ts, "branch": branch, "provider": provider, "journal": journal, "action": action, "policy_id": policy_id},
        root=Path(log_root),
    )


def _append_policy_gate_log(log_root: Path, *, ts: int, branch: str, gate_result: dict) -> None:
    append_jsonl(
        _GATE_LOG_PATH,
        {
            "ts": ts,
            "branch": branch,
            "candidate_id": gate_result.get("candidate_policy_id"),
            "incumbent_id": gate_result.get("incumbent_policy_id"),
            "candidate_score": gate_result.get("candidate_score"),
            "incumbent_score": gate_result.get("incumbent_score"),
            "verdict": gate_result.get("verdict"),
            "metrics": gate_result.get("metrics"),
        },
        root=Path(log_root),
    )


# ---------------------------------------------------------------------------
# 6. 主入口
# ---------------------------------------------------------------------------


def _cooldown_status(state: dict, now_ms: int, cooldown_hours: float) -> tuple[bool, float]:
    last_proposal_ms = state.get("last_proposal_ms")
    if last_proposal_ms is None:
        return False, 0.0
    elapsed_hours = (now_ms - int(last_proposal_ms)) / _MS_PER_HOUR
    remaining = cooldown_hours - elapsed_hours
    return remaining > 0, max(remaining, 0.0)


def run_brain_review(
    branch: str,
    meta: dict,
    deep_llm: Callable[[str], str],
    now_ms: int,
    *,
    config: dict,
    memory_store: Any,
    nav_series_lookup: Callable[[str], list[tuple[int, float]]],
    market_context: str,
    log_root: Path,
    policies_dir: Path,
    state: dict,
) -> dict:
    """M9+ 大脑德比研究员review入口,完整契约见模块docstring/任务书。

    健壮性:LLM异常、回测异常都被局部捕获,任何未预料的异常还有本函数体
    外层这一道兜底(见最下面的try/except)——绝不允许一次review把调用方
    (主循环)整个拖崩。"""
    state = dict(state or {})
    log_root = Path(log_root)
    policies_dir = Path(policies_dir)
    provider = meta.get("llm_provider")
    current_policy_id = meta.get("policy_id")

    try:
        # ---- 材料收集 ----
        current_policy_source = "(当前无运行中的策略代码)"
        policy_description = "(无运行中策略)"
        if current_policy_id:
            policy_path = policies_dir / f"{current_policy_id}.py"
            if policy_path.exists():
                current_policy_source = policy_path.read_text(encoding="utf-8")
                try:
                    module = load_policy_from_dir(current_policy_id, policies_dir)
                    policy_description = getattr(module, "DESCRIPTION", current_policy_id)
                except Exception:  # noqa: BLE001 - 加载失败不影响review继续,只是描述退化
                    policy_description = current_policy_id

        try:
            nav_series = list(nav_series_lookup(branch) or [])
        except Exception:  # noqa: BLE001 - 净值查询失败不阻塞review,退化为无数据
            nav_series = []
        last_review_ms = state.get("last_review_ms")
        since_ms = last_review_ms if last_review_ms is not None else (meta.get("created_ms") or 0)
        full_series = [pt for pt in nav_series if pt[0] <= now_ms]
        recent_series = _slice_since(full_series, since_ms, now_ms)
        if len(recent_series) < 2:
            recent_series = full_series
        return_since_last_review_pct = _return_pct(recent_series)
        current_drawdown_pct = _max_drawdown_pct(full_series)
        performance_section = _build_performance_section(
            last_review_ms, now_ms, return_since_last_review_pct, current_drawdown_pct
        )

        recent_journals = _read_recent_journals(log_root, branch)

        project_root = _infer_project_root(policies_dir)
        direction_memo_summary = _read_direction_memo_summary(project_root, config)

        cooldown_hours = float(
            (config.get("quant_derby", {}) or {}).get("proposal_cooldown_hours", _DEFAULT_PROPOSAL_COOLDOWN_HOURS)
        )
        in_cooldown, cooldown_remaining_hours = _cooldown_status(state, now_ms, cooldown_hours)

        prompt = build_review_prompt(
            branch=branch,
            meta=meta,
            current_policy_id=current_policy_id,
            current_policy_source=current_policy_source,
            policy_description=policy_description,
            performance_section=performance_section,
            recent_journals=recent_journals,
            market_context=market_context,
            direction_memo_summary=direction_memo_summary,
            in_cooldown=in_cooldown,
            cooldown_remaining_hours=cooldown_remaining_hours,
        )

        # ---- 调用LLM,解析失败重试最多2次,全败降级为keep ----
        try:
            parsed, parse_errors = _call_review_llm_with_retries(deep_llm, prompt)
        except Exception as exc:  # noqa: BLE001 - LLM调用本身抛异常(网络/超时等),降级为keep
            parsed, parse_errors = None, [f"llm_call_raised: {exc!r}"]

        state["last_review_ms"] = now_ms

        if parsed is None:
            journal = f"本小时研究员输出解析失败({'; '.join(parse_errors)}),降级为keep,不做任何策略变更。"
            _append_brain_journal(
                log_root, ts=now_ms, branch=branch, provider=provider, journal=journal,
                action="keep", policy_id=current_policy_id,
            )
            _write_memory_summary(memory_store, branch=branch, ts=now_ms, journal=journal)
            return {"journal": journal, "proposed_policy_id": None, "gate_result": None, "state": state}

        journal_text = parsed["journal"]
        llm_action = parsed["action"]

        if llm_action == "propose" and in_cooldown:
            journal_text = f"{journal_text}\n\n[系统] 当前处于提案冷却期,提案被搁置,按keep处理。"
            _append_brain_journal(
                log_root, ts=now_ms, branch=branch, provider=provider, journal=journal_text,
                action="keep", policy_id=current_policy_id,
            )
            _write_memory_summary(memory_store, branch=branch, ts=now_ms, journal=journal_text)
            return {"journal": journal_text, "proposed_policy_id": None, "gate_result": None, "state": state}

        if llm_action == "keep":
            _append_brain_journal(
                log_root, ts=now_ms, branch=branch, provider=provider, journal=journal_text,
                action="keep", policy_id=current_policy_id,
            )
            _write_memory_summary(memory_store, branch=branch, ts=now_ms, journal=journal_text)
            return {"journal": journal_text, "proposed_policy_id": None, "gate_result": None, "state": state}

        # ---- action == "propose" 且不在冷却期:走回测关卡 ----
        state["last_proposal_ms"] = now_ms
        provider_slug = _provider_slug(provider)
        version_counter = int(state.get("version_counter", 0) or 0)
        candidate_policy_id, new_version_counter = _next_candidate_policy_id(
            provider_slug, version_counter, policies_dir
        )
        state["version_counter"] = new_version_counter

        try:
            gate_result = run_policy_gate(
                candidate_policy_id=candidate_policy_id,
                policy_code=parsed["policy_code"],
                incumbent_policy_id=current_policy_id,
                policies_dir=policies_dir,
                config=config,
            )
        except Exception as exc:  # noqa: BLE001 - 兜底:run_policy_gate内部已经try/except,这里是双保险
            gate_result = {
                "verdict": "error",
                "candidate_policy_id": candidate_policy_id,
                "incumbent_policy_id": current_policy_id,
                "candidate_score": None,
                "incumbent_score": None,
                "error": f"{exc!r}\n{traceback.format_exc(limit=5)}",
            }

        _append_policy_gate_log(log_root, ts=now_ms, branch=branch, gate_result=gate_result)

        verdict = gate_result.get("verdict")
        verdict_note = {
            "accepted": f"回测关卡通过:候选{candidate_policy_id}分数({gate_result.get('candidate_score')})"
            f"严格优于现任{current_policy_id}({gate_result.get('incumbent_score')}),已换码。",
            "rejected": f"回测关卡未通过(reason={gate_result.get('reason')}):候选{candidate_policy_id}"
            f"分数({gate_result.get('candidate_score')})未能严格优于现任({gate_result.get('incumbent_score')}),"
            "策略文件留档但不上线。",
            "lint_failed": f"候选{candidate_policy_id}静态审查未通过,违规:{gate_result.get('violations')},不上线。",
            "error": f"回测关卡运行异常:{gate_result.get('error')},不上线。",
        }.get(verdict, f"未知verdict={verdict!r}")
        full_journal = f"{journal_text}\n\n[系统] {verdict_note}"

        proposed_policy_id = candidate_policy_id if verdict == "accepted" else None

        _append_brain_journal(
            log_root, ts=now_ms, branch=branch, provider=provider, journal=full_journal,
            action="propose", policy_id=current_policy_id,
        )
        _write_memory_summary(memory_store, branch=branch, ts=now_ms, journal=full_journal)

        return {
            "journal": full_journal,
            "proposed_policy_id": proposed_policy_id,
            "gate_result": gate_result,
            "state": state,
        }
    except Exception as exc:  # noqa: BLE001 - 最外层兜底,任何未预料异常都不能崩调用方
        journal = f"[brain_researcher] 内部异常,降级为keep:{exc!r}"
        try:
            _append_brain_journal(
                log_root, ts=now_ms, branch=branch, provider=provider, journal=journal,
                action="keep", policy_id=current_policy_id,
            )
        except Exception:  # noqa: BLE001 - 连日志都写不进去时,至少不能再抛出去
            pass
        state["last_review_ms"] = now_ms
        return {
            "journal": journal,
            "proposed_policy_id": None,
            "gate_result": {"verdict": "error", "error": f"{exc!r}\n{traceback.format_exc(limit=5)}"},
            "state": state,
        }


def _write_memory_summary(memory_store: Any, *, branch: str, ts: int, journal: str) -> None:
    """把journal摘要写进memory_store(L2,分支隔离)。memory_store为None
    (测试里不关心记忆写入路径时)直接跳过——这不是"允许调用方随意省略"的
    生产行为,只是让测试可以不必每次都构造一个真实MemoryStore。"""
    if memory_store is None:
        return
    content = journal if len(journal) <= _MEMORY_CONTENT_MAX_CHARS else journal[:_MEMORY_CONTENT_MAX_CHARS] + "...(截断)"
    memory_store.write(content=content, ts=ts, layer="L2", branch=branch)


__all__ = [
    "run_brain_review",
    "run_policy_gate",
    "build_review_prompt",
    "build_repair_prompt",
]
