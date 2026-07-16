"""
scripts/llm_bridge.py —— 让 Claude Code agent 本身充当 Trader/Researcher/
Reflector 的 llm_client,不需要单独的 ANTHROPIC_API_KEY / anthropic SDK。

("你是agent,所有ai操作都是由agent开始做" —— 用户原话,M5阶段三)

协议:AgentBridgeLLMClient 被 Trader/Researcher/Reflector 当作
Callable[[str], str] 调用时:
    1. 把 prompt 写到 state/llm_pending_request.json(带一个唯一 request_id)。
    2. 阻塞轮询,等待 state/llm_response_<request_id>.json 出现(轮询间隔
       poll_seconds,超时 timeout_seconds 后放弃)。
    3. 读到响应文件后删除两个文件,把响应文本原样返回给调用方
       (Trader/Researcher/Reflector 自己负责解析/校验这段文本)。
    4. 超时:抛出 TimeoutError——main.py 已经有的失败隔离机制会接住这个
       异常(Trader外层的超时线程包装、Researcher/Reflector的try/except),
       不需要本文件自己再实现一套重试逻辑。

respond_to_pending() 是 Claude Code agent(我)用来"回答"一条 pending
request 的入口——每次我签入(通过 ScheduleWakeup 定期检查)时,读一下有没有
pending request,有就读 prompt、推理、调这个函数把响应写回去。
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Optional

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATE_ROOT = PROJECT_ROOT / "state"
REQUEST_PATH = STATE_ROOT / "llm_pending_request.json"

logger = logging.getLogger(__name__)


class AgentBridgeLLMClient:
    def __init__(self, poll_seconds: float = 2.0, timeout_seconds: float = 1800.0):
        self.poll_seconds = poll_seconds
        self.timeout_seconds = timeout_seconds
        STATE_ROOT.mkdir(parents=True, exist_ok=True)

    def __call__(self, prompt: str) -> str:
        request_id = uuid.uuid4().hex
        response_path = STATE_ROOT / f"llm_response_{request_id}.json"

        REQUEST_PATH.write_text(
            json.dumps(
                {"request_id": request_id, "prompt": prompt, "written_at_ms": int(time.time() * 1000)},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        deadline = time.time() + self.timeout_seconds
        while time.time() < deadline:
            if response_path.exists():
                try:
                    data = json.loads(response_path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    time.sleep(self.poll_seconds)
                    continue
                response_path.unlink(missing_ok=True)
                return data["response"]
            time.sleep(self.poll_seconds)

        REQUEST_PATH.unlink(missing_ok=True)
        raise TimeoutError(
            f"AgentBridgeLLMClient: no response for request {request_id} within {self.timeout_seconds}s "
            "(the operating agent did not check in / respond in time)"
        )


def _strip_code_fences(text: str) -> str:
    """剥掉模型响应外层的 markdown 代码围栏(```json ... ``` / ``` ... ```)。

    子代理/API模型都真实犯过"明明要求纯JSON却包了围栏"的错误(签入agent
    人工审核时代靠手拦,无人值守后必须在客户端层统一兜住)。只剥最外层、
    且只在整段响应确实以围栏开头结尾时才剥,不碰正文里合法出现的反引号。"""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return text
    lines = stripped.splitlines()
    if len(lines) < 2:
        return text
    # 第一行是 ``` 或 ```json 之类;找配对的结尾围栏(最后一个非空行)
    if lines[-1].strip() != "```":
        return text
    return "\n".join(lines[1:-1])


class _DailyBudgetedLLMClient:
    """AnthropicLLMClient 与 OpenAIChatLLMClient(2026-07-16新增,多供应商
    分支路由)共用的每日调用预算记账逻辑——两个供应商的账单风险是同一件事
    (代码bug导致的死循环调用把账单烧穿),抽成基类避免逐字复制同一套
    读文件/判定/写回逻辑。

    预算文件默认共享同一份 state/llm_api_usage.json——即使分支路由到不同
    供应商,"今天总共调用了多少次真实付费API"仍然是同一个需要被封顶的
    总量,不需要、也不应该按供应商拆分成多份独立预算(拆分后总账单风险
    敞口反而变成"供应商数 x max_daily_calls",违背这道闸门本来的目的)。
    子类只需要在 __call__ 里调 _check_budget_or_raise() 拿到当天用量字典、
    真正调用成功后调 _record_usage() 记账。
    """

    def __init__(self, max_daily_calls: int = 600, usage_path: Optional[Path] = None):
        self.max_daily_calls = max_daily_calls
        self.usage_path = usage_path or (STATE_ROOT / "llm_api_usage.json")
        STATE_ROOT.mkdir(parents=True, exist_ok=True)

    def _load_usage(self) -> dict:
        today = time.strftime("%Y-%m-%d", time.gmtime())
        if self.usage_path.exists():
            try:
                usage = json.loads(self.usage_path.read_text(encoding="utf-8"))
                if usage.get("date") == today:
                    return usage
            except (json.JSONDecodeError, OSError):
                pass
        return {"date": today, "calls": 0, "input_tokens": 0, "output_tokens": 0}

    def _save_usage(self, usage: dict) -> None:
        self.usage_path.write_text(json.dumps(usage, ensure_ascii=False), encoding="utf-8")

    def _check_budget_or_raise(self) -> dict:
        usage = self._load_usage()
        if usage["calls"] >= self.max_daily_calls:
            raise RuntimeError(
                f"llm_daily_budget_exhausted: {usage['calls']} calls today >= "
                f"max_daily_calls {self.max_daily_calls}; degrading to safe fallback "
                "until UTC midnight"
            )
        return usage

    def _record_usage(self, usage: dict, input_tokens: int, output_tokens: int) -> None:
        usage["calls"] += 1
        usage["input_tokens"] += input_tokens
        usage["output_tokens"] += output_tokens
        self._save_usage(usage)


class AnthropicLLMClient(_DailyBudgetedLLMClient):
    """直连 Anthropic API 的 llm_client(用户2026-07-15要求,24h无人值守
    服务器模式)。与 AgentBridgeLLMClient 完全同构:Callable[[str], str],
    失败抛异常交给 main.py 既有的失败隔离层(Trader重试/兜底hold、
    Researcher/Reflector的try-except-skip),自己不做业务级重试。

    额外职责(桥模式不需要、无人值守必须有的):
      - 每日调用预算熔断(见 _DailyBudgetedLLMClient):超过 max_daily_calls
        直接抛 RuntimeError,让当天剩余周期全部走安全降级路径(hold/skip),
        第二天零点(UTC)自动恢复。
      - 响应围栏剥离:见 _strip_code_fences()。
    API key 从环境变量 ANTHROPIC_API_KEY 读取,不进代码/配置文件。

    认证方式(2026-07-16新增,接入火山方舟Ark Agent Plan专属端点后发现的
    真实差异):不是所有Anthropic消息格式兼容端点都用同一种认证头。
    DeepSeek/智谱GLM这类走 `x-api-key` 请求头(anthropic SDK对应
    `Anthropic(api_key=...)`);火山方舟Ark Agent Plan端点
    (https://ark.cn-beijing.volces.com/api/plan,官方文档就是给Claude Code
    配ANTHROPIC_BASE_URL用的)要求标准的 `Authorization: Bearer` 头
    (SDK对应 `Anthropic(auth_token=...)`)。用户提供的官方文档同时指出
    智谱GLM官方接入方式也是ANTHROPIC_AUTH_TOKEN风格。auth参数就是为了
    覆盖这个差异:"api_key"(默认,兼容老行为)走x-api-key,"auth_token"
    走Bearer,由调用方(ignite.py build_provider_clients)按供应商注册表里
    每个供应商的auth字段决定传哪一种。
    """

    def __init__(
        self,
        model: str,
        max_tokens: int = 4096,
        max_daily_calls: int = 600,
        timeout_seconds: float = 180.0,
        usage_path: Optional[Path] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        auth: str = "api_key",
    ):
        import anthropic  # 延迟导入:桥模式/本地测试不需要装 anthropic SDK

        if auth not in ("api_key", "auth_token"):
            raise ValueError(f"AnthropicLLMClient: auth 必须是 'api_key' 或 'auth_token',得到 {auth!r}")

        super().__init__(max_daily_calls=max_daily_calls, usage_path=usage_path)
        self.model = model
        self.max_tokens = max_tokens
        # base_url:兼容Anthropic消息格式的第三方端点(用户2026-07-15选定
        # DeepSeek,base_url=https://api.deepseek.com/anthropic,模型
        # deepseek-v4-flash/pro;2026-07-16新增智谱GLM、火山方舟Ark Agent
        # Plan同样走这个格式)。None时走官方Anthropic端点。
        client_kwargs: dict = {"timeout": timeout_seconds, "max_retries": 2}
        if base_url:
            client_kwargs["base_url"] = base_url
        # api_key/auth_token:2026-07-16多供应商改造前,anthropic SDK默认
        # 自己读ANTHROPIC_API_KEY环境变量;现在供应商注册表(见ignite.py
        # build_provider_clients)按各自的api_key_env读取key、显式传入,
        # 兼容老路径——不传key时SDK仍按自己的默认行为读ANTHROPIC_API_KEY。
        # auth决定这个key填进 api_key(x-api-key头)还是 auth_token
        # (Authorization: Bearer头),见上面class docstring的说明。
        if api_key:
            if auth == "auth_token":
                client_kwargs["auth_token"] = api_key
            else:
                client_kwargs["api_key"] = api_key
        self._client = anthropic.Anthropic(**client_kwargs)

    def __call__(self, prompt: str) -> str:
        usage = self._check_budget_or_raise()

        message = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(
            block.text for block in message.content if getattr(block, "type", None) == "text"
        )

        self._record_usage(
            usage,
            getattr(message.usage, "input_tokens", 0),
            getattr(message.usage, "output_tokens", 0),
        )

        return _strip_code_fences(text)


class OpenAIChatLLMClient(_DailyBudgetedLLMClient):
    """OpenAI Chat Completions 格式的 llm_client(2026-07-16新增,用户要求
    接入火山方舟豆包 Doubao-Seed-2.1-turbo,该端点是OpenAI格式而非
    Anthropic兼容格式,与 AnthropicLLMClient 服务的DeepSeek/智谱GLM端点
    不同,所以需要单独一个类)。

    与 AnthropicLLMClient 完全同构:Callable[[str], str],共用同一套每日
    预算记账(_DailyBudgetedLLMClient)和围栏剥离(_strip_code_fences)。
    直接用 requests POST {base_url}/chat/completions,不引入 openai SDK
    依赖——本项目已经在用 requests(见 scripts/search_client.py),没有
    必要为了一个额外的HTTP客户端多装一个SDK。

    重试:简单的"最多再试 max_retries 次"策略,不区分错误类型——网络抖动/
    限流/端点临时5xx都值得原地重试;真正的业务级失败隔离(重试耗尽后
    怎么办)仍然交给上层调用方(Trader的hold兜底、Researcher/Reflector的
    try-except-skip),这里的重试只覆盖"这一次HTTP调用本身"。

    思考类模型(reasoning models,豆包/DeepSeek系列都有对应版本)的响应
    可能把真实内容放进 reasoning_content 字段、把 content 留空——只在
    content为空时才回退读reasoning_content,并记一条warning,不能反过来
    优先用reasoning_content(那是模型的思考过程草稿,不是最终答案,格式
    通常也不满足下游要求的纯JSON)。
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        max_tokens: int = 4096,
        max_daily_calls: int = 600,
        timeout_seconds: float = 180.0,
        usage_path: Optional[Path] = None,
        max_retries: int = 2,
    ):
        super().__init__(max_daily_calls=max_daily_calls, usage_path=usage_path)
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

    def __call__(self, prompt: str) -> str:
        usage = self._check_budget_or_raise()

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
        }

        last_exc: Optional[Exception] = None
        data: Optional[dict] = None
        for attempt in range(self.max_retries + 1):  # 首次尝试 + 最多 max_retries 次重试
            try:
                resp = requests.post(url, headers=headers, json=body, timeout=self.timeout_seconds)
                resp.raise_for_status()
                data = resp.json()
                break
            except Exception as exc:  # noqa: BLE001 -- 网络/超时/HTTP错误统一走同一套重试
                last_exc = exc
                if attempt >= self.max_retries:
                    raise
                continue

        assert data is not None  # for循环要么break成功、要么在最后一次尝试raise,不会掉到这里
        _ = last_exc  # 仅用于上面raise前的赋值追踪,读到这里说明已经成功

        message = (data.get("choices") or [{}])[0].get("message") or {}
        text = message.get("content") or ""
        if not text.strip():
            reasoning = message.get("reasoning_content")
            if reasoning:
                logger.warning(
                    "OpenAIChatLLMClient: model %r response content 为空,回退到"
                    "reasoning_content(思考类模型常见现象)", self.model,
                )
                text = reasoning

        usage_stats = data.get("usage") or {}
        self._record_usage(
            usage,
            int(usage_stats.get("prompt_tokens", 0) or 0),
            int(usage_stats.get("completion_tokens", 0) or 0),
        )

        return _strip_code_fences(text)


def read_pending_request() -> Optional[dict]:
    """我(Claude Code agent)签入时调用:有 pending request 就返回它的内容,
    没有就返回 None。只读,不消费/不删除——真正"回答"要调
    respond_to_pending()。"""
    if not REQUEST_PATH.exists():
        return None
    try:
        return json.loads(REQUEST_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def respond_to_pending(request_id: str, response_text: str) -> None:
    """我(Claude Code agent)读完 prompt、推理完之后调用:把响应写回去,
    唤醒正在阻塞轮询的 AgentBridgeLLMClient。

    关键点(修复过一次的 bug,见 tests/test_llm_bridge.py 的回归测试):必须
    在这里**立即、原子性地"消费掉"这条 pending request**(删除
    REQUEST_PATH),而不是留给客户端那边在收到响应之后再回头清理。原来的
    设计是客户端收到响应后才顺手删 REQUEST_PATH,这中间存在一个窗口——
    responder 线程/进程紧接着开始等下一条请求时,REQUEST_PATH 可能还没被
    客户端删掉,于是会把"已经回答过一次的旧请求"错当成"一条新请求"再回答
    一遍,而真正的新请求永远等不到人理。这里改成 respond_to_pending 自己
    先校验 REQUEST_PATH 里的 request_id 确实是本次要回答的这条,校验通过才
    删除+写响应,两步之间不留能被误读的中间状态。
    """
    request_id_still_matches = False
    if REQUEST_PATH.exists():
        try:
            if json.loads(REQUEST_PATH.read_text(encoding="utf-8")).get("request_id") == request_id:
                request_id_still_matches = True
        except (json.JSONDecodeError, OSError):
            pass
    if request_id_still_matches:
        REQUEST_PATH.unlink(missing_ok=True)

    response_path = STATE_ROOT / f"llm_response_{request_id}.json"
    response_path.write_text(
        json.dumps({"request_id": request_id, "response": response_text}, ensure_ascii=False),
        encoding="utf-8",
    )
