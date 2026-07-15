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
import time
import uuid
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATE_ROOT = PROJECT_ROOT / "state"
REQUEST_PATH = STATE_ROOT / "llm_pending_request.json"


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


class AnthropicLLMClient:
    """直连 Anthropic API 的 llm_client(用户2026-07-15要求,24h无人值守
    服务器模式)。与 AgentBridgeLLMClient 完全同构:Callable[[str], str],
    失败抛异常交给 main.py 既有的失败隔离层(Trader重试/兜底hold、
    Researcher/Reflector的try-except-skip),自己不做业务级重试。

    额外职责(桥模式不需要、无人值守必须有的):
      - 每日调用预算熔断:state/llm_api_usage.json 记录当日调用数/税token,
        超过 max_daily_calls 直接抛 RuntimeError,让当天剩余周期全部走安全
        降级路径(hold/skip),第二天零点(UTC)自动恢复——防止代码bug导致
        的死循环调用把账单烧穿。
      - 响应围栏剥离:见 _strip_code_fences()。
    API key 从环境变量 ANTHROPIC_API_KEY 读取,不进代码/配置文件。
    """

    def __init__(
        self,
        model: str,
        max_tokens: int = 4096,
        max_daily_calls: int = 600,
        timeout_seconds: float = 180.0,
        usage_path: Optional[Path] = None,
        base_url: Optional[str] = None,
    ):
        import anthropic  # 延迟导入:桥模式/本地测试不需要装 anthropic SDK

        self.model = model
        self.max_tokens = max_tokens
        self.max_daily_calls = max_daily_calls
        self.usage_path = usage_path or (STATE_ROOT / "llm_api_usage.json")
        # base_url:兼容Anthropic消息格式的第三方端点(用户2026-07-15选定
        # DeepSeek,base_url=https://api.deepseek.com/anthropic,模型
        # deepseek-v4-flash/pro)。None时走官方Anthropic端点。
        client_kwargs: dict = {"timeout": timeout_seconds, "max_retries": 2}
        if base_url:
            client_kwargs["base_url"] = base_url
        self._client = anthropic.Anthropic(**client_kwargs)
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

    def __call__(self, prompt: str) -> str:
        usage = self._load_usage()
        if usage["calls"] >= self.max_daily_calls:
            raise RuntimeError(
                f"llm_daily_budget_exhausted: {usage['calls']} calls today >= "
                f"max_daily_calls {self.max_daily_calls}; degrading to safe fallback "
                "until UTC midnight"
            )

        message = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(
            block.text for block in message.content if getattr(block, "type", None) == "text"
        )

        usage["calls"] += 1
        usage["input_tokens"] += getattr(message.usage, "input_tokens", 0)
        usage["output_tokens"] += getattr(message.usage, "output_tokens", 0)
        self._save_usage(usage)

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
