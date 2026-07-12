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
