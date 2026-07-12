"""
tests/test_llm_bridge.py -- scripts/llm_bridge.py 的文件握手协议单测。

用一个后台线程模拟 AgentBridgeLLMClient 被 Trader/Researcher/Reflector 阻塞
调用的场景,主线程扮演"签入的 Claude Code agent"去读 pending request、
调 respond_to_pending() 回应它,断言阻塞调用能被正确唤醒并拿到响应内容。
"""
from __future__ import annotations

import importlib
import sys
import threading
import time
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))


@pytest.fixture
def bridge(tmp_path, monkeypatch):
    import llm_bridge as bridge_module
    importlib.reload(bridge_module)

    state_root = tmp_path / "state"
    monkeypatch.setattr(bridge_module, "STATE_ROOT", state_root)
    monkeypatch.setattr(bridge_module, "REQUEST_PATH", state_root / "llm_pending_request.json")
    return bridge_module


def test_request_response_roundtrip(bridge):
    client = bridge.AgentBridgeLLMClient(poll_seconds=0.05, timeout_seconds=10.0)
    result_holder = {}

    def call_client():
        result_holder["response"] = client("this is the prompt")

    thread = threading.Thread(target=call_client)
    thread.start()

    # Wait for the pending request to actually appear (bounded poll, not a fixed sleep guess).
    deadline = time.time() + 5.0
    pending = None
    while time.time() < deadline:
        pending = bridge.read_pending_request()
        if pending is not None:
            break
        time.sleep(0.02)

    assert pending is not None
    assert pending["prompt"] == "this is the prompt"

    bridge.respond_to_pending(pending["request_id"], "this is the response")
    thread.join(timeout=5.0)

    assert result_holder["response"] == "this is the response"
    assert not bridge.REQUEST_PATH.exists()


def test_timeout_raises_when_never_answered(bridge):
    client = bridge.AgentBridgeLLMClient(poll_seconds=0.02, timeout_seconds=0.15)
    with pytest.raises(TimeoutError):
        client("nobody will answer this")
    assert not bridge.REQUEST_PATH.exists()  # cleaned up on timeout, not left dangling


def test_read_pending_request_returns_none_when_nothing_pending(bridge):
    assert bridge.read_pending_request() is None


def test_multiple_sequential_calls_use_distinct_request_ids(bridge):
    client = bridge.AgentBridgeLLMClient(poll_seconds=0.02, timeout_seconds=5.0)
    seen_ids = []

    def responder():
        for _ in range(2):
            deadline = time.time() + 5.0
            pending = None
            while time.time() < deadline:
                pending = bridge.read_pending_request()
                if pending is not None:
                    break
                time.sleep(0.01)
            assert pending is not None
            seen_ids.append(pending["request_id"])
            bridge.respond_to_pending(pending["request_id"], f"response-to-{pending['request_id']}")

    thread = threading.Thread(target=responder)
    thread.start()

    r1 = client("first prompt")
    r2 = client("second prompt")
    thread.join(timeout=5.0)

    assert r1 == f"response-to-{seen_ids[0]}"
    assert r2 == f"response-to-{seen_ids[1]}"
    assert seen_ids[0] != seen_ids[1]
