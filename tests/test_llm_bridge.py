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


# ---------------------------------------------------------------------------
# 2026-07-15新增:API模式的两项无人值守防线(围栏剥离+每日预算熔断)。
# 不真调Anthropic API——AnthropicLLMClient需要SDK和key,这里只测纯函数
# _strip_code_fences和预算文件的读写判定逻辑。
# ---------------------------------------------------------------------------


def test_strip_code_fences_removes_json_fence():
    from llm_bridge import _strip_code_fences

    fenced = '```json\n[{"a": 1}]\n```'
    assert _strip_code_fences(fenced) == '[{"a": 1}]'


def test_strip_code_fences_removes_bare_fence():
    from llm_bridge import _strip_code_fences

    fenced = '```\n{"b": 2}\n```'
    assert _strip_code_fences(fenced) == '{"b": 2}'


def test_strip_code_fences_leaves_plain_text_alone():
    from llm_bridge import _strip_code_fences

    plain = '[{"a": 1}]'
    assert _strip_code_fences(plain) == plain


def test_strip_code_fences_leaves_unterminated_fence_alone():
    from llm_bridge import _strip_code_fences

    weird = '```json\n[{"a": 1}]'  # 只有开头围栏没有结尾——不动它,交给下游解析报错重试
    assert _strip_code_fences(weird) == weird


def test_strip_code_fences_does_not_touch_backticks_inside_text():
    from llm_bridge import _strip_code_fences

    inner = 'thesis mentions `code` but response is plain JSON: [1]'
    assert _strip_code_fences(inner) == inner


# ---------------------------------------------------------------------------
# 2026-07-15:Tavily搜索客户端(scripts/search_client.py)——离线测试:
# 响应解析、失败降级、月度保险丝。不发真实网络请求。
# ---------------------------------------------------------------------------


def test_tavily_client_parses_answer_and_results(tmp_path, monkeypatch):
    from search_client import TavilySearchClient

    class FakeResp:
        status_code = 200
        def raise_for_status(self): pass
        def json(self):
            return {"answer": "BTC下跌因宏观避险", "results": [
                {"title": "t1", "url": "http://x", "content": "c1"},
            ]}

    import search_client as sc
    monkeypatch.setattr(sc.requests, "post", lambda *a, **k: FakeResp())
    client = TavilySearchClient(api_key="k", usage_path=tmp_path / "u.json")
    out = client("btc why down")
    assert out[0]["content"] == "BTC下跌因宏观避险"
    assert out[1]["title"] == "t1"
    assert client._load_usage()["calls"] == 1


def test_tavily_client_network_error_returns_empty_and_no_usage(tmp_path, monkeypatch):
    from search_client import TavilySearchClient
    import search_client as sc

    def boom(*a, **k): raise RuntimeError("network down")
    monkeypatch.setattr(sc.requests, "post", boom)
    client = TavilySearchClient(api_key="k", usage_path=tmp_path / "u.json")
    assert client("q") == []
    assert client._load_usage()["calls"] == 0  # 失败调用不消耗额度计数


def test_tavily_monthly_fuse_blocks_calls(tmp_path, monkeypatch):
    from search_client import TavilySearchClient
    import search_client as sc

    called = {"n": 0}
    class FakeResp:
        def raise_for_status(self): pass
        def json(self): return {"results": []}
    def post(*a, **k):
        called["n"] += 1
        return FakeResp()
    monkeypatch.setattr(sc.requests, "post", post)
    client = TavilySearchClient(api_key="k", usage_path=tmp_path / "u.json", max_monthly_calls=1)
    client("q1")  # 消耗唯一额度
    assert client("q2") == []  # 保险丝生效
    assert called["n"] == 1  # 第二次根本没发请求


# ---------------------------------------------------------------------------
# 2026-07-16:OpenAIChatLLMClient(火山方舟豆包等OpenAI Chat Completions格式
# 供应商)——离线测试:mock requests.post,覆盖正常解析/重试/预算熔断/
# reasoning_content回退/围栏剥离。同一份 state/llm_api_usage.json 预算记账
# 逻辑(_DailyBudgetedLLMClient)也顺带被 AnthropicLLMClient 的既有测试覆盖,
# 这里只需要确认 OpenAIChatLLMClient 正确复用它,不需要重复测所有预算场景。
# ---------------------------------------------------------------------------


class _FakeResp:
    def __init__(self, payload, status_ok=True):
        self._payload = payload
        self._status_ok = status_ok

    def raise_for_status(self):
        if not self._status_ok:
            raise RuntimeError("simulated HTTP error")

    def json(self):
        return self._payload


def _make_openai_client(tmp_path, **overrides):
    from llm_bridge import OpenAIChatLLMClient

    kwargs = dict(
        model="Doubao-Seed-2.1-turbo",
        api_key="fake-key",
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        usage_path=tmp_path / "llm_api_usage.json",
        max_daily_calls=600,
        max_retries=2,
    )
    kwargs.update(overrides)
    return OpenAIChatLLMClient(**kwargs)


def test_openai_chat_client_parses_normal_response(tmp_path, monkeypatch):
    import llm_bridge as bridge_module

    calls = {"n": 0}

    def fake_post(url, headers=None, json=None, timeout=None):
        calls["n"] += 1
        assert url == "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
        assert headers["Authorization"] == "Bearer fake-key"
        assert json["model"] == "Doubao-Seed-2.1-turbo"
        assert json["messages"] == [{"role": "user", "content": "hello"}]
        return _FakeResp({
            "choices": [{"message": {"content": '[{"a": 1}]'}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        })

    monkeypatch.setattr(bridge_module.requests, "post", fake_post)
    client = _make_openai_client(tmp_path)
    result = client("hello")
    assert result == '[{"a": 1}]'
    assert calls["n"] == 1
    usage = client._load_usage()
    assert usage["calls"] == 1
    assert usage["input_tokens"] == 10
    assert usage["output_tokens"] == 5


def test_openai_chat_client_strips_code_fences(tmp_path, monkeypatch):
    import llm_bridge as bridge_module

    monkeypatch.setattr(
        bridge_module.requests, "post",
        lambda *a, **k: _FakeResp({"choices": [{"message": {"content": '```json\n[1,2]\n```'}}]}),
    )
    client = _make_openai_client(tmp_path)
    assert client("prompt") == "[1,2]"


def test_openai_chat_client_falls_back_to_reasoning_content_when_content_empty(tmp_path, monkeypatch, caplog):
    import llm_bridge as bridge_module

    monkeypatch.setattr(
        bridge_module.requests, "post",
        lambda *a, **k: _FakeResp({
            "choices": [{"message": {"content": "", "reasoning_content": '[{"b": 2}]'}}],
        }),
    )
    client = _make_openai_client(tmp_path)
    with caplog.at_level("WARNING"):
        result = client("prompt")
    assert result == '[{"b": 2}]'
    assert any("reasoning_content" in rec.message for rec in caplog.records)


def test_openai_chat_client_retries_then_succeeds(tmp_path, monkeypatch):
    import llm_bridge as bridge_module

    attempts = {"n": 0}

    def flaky_post(*a, **k):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RuntimeError("transient network error")
        return _FakeResp({"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr(bridge_module.requests, "post", flaky_post)
    client = _make_openai_client(tmp_path, max_retries=2)
    result = client("prompt")
    assert result == "ok"
    assert attempts["n"] == 3  # 第1次失败 + 2次重试(第2次重试成功)


def test_openai_chat_client_raises_after_exhausting_retries(tmp_path, monkeypatch):
    import llm_bridge as bridge_module

    attempts = {"n": 0}

    def always_fails(*a, **k):
        attempts["n"] += 1
        raise RuntimeError("network down")

    monkeypatch.setattr(bridge_module.requests, "post", always_fails)
    client = _make_openai_client(tmp_path, max_retries=2)
    with pytest.raises(RuntimeError):
        client("prompt")
    assert attempts["n"] == 3  # 首次 + 2次重试,全部失败后向上抛


def test_openai_chat_client_respects_daily_budget_fuse(tmp_path):
    client = _make_openai_client(tmp_path, max_daily_calls=1)
    usage_path = tmp_path / "llm_api_usage.json"
    import json as _json
    import time as _time
    usage_path.write_text(
        _json.dumps({"date": _time.strftime("%Y-%m-%d", _time.gmtime()), "calls": 1,
                     "input_tokens": 0, "output_tokens": 0}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="llm_daily_budget_exhausted"):
        client("prompt")


def test_openai_and_anthropic_clients_share_daily_budget_file(tmp_path, monkeypatch):
    """两个供应商共用同一份预算文件时,调用次数应该是累加的总量,不是
    各自独立的计数器(见 _DailyBudgetedLLMClient 的设计说明)。"""
    import llm_bridge as bridge_module

    class FakeUsage:
        input_tokens = 1
        output_tokens = 1

    class FakeMessage:
        content = []
        usage = FakeUsage()

    class FakeMessages:
        def create(self, **kwargs):
            return FakeMessage()

    class FakeAnthropicClient:
        def __init__(self, **kwargs):
            self.messages = FakeMessages()

    monkeypatch.setattr(bridge_module, "requests", bridge_module.requests)
    monkeypatch.setattr(
        bridge_module.requests, "post",
        lambda *a, **k: _FakeResp({"choices": [{"message": {"content": "ok"}}]}),
    )

    import types
    fake_anthropic_module = types.SimpleNamespace(Anthropic=FakeAnthropicClient)
    monkeypatch.setitem(sys.modules, "anthropic", fake_anthropic_module)

    shared_usage_path = tmp_path / "shared_usage.json"
    openai_client = _make_openai_client(tmp_path, usage_path=shared_usage_path, max_daily_calls=2)
    anthropic_client = bridge_module.AnthropicLLMClient(
        model="deepseek-v4-flash", usage_path=shared_usage_path, max_daily_calls=2,
    )

    openai_client("p1")
    assert openai_client._load_usage()["calls"] == 1
    # 第二次调用换成anthropic客户端,应该看到累计到2,而不是各自独立从0开始
    anthropic_client("p2")
    assert anthropic_client._load_usage()["calls"] == 2
    # 预算已经打满,第三次不管用哪个客户端都应该被熔断
    with pytest.raises(RuntimeError, match="llm_daily_budget_exhausted"):
        openai_client("p3")


# ---------------------------------------------------------------------------
# 2026-07-16:AnthropicLLMClient 的 auth 参数(api_key vs auth_token)——用户
# 提供的火山方舟Ark Agent Plan官方文档指出该端点要求标准 Authorization:
# Bearer 认证(SDK对应auth_token参数),而DeepSeek走x-api-key(SDK对应
# api_key参数),两者不能用同一套认证逻辑硬编码。这里只验证"传给底层
# anthropic.Anthropic(...)构造函数的到底是哪个kwarg",不真连网络。
# ---------------------------------------------------------------------------


def _install_fake_anthropic_module(monkeypatch):
    import types

    captured = {}

    class _FakeAnthropicClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.messages = types.SimpleNamespace(create=lambda **kw: None)

    fake_module = types.SimpleNamespace(Anthropic=_FakeAnthropicClient)
    monkeypatch.setitem(sys.modules, "anthropic", fake_module)
    return captured


def test_anthropic_client_defaults_to_api_key_auth(tmp_path, monkeypatch):
    from llm_bridge import AnthropicLLMClient

    captured = _install_fake_anthropic_module(monkeypatch)
    AnthropicLLMClient(model="deepseek-v4-flash", usage_path=tmp_path / "u.json", api_key="k-deepseek")

    assert captured.get("api_key") == "k-deepseek"
    assert "auth_token" not in captured


def test_anthropic_client_auth_token_mode_routes_to_bearer_kwarg(tmp_path, monkeypatch):
    from llm_bridge import AnthropicLLMClient

    captured = _install_fake_anthropic_module(monkeypatch)
    AnthropicLLMClient(
        model="doubao-seed-2.1-turbo", usage_path=tmp_path / "u.json",
        api_key="k-doubao", auth="auth_token",
    )

    assert captured.get("auth_token") == "k-doubao"
    assert "api_key" not in captured


def test_anthropic_client_rejects_unknown_auth_mode(tmp_path, monkeypatch):
    from llm_bridge import AnthropicLLMClient

    _install_fake_anthropic_module(monkeypatch)
    with pytest.raises(ValueError):
        AnthropicLLMClient(model="m", usage_path=tmp_path / "u.json", api_key="k", auth="bogus")
