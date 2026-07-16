"""覆盖 scripts/ignite.py 里"多LLM供应商+分支级模型路由"这段
(2026-07-16用户要求:接入豆包/智谱GLM,让不同交易分支用不同的大脑)。

测试范围:
  - build_provider_clients(config):按 llm.api.providers 注册表构造客户端,
    缺环境变量的供应商被跳过,format路由到正确的类(AnthropicLLMClient /
    OpenAIChatLLMClient)。
  - assign_providers_to_roster(roster, available_providers):3供应商x7分支
    轮转分配的确定性、已有llm_provider的分支不被重新分配、幂等。
  - make_llm_resolver(available_clients, roster_loader):按名册查供应商,
    查不到/未配置时回退默认供应商。

完全离线:不真调任何API。format=anthropic的路径需要 anthropic SDK 才能
实例化 AnthropicLLMClient(本仓库运行环境未装该SDK),这里用
sys.modules注入一个假的 anthropic 模块,与 tests/test_llm_bridge.py 里
测预算共享的既有手法一致。
"""
from __future__ import annotations

import sys
import types

import pytest

import scripts.ignite as ignite
from llm_bridge import AnthropicLLMClient, OpenAIChatLLMClient


@pytest.fixture(autouse=True)
def _isolate_state_root(tmp_path, monkeypatch):
    """所有测试都不能碰真实项目的 state/ 目录(共享预算文件/名册文件)。"""
    state_root = tmp_path / "state"
    state_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(ignite, "STATE_ROOT", state_root)
    monkeypatch.setattr(ignite, "TOURNAMENT_ROSTER_PATH", state_root / "tactic_tournament_roster.json")
    return state_root


@pytest.fixture
def fake_anthropic_sdk(monkeypatch):
    """anthropic SDK 在这个环境里没装——build_provider_clients对
    format=anthropic的供应商会真的执行 `import anthropic`,这里注入一个
    最小的假模块满足这一步,不涉及任何真实网络调用。"""

    class _FakeAnthropicClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.messages = types.SimpleNamespace(create=lambda **kw: None)

    fake_module = types.SimpleNamespace(Anthropic=_FakeAnthropicClient)
    monkeypatch.setitem(sys.modules, "anthropic", fake_module)
    return fake_module


# ---------------------------------------------------------------------------
# build_provider_clients
# ---------------------------------------------------------------------------

_MULTI_PROVIDER_CONFIG = {
    "llm": {
        "mode": "api",
        "api": {
            "max_daily_calls": 600,
            "providers": {
                "deepseek": {
                    "format": "anthropic",
                    "base_url": "https://api.deepseek.com/anthropic",
                    "api_key_env": "TEST_DEEPSEEK_KEY",
                    # auth省略 -> 默认"api_key"(x-api-key头)
                    "trader_model": "deepseek-v4-flash",
                    "deep_model": "deepseek-v4-pro",
                },
                "glm": {
                    "format": "anthropic",
                    "base_url": "https://open.bigmodel.cn/api/anthropic",
                    "api_key_env": "TEST_GLM_KEY",
                    "auth": "auth_token",  # 官方文档:ANTHROPIC_AUTH_TOKEN风格
                    "trader_model": "glm-5-2-260617",
                    "deep_model": "glm-5-2-260617",
                },
                "doubao": {
                    # 火山方舟Ark Agent Plan专属端点(.../api/plan):Anthropic
                    # 消息格式 + auth_token(Authorization: Bearer),不是
                    # OpenAI Chat Completions格式——这是用户2026-07-16提供
                    # 官方文档后的修正,与最初假设的openai格式不同。
                    "format": "anthropic",
                    "base_url": "https://ark.cn-beijing.volces.com/api/plan",
                    "api_key_env": "TEST_DOUBAO_KEY",
                    "auth": "auth_token",
                    "trader_model": "doubao-seed-2.1-turbo",
                    "deep_model": "doubao-seed-2.1-turbo",
                },
                "qwen": {
                    # 假设的未来供应商,真正走OpenAI Chat Completions格式,
                    # 用来验证format=openai仍然正确路由到OpenAIChatLLMClient
                    # (OpenAIChatLLMClient本身是通用能力,不因doubao改用
                    # anthropic格式而被移除)。
                    "format": "openai",
                    "base_url": "https://example-qwen-endpoint.com/v1",
                    "api_key_env": "TEST_QWEN_KEY",
                    "trader_model": "qwen-max",
                    "deep_model": "qwen-max",
                },
                "mystery": {
                    "format": "some-unknown-format",
                    "base_url": "https://example.com",
                    "api_key_env": "TEST_MYSTERY_KEY",
                    "trader_model": "m1",
                    "deep_model": "m1",
                },
            },
        },
    },
}


def test_build_provider_clients_bridge_mode_returns_single_bridge_entry():
    clients = ignite.build_provider_clients({"llm": {"mode": "bridge"}})
    assert set(clients.keys()) == {"bridge"}
    routine, deep = clients["bridge"]
    assert routine is deep
    assert isinstance(routine, ignite.AgentBridgeLLMClient)


def test_build_provider_clients_skips_providers_without_api_key(monkeypatch, fake_anthropic_sdk):
    monkeypatch.setenv("TEST_DEEPSEEK_KEY", "real-deepseek-key")
    monkeypatch.setenv("TEST_GLM_KEY", "")  # 空字符串视同未配置
    monkeypatch.delenv("TEST_DOUBAO_KEY", raising=False)  # 环境变量根本不存在
    monkeypatch.delenv("TEST_QWEN_KEY", raising=False)
    monkeypatch.setenv("TEST_MYSTERY_KEY", "some-key")  # 有key但format未知

    clients = ignite.build_provider_clients(_MULTI_PROVIDER_CONFIG)

    # 只有配了真实key且format已知的deepseek被构造;glm(空key)、
    # doubao/qwen(缺key)、mystery(未知format)全部跳过。
    assert set(clients.keys()) == {"deepseek"}


def test_build_provider_clients_routes_format_to_correct_class(monkeypatch, fake_anthropic_sdk):
    monkeypatch.setenv("TEST_DEEPSEEK_KEY", "real-deepseek-key")
    monkeypatch.setenv("TEST_GLM_KEY", "real-glm-key")
    monkeypatch.setenv("TEST_DOUBAO_KEY", "real-doubao-key")
    monkeypatch.setenv("TEST_QWEN_KEY", "real-qwen-key")
    monkeypatch.delenv("TEST_MYSTERY_KEY", raising=False)

    clients = ignite.build_provider_clients(_MULTI_PROVIDER_CONFIG)

    assert set(clients.keys()) == {"deepseek", "glm", "doubao", "qwen"}
    # deepseek/glm/doubao 都是Anthropic消息格式(doubao走Ark Agent Plan
    # 专属端点,也是anthropic格式,见_MULTI_PROVIDER_CONFIG注释)
    for name in ("deepseek", "glm", "doubao"):
        routine, deep = clients[name]
        assert isinstance(routine, AnthropicLLMClient)
        assert isinstance(deep, AnthropicLLMClient)
    assert clients["doubao"][0].model == "doubao-seed-2.1-turbo"
    # qwen是唯一真正走OpenAI Chat Completions格式的供应商
    routine, deep = clients["qwen"]
    assert isinstance(routine, OpenAIChatLLMClient)
    assert isinstance(deep, OpenAIChatLLMClient)
    assert routine.model == "qwen-max"


def test_build_provider_clients_routes_auth_mode_per_provider(monkeypatch, fake_anthropic_sdk):
    """deepseek省略auth字段应该走默认的api_key(x-api-key);glm/doubao
    显式声明auth: auth_token应该走Authorization: Bearer(SDK的auth_token
    参数)——这是用户提供火山方舟Ark Agent Plan官方文档后新增的真实差异,
    不能对所有anthropic格式供应商用同一套认证逻辑硬编码。"""
    monkeypatch.setenv("TEST_DEEPSEEK_KEY", "real-deepseek-key")
    monkeypatch.setenv("TEST_GLM_KEY", "real-glm-key")
    monkeypatch.setenv("TEST_DOUBAO_KEY", "real-doubao-key")
    monkeypatch.delenv("TEST_QWEN_KEY", raising=False)
    monkeypatch.delenv("TEST_MYSTERY_KEY", raising=False)

    clients = ignite.build_provider_clients(_MULTI_PROVIDER_CONFIG)

    deepseek_routine = clients["deepseek"][0]
    assert deepseek_routine._client.kwargs.get("api_key") == "real-deepseek-key"
    assert "auth_token" not in deepseek_routine._client.kwargs

    for name, expected_key in (("glm", "real-glm-key"), ("doubao", "real-doubao-key")):
        routine = clients[name][0]
        assert routine._client.kwargs.get("auth_token") == expected_key
        assert "api_key" not in routine._client.kwargs


_SHARED_KEY_CONFIG = {
    "llm": {
        "mode": "api",
        "api": {
            "max_daily_calls": 600,
            "providers": {
                # 火山方舟Ark Agent Plan最终形态(2026-07-16二次修正):
                # 多个"大脑"共享同一份订阅端点+同一个api_key_env,只有模型
                # 名不同——这与deepseek(独立环境变量)形成对比。
                "ark-doubao": {
                    "format": "anthropic", "base_url": "https://ark.example/plan",
                    "api_key_env": "TEST_ARK_KEY", "auth": "auth_token",
                    "trader_model": "doubao-seed-2.0-lite", "deep_model": "doubao-seed-2.0-pro",
                },
                "ark-glm": {
                    "format": "anthropic", "base_url": "https://ark.example/plan",
                    "api_key_env": "TEST_ARK_KEY", "auth": "auth_token",
                    "trader_model": "glm-5.2", "deep_model": "glm-5.2",
                },
                "ark-kimi": {
                    "format": "anthropic", "base_url": "https://ark.example/plan",
                    "api_key_env": "TEST_ARK_KEY", "auth": "auth_token",
                    "trader_model": "kimi-k2.6", "deep_model": "kimi-k2.7-code",
                },
            },
        },
    },
}


def test_build_provider_clients_shared_api_key_env_builds_all_providers(monkeypatch, fake_anthropic_sdk):
    monkeypatch.setenv("TEST_ARK_KEY", "shared-ark-key")

    clients = ignite.build_provider_clients(_SHARED_KEY_CONFIG)

    assert set(clients.keys()) == {"ark-doubao", "ark-glm", "ark-kimi"}
    for routine, _deep in clients.values():
        assert routine._client.kwargs.get("auth_token") == "shared-ark-key"


def test_build_provider_clients_shared_api_key_env_missing_skips_all_and_prints_once(monkeypatch, capsys):
    monkeypatch.delenv("TEST_ARK_KEY", raising=False)

    clients = ignite.build_provider_clients(_SHARED_KEY_CONFIG)

    assert clients == {}  # 全部三个共享同一个缺失的环境变量,一起跳过
    captured = capsys.readouterr()
    # 只应该看到一条关于TEST_ARK_KEY缺失的提示,不是三条(每个供应商一条)
    assert captured.out.count("TEST_ARK_KEY") == 1


# ---------------------------------------------------------------------------
# evo_rotation_pool
# ---------------------------------------------------------------------------


def test_evo_rotation_pool_excludes_default_provider_only():
    available = {
        "deepseek": (_fake_client_placeholder(), _fake_client_placeholder()),
        "ark-doubao": (_fake_client_placeholder(), _fake_client_placeholder()),
        "ark-deepseek": (_fake_client_placeholder(), _fake_client_placeholder()),
    }
    pool = ignite.evo_rotation_pool(available)
    # "deepseek"(main专属+全局兜底)被排除,但"ark-deepseek"(同模型不同
    # 供应商身份)必须留在池子里——排除规则只精确匹配供应商名字。
    assert set(pool) == {"ark-doubao", "ark-deepseek"}


def test_evo_rotation_pool_empty_when_only_default_provider_available():
    available = {"deepseek": (_fake_client_placeholder(), _fake_client_placeholder())}
    assert ignite.evo_rotation_pool(available) == []


def _fake_client_placeholder():
    return lambda prompt: "unused"


def test_build_provider_clients_shares_single_usage_path(monkeypatch, fake_anthropic_sdk, tmp_path):
    monkeypatch.setenv("TEST_DEEPSEEK_KEY", "k1")
    monkeypatch.setenv("TEST_GLM_KEY", "k2")
    monkeypatch.setenv("TEST_DOUBAO_KEY", "k3")
    monkeypatch.setenv("TEST_QWEN_KEY", "k4")
    monkeypatch.delenv("TEST_MYSTERY_KEY", raising=False)

    clients = ignite.build_provider_clients(_MULTI_PROVIDER_CONFIG)
    expected_path = ignite.STATE_ROOT / "llm_api_usage.json"
    for routine, deep in clients.values():
        assert routine.usage_path == expected_path
        assert deep.usage_path == expected_path


# ---------------------------------------------------------------------------
# assign_providers_to_roster
# ---------------------------------------------------------------------------


def _make_roster(branch_names, status="active", with_provider=None):
    roster = {}
    for name in branch_names:
        entry = {"tactics": f"tactics for {name}", "status": status, "created_ms": 0}
        if with_provider and name in with_provider:
            entry["llm_provider"] = with_provider[name]
        roster[name] = entry
    return roster


def test_assign_providers_round_robin_is_deterministic():
    branches = [f"evo/branch-{i}" for i in range(7)]
    roster = _make_roster(branches)
    # 传入顺序故意打乱,函数内部应该自己排序再轮转
    providers = ["glm", "doubao", "deepseek"]

    result = ignite.assign_providers_to_roster(roster, providers)

    sorted_branches = sorted(branches)
    sorted_providers = sorted(providers)  # ["deepseek", "doubao", "glm"]
    expected = {
        branch: sorted_providers[i % len(sorted_providers)]
        for i, branch in enumerate(sorted_branches)
    }
    actual = {branch: meta["llm_provider"] for branch, meta in result.items()}
    assert actual == expected

    # 再跑一次(不同调用顺序的providers列表),分配结果必须完全一样
    result_again = ignite.assign_providers_to_roster(
        _make_roster(branches), list(reversed(providers))
    )
    actual_again = {branch: meta["llm_provider"] for branch, meta in result_again.items()}
    assert actual_again == expected


def test_assign_providers_does_not_overwrite_existing_assignment():
    branches = ["evo/a", "evo/b", "evo/c"]
    roster = _make_roster(branches, with_provider={"evo/a": "glm"})

    result = ignite.assign_providers_to_roster(roster, ["deepseek", "glm", "doubao"])

    assert result["evo/a"]["llm_provider"] == "glm"  # 保持不变
    # b/c 是唯二需要分配的,按排序后的分支名+供应商轮转
    assert result["evo/b"]["llm_provider"] == sorted(["deepseek", "glm", "doubao"])[0]
    assert result["evo/c"]["llm_provider"] == sorted(["deepseek", "glm", "doubao"])[1]


def test_assign_providers_is_idempotent(monkeypatch):
    branches = ["evo/a", "evo/b"]
    roster = _make_roster(branches)

    save_calls = {"n": 0}
    original_save = ignite.save_tournament_roster

    def counting_save(r):
        save_calls["n"] += 1
        return original_save(r)

    monkeypatch.setattr(ignite, "save_tournament_roster", counting_save)

    first = ignite.assign_providers_to_roster(roster, ["deepseek", "glm"])
    assert save_calls["n"] == 1
    second = ignite.assign_providers_to_roster(first, ["deepseek", "glm"])
    assert save_calls["n"] == 1  # 第二次没有任何未分配分支,不应该再写盘
    assert second == first


def test_assign_providers_ignores_non_active_branches():
    roster = _make_roster(["evo/dead"], status="failed")
    result = ignite.assign_providers_to_roster(roster, ["deepseek", "glm"])
    assert "llm_provider" not in result["evo/dead"]


def test_assign_providers_noop_when_no_providers_available():
    roster = _make_roster(["evo/a"])
    result = ignite.assign_providers_to_roster(roster, [])
    assert "llm_provider" not in result["evo/a"]


# ---------------------------------------------------------------------------
# make_llm_resolver
# ---------------------------------------------------------------------------


def _fake_client(tag):
    def _call(prompt):
        return f"{tag}:{prompt}"
    _call.tag = tag
    return _call


def test_resolver_routes_branch_to_roster_specified_provider():
    available = {
        "deepseek": (_fake_client("deepseek-routine"), _fake_client("deepseek-deep")),
        "glm": (_fake_client("glm-routine"), _fake_client("glm-deep")),
    }
    roster = {"evo/x": {"llm_provider": "glm", "status": "active"}}
    resolver = ignite.make_llm_resolver(available, lambda: roster)

    resolved = resolver("evo/x")
    assert resolved.tag == "glm-routine"


def test_resolver_falls_back_to_default_when_branch_not_in_roster_or_unassigned():
    available = {
        "deepseek": (_fake_client("deepseek-routine"), _fake_client("deepseek-deep")),
        "glm": (_fake_client("glm-routine"), _fake_client("glm-deep")),
    }
    roster = {"evo/unassigned": {"status": "active"}}  # 没有 llm_provider 字段
    resolver = ignite.make_llm_resolver(available, lambda: roster)

    assert resolver("evo/unassigned").tag == "deepseek-routine"
    assert resolver("evo/not-in-roster-at-all").tag == "deepseek-routine"


def test_resolver_falls_back_to_default_when_provider_not_configured():
    available = {"deepseek": (_fake_client("deepseek-routine"), _fake_client("deepseek-deep"))}
    roster = {"evo/x": {"llm_provider": "doubao", "status": "active"}}  # doubao没配key,不在available里
    resolver = ignite.make_llm_resolver(available, lambda: roster)

    assert resolver("evo/x").tag == "deepseek-routine"


def test_resolver_main_branch_always_uses_default_provider_ignoring_roster():
    available = {
        "deepseek": (_fake_client("deepseek-routine"), _fake_client("deepseek-deep")),
        "glm": (_fake_client("glm-routine"), _fake_client("glm-deep")),
    }
    # 即使有人手工在roster里给main塞了个llm_provider,main也不应该采用它
    roster = {"main": {"llm_provider": "glm", "status": "active"}}
    resolver = ignite.make_llm_resolver(available, lambda: roster)

    assert resolver("main").tag == "deepseek-routine"


def test_resolver_default_falls_back_to_first_sorted_provider_when_deepseek_unavailable():
    available = {"glm": (_fake_client("glm-routine"), _fake_client("glm-deep")),
                 "doubao": (_fake_client("doubao-routine"), _fake_client("doubao-deep"))}
    resolver = ignite.make_llm_resolver(available, lambda: {})
    # deepseek不在available里,退到排序后的第一个:doubao < glm
    assert resolver("main").tag == "doubao-routine"


def test_resolver_returns_none_when_no_providers_available_at_all():
    resolver = ignite.make_llm_resolver({}, lambda: {})
    assert resolver("main") is None
    assert resolver("evo/x") is None
