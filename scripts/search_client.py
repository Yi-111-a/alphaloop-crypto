"""
scripts/search_client.py —— Tavily 搜索客户端(2026-07-15,用户选定)。

给 ASSET/strategy/researcher.py 预留的 search_client 接口提供第一个真实实现:
Callable[[str], list[dict]],查询字符串进,[{title, url, content}] 出。

为什么是 Tavily(用户调研后拍板):
  - 免费额度 1000次/月,无需绑卡;我们的用量是每小时1次分支轮换研究
    ≈ 720次/月,正好落在免费层内
  - 返回的是给LLM用的清洗过的正文(raw_content/content),不需要自己再抓网页
  - LangChain 默认搜索工具,接口稳定

设计约束:
  - key 从环境变量 TAVILY_API_KEY 读,不进代码/配置文件(与 ANTHROPIC_API_KEY
    同一条纪律);没有key时调用方(ignite.py)根本不会构造本客户端,
    Researcher 依旧走"无搜索源"的既有降级路径
  - 月度额度保险丝:state/search_api_usage.json 记录当月调用数,超过
    max_monthly_calls(默认950,给免费层留50余量)后直接返回空列表——
    Researcher 对空结果本来就有完整的降级行为,不会报错
  - 单次调用失败(网络/限流/4xx)返回空列表而不是抛异常——researcher.
    _run_searches 本来也会吞掉异常,这里再兜一层是让保险丝计数不被
    失败调用消耗
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_USAGE_PATH = PROJECT_ROOT / "state" / "search_api_usage.json"

_TAVILY_ENDPOINT = "https://api.tavily.com/search"


class TavilySearchClient:
    def __init__(
        self,
        api_key: str,
        max_results: int = 5,
        timeout_seconds: float = 20.0,
        max_monthly_calls: int = 950,
        usage_path: Optional[Path] = None,
    ):
        self.api_key = api_key
        self.max_results = max_results
        self.timeout_seconds = timeout_seconds
        self.max_monthly_calls = max_monthly_calls
        self.usage_path = usage_path or _USAGE_PATH

    # ------------------------------------------------------------------
    # 月度额度保险丝
    # ------------------------------------------------------------------

    def _load_usage(self) -> dict:
        month = time.strftime("%Y-%m", time.gmtime())
        if self.usage_path.exists():
            try:
                usage = json.loads(self.usage_path.read_text(encoding="utf-8"))
                if usage.get("month") == month:
                    return usage
            except (json.JSONDecodeError, OSError):
                pass
        return {"month": month, "calls": 0}

    def _save_usage(self, usage: dict) -> None:
        self.usage_path.parent.mkdir(parents=True, exist_ok=True)
        self.usage_path.write_text(json.dumps(usage, ensure_ascii=False), encoding="utf-8")

    # ------------------------------------------------------------------
    # 调用
    # ------------------------------------------------------------------

    def __call__(self, query: str) -> list[dict]:
        usage = self._load_usage()
        if usage["calls"] >= self.max_monthly_calls:
            return []  # 免费额度将尽,静默降级为"无检索结果",下月自动恢复

        try:
            resp = requests.post(
                _TAVILY_ENDPOINT,
                json={
                    "api_key": self.api_key,
                    "query": query,
                    "max_results": self.max_results,
                    "include_answer": True,
                    "search_depth": "basic",
                },
                timeout=self.timeout_seconds,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception:  # noqa: BLE001 -- 搜索抖动不应影响研究周期,降级为空结果
            return []

        usage["calls"] += 1
        self._save_usage(usage)

        results: list[dict] = []
        answer = data.get("answer")
        if isinstance(answer, str) and answer.strip():
            results.append({"title": "tavily_synthesized_answer", "url": "", "content": answer.strip()})
        for item in data.get("results", []) or []:
            if not isinstance(item, dict):
                continue
            results.append({
                "title": str(item.get("title", ""))[:200],
                "url": str(item.get("url", ""))[:300],
                "content": str(item.get("content", ""))[:1500],
            })
        return results
