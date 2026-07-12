"""
ASSET/memory/engine.py —— 分层记忆引擎(移植 FinMem 思路，§3.4)。

ASSET 区代码(策略agent自由读写区，可被agent自我进化，见§3.5)。存放在 ASSET 而
非 LOCKED 是因为 spec §3.5 允许策略agent逐步进化自己的记忆使用方式 —— 但下面
这条"时间边界纪律"是不可协商的硬约束，不随进化改变：

    TIME-BOUNDARY RULE(与 M1 的爆仓插针测试同等优先级)
    ----------------------------------------------------
    retrieve(query, query_ts, ...) 对"现在"的唯一认知来源是调用方显式传入的
    query_ts 参数。本模块的检索路径里 **不允许** 出现任何墙钟时间调用
    (time.time() / datetime.now() / datetime.utcnow() / 等价物)——全文件搜索
    过，零处使用，仅在本注释里作为"禁止对象"提及。这不是风格洁癖：AlphaLoop
    的回放/重跑必须是确定性的(同一份历史数据，不管你在哪一天点"运行"，得到的
    记忆检索结果必须一致)；更重要的是，这条纪律直接堵死了"未来数据泄漏进当下
    决策"这一类信息泄漏 —— 一条 ts 晚于 query_ts 的记忆记录，无论内容和查询语
    义多么完美匹配，都必须被当作不存在，在 retrieve() 里作为最早、无条件的一
    步过滤掉(见 retrieve() 实现，在打分和 top_k 截断之前)。

    这类 bug 的现实场景：一次把研究笔记批量导入记忆库的脚本，时间戳字段填错
    (比如误用了导入时刻而不是笔记本身的观察时刻)，就会让"答案"提前泄漏进决策
    ——不会报错、不会崩溃，只会让 agent 的决策看起来莫名其妙地"聪明"，直到有人
    回头审计才会发现。THE TIME-TRAVEL INJECTION TEST(见 tests/test_memory_engine.py)
    专门针对这个场景构造，必须常绿。

边界约定：record.ts == query_ts 视为"可见"(决策发生的那一刻，同一时刻写入的
记忆已经可用)；只有 record.ts > query_ts(严格晚于)才不可见。spec 原文正是
"ts > query_ts"（严格大于）不可见，因此这里的 SQL 过滤条件是 `ts <= query_ts`。

检索得分公式(§3.4，原文照搬)：
    score = semantic_relevance(query, record.content) × exp(-Δt / τ) × importance
    Δt = query_ts - record.ts，换算成"天"（与 MEMORY_TIME_CONSTANT_DAYS 的单位一致）。
    L3 记录 τ=None → decay 恒为 1，不随时间衰减(§3.4："被证伪的教训、经
    过多次验证的规律"不衰减)。

    术语勘误(spec v1.3，产品侧裁决，本模块公式未改动，仅名字改动)：这个公式
    exp(-Δt/τ) 是"e-折时间"(时间常数 τ)的标准写法，Δt=τ 时衰减到 exp(-1)≈0.368，
    不是严格物理意义上的"半衰期"(那需要 0.5**(Δt/half_life) 或等价的
    exp(-ln(2)·Δt/half_life)，Δt=half_life 时精确得到 0.5)。产品侧的裁决是：
    L1/L2 的具体取值(3天/30天)本来就是拍的经验参数，0.368 与 0.5 的差异落在
    参数任意性之内，且 exp(-Δt/τ) 是 FinMem 一系的标准写法——因此公式保持
    不动，只把此前"半衰期"这个误导性命名改成"时间常数(τ)/e-折时间"。本文件
    对应的测试 test_memory_decay_l1 断言的是 exp(-1) 这个精确值，不是"约等于
    0.5"，与此裁决一致。

返回类型的选择(供 Trader 等下游消费者对齐)：
    write()    -> MemoryRecord                       (与 LOCKED.schemas 完全一致)
    retrieve() -> list[tuple[MemoryRecord, float]]    (record, score)，按 score
                  降序排列，长度 <= top_k。选择返回 (record, score) 而不是裸
                  record 列表，是为了让"检索得分随时间衰减"这个 M2 验收标准
                  在调用方也能被直接断言，不需要重新计算一遍。
                  ASSET/strategy/trader.py 已经用 duck-typing 兼容了这个形状
                  (retrieve 结果 item 是 tuple 时取 item[0].content)。
    另外提供 score_one(record, query, query_ts, ...) 作为独立可测试的打分入口，
    也是 retrieve() 内部实际调用的同一份逻辑，避免"测试断言的公式"和"实现里
    真正跑的公式"出现两份互相漂移的实现。
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import uuid
from pathlib import Path
from typing import Callable, Optional

from LOCKED.schemas import MEMORY_TIME_CONSTANT_DAYS, MemoryLayer, MemoryRecord

_DEFAULT_DB_PATH = Path(__file__).resolve().parent / "memory.db"
_DEFAULT_EMBEDDING_DIM = 256
_MS_PER_DAY = 86_400_000


# ---------------------------------------------------------------------------
# Embedders
# ---------------------------------------------------------------------------
#
# 默认 embedder 是一个确定性、纯本地、零第三方依赖的 hashing-trick
# bag-of-words 向量化器 —— 不是因为它在语义质量上比 sentence-transformers 好
# (显然不是)，而是因为它保证了：
#   1. 离线可跑，import 时不触发任何网络请求或模型下载；
#   2. 同一输入在任何机器、任何时间跑，输出严格一致(用 hashlib.sha256 而不是
#      内置 hash()，因为内置 hash() 对 str 默认加了每进程随机种子(PYTHONHASHSEED)，
#      不满足"确定性"要求)；
#   3. 测试套件因此能在几毫秒内跑完，不依赖网络或一次性模型下载是否成功。
#
# sentence-transformers(all-MiniLM-L6-v2)在本沙箱环境里实测是可以装、可以跑的
# (pip install 几秒，模型首次下载约 48s，之后走本地缓存)，作为可选的更优
# embedder 提供在 build_sentence_transformer_embedder()，通过依赖注入
# (MemoryStore(embedder=...)) 使用。但它 **不是默认值** —— 默认值必须是那个
# 不依赖"重型 ML 库恰好装成功"的确定性 fallback，否则同一份代码在不同机器上
# 会因为一个可选依赖装没装成功而产生不同的检索行为，这本身就是另一种意义上的
# "不确定性泄漏"。


def hashing_embedder(text: str, dim: int = _DEFAULT_EMBEDDING_DIM) -> list[float]:
    """确定性、离线的 hashing-trick 向量化，作为默认 embedder。

    对文本分词(按 Unicode 词字符切分，中英文都覆盖)，每个 token 用
    sha256 摘要确定性地映射到 [0, dim) 的一个下标，并用摘要的另一个字节确定
    +1/-1 的符号(标准 hashing-trick 手法，用符号位抵消哈希碰撞造成的系统性
    偏置)，最后做 L2 归一化，使得 cosine similarity 是良定义的。
    """
    vec = [0.0] * dim
    tokens = re.findall(r"\w+", (text or "").lower(), flags=re.UNICODE)
    for tok in tokens:
        digest = hashlib.sha256(tok.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if (digest[4] & 1) == 0 else -1.0
        vec[idx] += sign
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0.0:
        vec = [v / norm for v in vec]
    return vec


def build_sentence_transformer_embedder(
    model_name: str = "all-MiniLM-L6-v2",
) -> Callable[[str], list[float]]:
    """可选的真实语义 embedder，基于 sentence-transformers 本地模型(§3.4 建议)。

    懒加载：只有实际调用这个工厂函数时才会 import sentence_transformers /
    下载模型，模块顶层 import ASSET.memory.engine 本身不会触发任何网络行为。
    不作为 MemoryStore 的默认 embedder，见上面 hashing_embedder 的说明。
    """
    from sentence_transformers import SentenceTransformer  # optional dependency

    model = SentenceTransformer(model_name)

    def _embed(text: str) -> list[float]:
        vector = model.encode(text or "", normalize_embeddings=True)
        return [float(x) for x in vector]

    return _embed


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    dot = sum(a[i] * b[i] for i in range(n))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _decay_factor(delta_t_days: float, time_constant_days: Optional[float]) -> float:
    """exp(-Δt/τ)；τ=None(L3永久层) 恒返回 1.0，不衰减。命名说明见模块顶部
    docstring 的术语勘误(spec v1.3):τ 是时间常数/e-折时间,不是半衰期。

    delta_t_days 在正常调用路径里永远 >= 0，因为 retrieve()/score_one() 已经
    把 record.ts > query_ts 的记录在更早的一步过滤/拦截掉了 —— 这里不做"未来
    记录"的特殊处理，是因为按设计根本不会有未来记录走到这一步。
    """
    if time_constant_days is None:
        return 1.0
    return math.exp(-delta_t_days / time_constant_days)


class MemoryStore:
    """分层记忆库(§3.4)，sqlite 存储 + 可插拔 embedder。"""

    def __init__(
        self,
        db_path: "str | Path | None" = None,
        embedder: Optional[Callable[[str], list[float]]] = None,
    ):
        self.db_path = Path(db_path) if db_path is not None else _DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._init_schema()
        self.embedder: Callable[[str], list[float]] = embedder or hashing_embedder

    def _init_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_records (
                id TEXT PRIMARY KEY,
                ts INTEGER NOT NULL,
                layer TEXT NOT NULL,
                content TEXT NOT NULL,
                importance REAL NOT NULL DEFAULT 1.0,
                embedding TEXT
            )
            """
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_ts ON memory_records(ts)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_layer ON memory_records(layer)")
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "MemoryStore":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # ------------------------------------------------------------------
    # write
    # ------------------------------------------------------------------

    def write(
        self,
        content: str,
        ts: int,
        layer: MemoryLayer,
        importance: float = 1.0,
    ) -> MemoryRecord:
        if layer not in MEMORY_TIME_CONSTANT_DAYS:
            raise ValueError(f"unknown memory layer: {layer!r}, expected one of {sorted(MEMORY_TIME_CONSTANT_DAYS)}")

        record_id = uuid.uuid4().hex
        embedding = self.embedder(content)
        self._conn.execute(
            "INSERT INTO memory_records (id, ts, layer, content, importance, embedding) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (record_id, ts, layer, content, importance, json.dumps(embedding)),
        )
        self._conn.commit()
        return MemoryRecord(
            id=record_id,
            ts=ts,
            layer=layer,
            content=content,
            importance=importance,
            embedding=embedding,
        )

    # ------------------------------------------------------------------
    # scoring
    # ------------------------------------------------------------------

    def score_one(
        self,
        record: MemoryRecord,
        query: str,
        query_ts: int,
        query_embedding: Optional[list[float]] = None,
    ) -> Optional[float]:
        """对单条 record 相对 (query, query_ts) 打分。

        独立于 retrieve() 暴露出来，方便测试直接断言 §3.4 的打分公式，而不用
        每次都绕经 sqlite 往返。retrieve() 内部调用的正是这同一个函数,所以
        "测试断言的公式"与"实现里真正跑的公式"不会出现两份实现分叉的风险。

        时间边界：record.ts > query_ts(严格晚于,即这条记忆在决策发生的那一刻
        还没被写入)时返回 None,代表"不可见/视为不存在"。record.ts == query_ts
        视为可见。这里的检查是防御性冗余 —— retrieve() 已经在 SQL 层面用
        `WHERE ts <= query_ts` 早一步过滤掉了未来记录,这里再挡一层是为了让
        score_one() 单独调用时(比如测试或未来别的调用方)也不会意外算出一个
        "本不该存在"的分数。
        """
        if record.ts > query_ts:
            return None

        q_emb = query_embedding if query_embedding is not None else self.embedder(query)
        r_emb = record.embedding if record.embedding is not None else self.embedder(record.content)
        relevance = _cosine_similarity(q_emb, r_emb)

        time_constant = MEMORY_TIME_CONSTANT_DAYS.get(record.layer)
        delta_t_days = (query_ts - record.ts) / _MS_PER_DAY
        decay = _decay_factor(delta_t_days, time_constant)

        return relevance * decay * record.importance

    # ------------------------------------------------------------------
    # retrieve
    # ------------------------------------------------------------------

    def _row_to_record(self, row: sqlite3.Row) -> MemoryRecord:
        raw_embedding = row["embedding"]
        embedding = json.loads(raw_embedding) if raw_embedding is not None else None
        return MemoryRecord(
            id=row["id"],
            ts=row["ts"],
            layer=row["layer"],
            content=row["content"],
            importance=row["importance"],
            embedding=embedding,
        )

    def retrieve(
        self,
        query: str,
        query_ts: int,
        top_k: int = 5,
        layers: Optional[list[str]] = None,
    ) -> list[tuple[MemoryRecord, float]]:
        """检索 top_k 条记忆，按 §3.4 公式打分降序返回。

        返回类型: list[tuple[MemoryRecord, float]] —— (record, score)，score
        降序，长度 <= top_k。这是本模块对外的契约形状,下游(如 Trader)按此消费。

        时间边界(THE non-negotiable 部分): 第一步、无条件地把
        record.ts > query_ts 的记录从候选集里剔除 —— 在任何打分、排序、
        top_k 截断之前。这一步只依赖调用方传入的 query_ts,函数体内没有任何
        墙钟时间调用(time.time()/datetime.now()/datetime.utcnow()均未出现,
        也不会出现)。
        """
        sql = "SELECT * FROM memory_records WHERE ts <= ?"
        params: list = [query_ts]
        if layers:
            placeholders = ",".join("?" for _ in layers)
            sql += f" AND layer IN ({placeholders})"
            params.extend(layers)

        rows = self._conn.execute(sql, params).fetchall()

        query_embedding = self.embedder(query)

        scored: list[tuple[MemoryRecord, float]] = []
        for row in rows:
            record = self._row_to_record(row)
            score = self.score_one(record, query, query_ts, query_embedding=query_embedding)
            if score is None:
                # 防御性冗余：SQL 已经用 ts <= query_ts 过滤过了，这里不应该
                # 再遇到未来记录；如果真的遇到了，宁可丢弃也不能让它泄漏出去。
                continue
            scored.append((record, score))

        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]
