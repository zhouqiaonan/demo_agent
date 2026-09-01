"""ChromaStore 性能/压力测试 —— 写入 QPS 与自监督召回率。

本模块使用确定性的 :class:`HashEmbedding`（维度 64）与真实 ChromaDB
（``chromadb.Client()`` 进程内内存存储）构造 1000 条文档，测量
:meth:`ChromaStore.add_documents` 的插入吞吐（QPS）与自监督召回率：

- **QPS**：``1000 / 插入耗时秒``，仅打印、不做硬断言（避免 CI 机器性能差异
  导致 flaky）。
- **召回**：抽样 50 条（idx 为 0, 20, 40, ..., 980），对每条做 ``query``，
  断言目标 id 出现在 top-5 中；``recall = 命中数 / 50``。自监督场景下
  理论上应全命中，故 ``assert recall == 1.0``。

本模块顶部通过 ``pytest.importorskip("chromadb")`` 保护：chromadb 未安装时
整个模块会被优雅跳过（skipped）。
"""

from __future__ import annotations

import time
from uuid import uuid4

import pytest

# 未安装 chromadb 时跳过整个模块，而非报错。
chromadb = pytest.importorskip("chromadb")

from tests.fakes import HashEmbedding
from vector_store.chroma_store import ChromaStore
from vector_store.document import Document


# ---------------------------------------------------------------------------
# 标记：整个模块均为性能测试
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

_DOC_COUNT: int = 1000
"""性能测试的文档总数。"""

_DIMENSION: int = 64
"""嵌入向量的维度。"""

_SAMPLE_STEP: int = 20
"""召回抽样的步长（1000 条中每 20 条抽 1 条，共 50 条）。"""


def _build_documents(n: int) -> list[Document]:
    """构造 n 条确定性的性能测试文档。

    Args:
        n: 文档数量。

    Returns:
        文本为 ``性能测试文档 {i} 号``、id 为 ``perf-{i}``、
        metadata 为 ``{"idx": i}`` 的文档列表。
    """
    return [
        Document(
            text=f"性能测试文档 {i} 号",
            metadata={"idx": i},
            id=f"perf-{i}",
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# 性能测试
# ---------------------------------------------------------------------------


def test_add_documents_qps_and_recall() -> None:
    """测量插入 QPS 与自监督召回率。

    打印 QPS 与 recall 供报告；QPS 不做硬断言，recall 断言为 1.0（自监督
    场景下理论上应全命中）。
    """
    embedding = HashEmbedding(dimension=_DIMENSION)
    client = chromadb.Client()
    store = ChromaStore(
        embedding=embedding,
        collection_name=f"perf_{uuid4().hex[:8]}",
        client=client,
    )

    docs: list[Document] = _build_documents(_DOC_COUNT)

    # ---- 计时插入 ----
    start: float = time.perf_counter()
    store.add_documents(docs, chunk_size=100, concurrency=4)
    elapsed: float = time.perf_counter() - start
    qps: float = _DOC_COUNT / elapsed

    # ---- 抽样自监督召回 ----
    sample_indices: list[int] = list(range(0, _DOC_COUNT, _SAMPLE_STEP))
    hits: int = 0
    for idx in sample_indices:
        doc: Document = docs[idx]
        results = store.query(doc.text, k=5)
        top_ids: list[str] = [r.id for r in results]
        if doc.id in top_ids:
            hits += 1
    recall: float = hits / len(sample_indices)

    # ---- 打印报告（QPS 不做硬断言，仅打印） ----
    print(
        f"[perf] 插入 {_DOC_COUNT} 条耗时 {elapsed:.3f}s，"
        f"QPS = {qps:.2f} 条/秒"
    )
    print(
        f"[perf] 自监督召回 recall = {recall:.3f} "
        f"({hits}/{len(sample_indices)})"
    )

    # 自监督场景下，每条查询向量 == 自身向量，理论上应全命中。
    assert recall == 1.0
