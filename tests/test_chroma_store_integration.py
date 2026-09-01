"""ChromaStore 集成测试 —— 使用真实 ChromaDB（进程内内存客户端）。

本模块测试 :class:`ChromaStore` 与真实 ChromaDB（``chromadb.Client()``，
进程内、内存存储、无 docker）的端到端交互，覆盖插入/计数、文本查询、
向量查询、删除与维度不匹配校验等核心链路。

由于依赖真实 chromadb，本模块在顶部通过 ``pytest.importorskip("chromadb")``
保护：当 chromadb 未安装时，pytest 会优雅跳过本模块（skipped）而非报错。

所有测试使用唯一 collection 名称（``f"it_{uuid4().hex[:8]}"``）与独立的
内存客户端，避免测试之间互相污染。
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest

# 未安装 chromadb 时跳过整个模块，而非报错。
chromadb = pytest.importorskip("chromadb")

from tests.fakes import HashEmbedding
from vector_store.chroma_store import ChromaStore
from vector_store.document import Document
from vector_store.exceptions import DimensionMismatchError


# ---------------------------------------------------------------------------
# 标记：整个模块均为集成测试
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client() -> Any:
    """返回一个全新的进程内内存 chromadb 客户端（每个测试独立）。

    Returns:
        新创建的 ``chromadb.Client()`` 实例，随测试结束释放。
    """
    return chromadb.Client()


@pytest.fixture
def collection_name() -> str:
    """生成唯一的 collection 名称，避免测试之间互相污染。

    Returns:
        形如 ``it_xxxxxxxx`` 的唯一名称（``uuid4().hex`` 前 8 位）。
    """
    return f"it_{uuid4().hex[:8]}"


@pytest.fixture
def store(client: Any, collection_name: str) -> ChromaStore:
    """构造一个使用真实客户端与 HashEmbedding 的 ChromaStore。

    Args:
        client: 进程内内存 chromadb 客户端。
        collection_name: 唯一 collection 名称。

    Returns:
        已构造、尚未触发懒初始化的 ChromaStore。
    """
    return ChromaStore(
        embedding=HashEmbedding(dimension=16),
        collection_name=collection_name,
        client=client,
    )


def _make_docs(n: int) -> list[Document]:
    """构造 n 条带确定性 id 的文档。

    Args:
        n: 文档数量。

    Returns:
        文本为 ``测试文档 {i}``、id 为 ``doc-{i}`` 的文档列表。
    """
    return [
        Document(text=f"测试文档 {i}", id=f"doc-{i}") for i in range(n)
    ]


# ---------------------------------------------------------------------------
# 集成测试
# ---------------------------------------------------------------------------


class TestChromaStoreIntegration:
    """测试 ChromaStore 与真实 ChromaDB 的端到端行为。"""

    def test_add_documents_and_count(
        self, store: ChromaStore
    ) -> None:
        """add_documents 后 count() 应返回正确的文档数量。"""
        docs: list[Document] = _make_docs(5)
        ids: list[str] = store.add_documents(docs)

        # 返回的 id 列表应保持输入顺序。
        assert ids == [d.id for d in docs]
        assert store.count() == 5

    def test_text_query_top1_correct(
        self, store: ChromaStore
    ) -> None:
        """文本查询的 top-k 应命中自身（自监督：距离应为 0）。"""
        docs: list[Document] = _make_docs(10)
        store.add_documents(docs)

        target: Document = docs[3]
        results = store.query(target.text, k=5)

        top_ids: list[str] = [r.id for r in results]
        # 目标文档应出现在 top-k 中。
        assert target.id in top_ids
        # 自监督：查询向量 == 文档自身向量，top-1 应为自身，距离 ≈ 0。
        assert results[0].id == target.id
        assert results[0].distance == pytest.approx(0.0, abs=1e-4)

    def test_query_by_vector_structure(
        self, store: ChromaStore
    ) -> None:
        """query_by_vector 返回的 SearchResult 字段应齐全且类型正确。"""
        docs: list[Document] = _make_docs(5)
        store.add_documents(docs)

        # 用文档自身向量作为查询向量，保证有确定命中。
        query_vector: list[float] = store._embedding.embed([docs[0].text])[0]
        results = store.query_by_vector(query_vector, k=3)

        assert len(results) == 3
        for result in results:
            assert isinstance(result.id, str)
            assert isinstance(result.text, str)
            assert isinstance(result.metadata, dict)
            assert isinstance(result.score, float)
            assert isinstance(result.distance, float)

    def test_delete_decrements_count(
        self, store: ChromaStore
    ) -> None:
        """delete([id]) 后 count() 应递减。"""
        docs: list[Document] = _make_docs(5)
        store.add_documents(docs)
        assert store.count() == 5

        store.delete([docs[0].id])
        assert store.count() == 4

    def test_dimension_mismatch_raises(
        self, client: Any, collection_name: str
    ) -> None:
        """同 collection 下维度不同的第二次写入应抛出 DimensionMismatchError。

        先用 8 维嵌入写入，再用同一个 client 与同一个 collection 名称以 4 维
        嵌入写入，触发 ``_ensure_collection`` 中的维度校验。
        """
        # 第一次写入：8 维。
        store8 = ChromaStore(
            embedding=HashEmbedding(dimension=8),
            collection_name=collection_name,
            client=client,
        )
        store8.add_documents([Document(text="先写入 8 维", id="d8")])

        # 第二次写入：4 维，同 collection_name + 同 client。
        store4 = ChromaStore(
            embedding=HashEmbedding(dimension=4),
            collection_name=collection_name,
            client=client,
        )
        with pytest.raises(DimensionMismatchError) as exc_info:
            store4.add_documents([Document(text="再写 4 维", id="d4")])

        assert exc_info.value.expected_dim == 4
        assert exc_info.value.actual_dim == 8
