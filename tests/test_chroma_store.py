"""Tests for vector_store/chroma_store.py — ChromaStore 与 VectorStore 抽象。

所有测试通过注入 ``MagicMock`` 客户端与确定性的 :class:`HashEmbedding` 完成，
完全离线、无需安装 chromadb / sentence-transformers。
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from tests.fakes import HashEmbedding
from vector_store.chroma_store import ChromaStore, VectorStore
from vector_store.distance import DistanceMetric
from vector_store.document import Document
from vector_store.exceptions import DimensionMismatchError


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client() -> MagicMock:
    """注入 ChromaStore 的 mock chromadb 客户端。

    Returns:
        配置好的 MagicMock：``get_or_create_collection`` 返回一个空的 mock
        collection（``count() == 0``、``get()`` 返回空 embeddings），以跳过
        维度校验。
    """
    mock = MagicMock()
    collection = MagicMock()
    collection.count.return_value = 0
    collection.get.return_value = {"embeddings": []}
    mock.get_or_create_collection.return_value = collection
    return mock


@pytest.fixture
def collection(client: MagicMock) -> MagicMock:
    """mock 客户端 ``get_or_create_collection`` 返回的 collection 对象。

    Args:
        client: mock chromadb 客户端。

    Returns:
        mock collection 对象。
    """
    return client.get_or_create_collection.return_value


@pytest.fixture
def store(client: MagicMock) -> ChromaStore:
    """使用 HashEmbedding 与 mock client 构造的 ChromaStore。

    Args:
        client: mock chromadb 客户端。

    Returns:
        已构造但尚未触发懒初始化的 ChromaStore。
    """
    return ChromaStore(
        embedding=HashEmbedding(dimension=16),
        collection_name="default",
        client=client,
    )


# ---------------------------------------------------------------------------
# VectorStore 抽象接口
# ---------------------------------------------------------------------------


class TestVectorStoreABC:
    """测试 VectorStore 抽象基类契约。"""

    def test_abstract_cannot_instantiate(self) -> None:
        """抽象基类不可直接实例化，应抛出 TypeError。"""
        with pytest.raises(TypeError):
            VectorStore()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# 懒创建 collection
# ---------------------------------------------------------------------------


class TestChromaStoreLazyCollection:
    """测试 collection 的懒创建行为。"""

    def test_collection_not_created_before_write(
        self, client: MagicMock, store: ChromaStore
    ) -> None:
        """首次写入前不应调用 get_or_create_collection。"""
        client.get_or_create_collection.assert_not_called()

    def test_collection_created_on_first_write(
        self, client: MagicMock, store: ChromaStore
    ) -> None:
        """首次写入后才创建 collection，并传入 name 与 metadata。"""
        store.add_documents([Document(text="x")])
        client.get_or_create_collection.assert_called_once()
        kwargs: dict = client.get_or_create_collection.call_args[1]
        assert kwargs["name"] == "default"
        assert kwargs["metadata"] == {"hnsw:space": DistanceMetric.COSINE.value}

    def test_collection_custom_name_and_metric(self) -> None:
        """自定义 collection_name 与 metric 应透传给 get_or_create_collection。"""
        client = MagicMock()
        collection = MagicMock()
        collection.count.return_value = 0
        collection.get.return_value = {"embeddings": []}
        client.get_or_create_collection.return_value = collection

        store = ChromaStore(
            embedding=HashEmbedding(dimension=8),
            collection_name="my-col",
            metric=DistanceMetric.L2,
            client=client,
        )
        store.add_documents([Document(text="x")])
        kwargs: dict = client.get_or_create_collection.call_args[1]
        assert kwargs["name"] == "my-col"
        assert kwargs["metadata"] == {"hnsw:space": "l2"}


# ---------------------------------------------------------------------------
# 维度不匹配
# ---------------------------------------------------------------------------


class TestChromaStoreDimensionMismatch:
    """测试写入前的维度校验。"""

    def test_dimension_mismatch_raises(
        self, client: MagicMock, collection: MagicMock
    ) -> None:
        """既有数据维度与嵌入维度不一致时应抛出 DimensionMismatchError。"""
        collection.count.return_value = 1
        collection.get.return_value = {"embeddings": [[0.0] * 8]}

        store = ChromaStore(embedding=HashEmbedding(dimension=4), client=client)
        with pytest.raises(DimensionMismatchError) as exc_info:
            store.add_documents([Document(text="x")])
        assert exc_info.value.expected_dim == 4
        assert exc_info.value.actual_dim == 8


# ---------------------------------------------------------------------------
# 写入：分片并发 + 空 metadata 转换
# ---------------------------------------------------------------------------


class TestChromaStoreAddDocuments:
    """测试 add_documents 的分片与 metadata 处理。"""

    def test_chunking_calls_add_three_times(
        self, collection: MagicMock, store: ChromaStore
    ) -> None:
        """250 条文档、chunk_size=100 时应分 3 次调用 collection.add。"""
        docs: list[Document] = [
            Document(text=f"doc-{i}", id=f"id-{i}") for i in range(250)
        ]
        ids: list[str] = store.add_documents(docs, chunk_size=100)

        assert collection.add.call_count == 3
        assert len(ids) == 250
        # 返回的 id 列表应保持输入顺序。
        assert ids == [d.id for d in docs]

    def test_empty_metadata_converted_to_none(
        self, collection: MagicMock, store: ChromaStore
    ) -> None:
        """空 metadata 文档在 metadatas 中对应位置应为 None。"""
        docs: list[Document] = [
            Document(text="a", metadata={}),
            Document(text="b", metadata={"x": "y"}),
            Document(text="c", metadata={}),
        ]
        store.add_documents(docs)

        metadatas: list = collection.add.call_args[1]["metadatas"]
        assert metadatas[0] is None
        assert metadatas[1] == {"x": "y"}
        assert metadatas[2] is None

    def test_chunk_size_zero_raises(self, store: ChromaStore) -> None:
        """chunk_size=0 时应抛出 ValueError。"""
        with pytest.raises(ValueError, match="chunk_size"):
            store.add_documents([Document(text="x")], chunk_size=0)

    def test_concurrency_zero_raises(self, store: ChromaStore) -> None:
        """concurrency=0 时应抛出 ValueError。"""
        with pytest.raises(ValueError, match="concurrency"):
            store.add_documents([Document(text="x")], concurrency=0)


# ---------------------------------------------------------------------------
# 检索：query 映射 + 文本转向量
# ---------------------------------------------------------------------------


class TestChromaStoreQuery:
    """测试 query / query_by_vector 的结果映射与参数传递。"""

    def _set_query_result(self, collection: MagicMock) -> None:
        """为 mock collection 配置一个标准的 query 返回结果。

        Args:
            collection: mock collection 对象。
        """
        collection.query.return_value = {
            "ids": [["id1", "id2"]],
            "documents": [["hello", "world"]],
            "metadatas": [[{"a": 1}, None]],
            "distances": [[0.1, 0.9]],
        }

    def test_query_by_vector_maps_distance_to_score(
        self, collection: MagicMock, store: ChromaStore
    ) -> None:
        """COSINE 度量下 score 应等于 1 - distance，字段对应正确。"""
        self._set_query_result(collection)
        results = store.query_by_vector([0.0] * 16, k=2)

        assert len(results) == 2

        assert results[0].id == "id1"
        assert results[0].text == "hello"
        assert results[0].metadata == {"a": 1}
        assert results[0].distance == 0.1
        assert results[0].score == pytest.approx(1.0 - 0.1)

        assert results[1].id == "id2"
        assert results[1].text == "world"
        # None 元数据应归一化为空字典。
        assert results[1].metadata == {}
        assert results[1].distance == 0.9
        assert results[1].score == pytest.approx(1.0 - 0.9)

    def test_query_text_embeds_then_queries(
        self, collection: MagicMock, store: ChromaStore
    ) -> None:
        """query(text) 应先向量化再调用 collection.query；where=None 时不传 where。"""
        self._set_query_result(collection)
        text: str = "hello world"
        expected_vector: list[float] = store._embedding.embed([text])[0]

        store.query(text, k=3)

        collection.query.assert_called_once()
        kwargs: dict = collection.query.call_args[1]
        assert kwargs["query_embeddings"] == [expected_vector]
        assert kwargs["n_results"] == 3
        assert "where" not in kwargs

    def test_query_passes_where_when_provided(
        self, collection: MagicMock, store: ChromaStore
    ) -> None:
        """where 非 None 时应透传给 collection.query。"""
        self._set_query_result(collection)
        store.query_by_vector([0.0] * 16, k=2, where={"source": "wiki"})
        kwargs: dict = collection.query.call_args[1]
        assert kwargs["where"] == {"source": "wiki"}


# ---------------------------------------------------------------------------
# 删除与计数
# ---------------------------------------------------------------------------


class TestChromaStoreDeleteAndCount:
    """测试 delete 与 count 的委托。"""

    def test_delete_delegates(
        self, collection: MagicMock, store: ChromaStore
    ) -> None:
        """delete(ids) 应委托给 collection.delete(ids=...)。"""
        store.delete(["id1", "id2"])
        collection.delete.assert_called_once_with(ids=["id1", "id2"])

    def test_count_delegates(
        self, collection: MagicMock, store: ChromaStore
    ) -> None:
        """count() 应返回 collection.count() 的值。"""
        collection.count.return_value = 42
        assert store.count() == 42
