"""rag 包检索器（VectorRetriever / KeywordRetriever / HybridRetriever）的单元测试。

本模块使用 MagicMock 与手写 fake 检索器，在完全离线、不依赖任何向量数据库或
外部检索库的前提下，验证三类检索器的核心契约：

- :class:`VectorRetriever`：是否正确委托注入的 store 并透传 ``k``、原样返回结果。
- :class:`KeywordRetriever`：手写 BM25 + 字符 bigram 的关键词排序是否正确。
- :class:`HybridRetriever`：RRF 融合是否正确累加多路排名，以及构造参数校验。
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from rag import HybridRetriever, KeywordRetriever, RetrievalError, VectorRetriever
from rag.retrievers.keyword import _bigrams
from vector_store.document import Document, SearchResult


def _make_result(doc_id: str, text: str) -> SearchResult:
    """构造一个用于测试的 SearchResult 实例。

    Args:
        doc_id: 命中结果的唯一标识。
        text: 命中结果的正文文本。

    Returns:
        带有默认 score / distance 的 SearchResult。
    """
    return SearchResult(id=doc_id, text=text, metadata={}, score=0.0, distance=0.0)


class _FakeRetriever:
    """返回固定结果列表的 fake 检索器。

    用于 :class:`HybridRetriever` 的融合测试，模拟多路检索器各自返回不同的
    排序结果。
    """

    def __init__(self, results: list[SearchResult]) -> None:
        """初始化 fake 检索器。

        Args:
            results: 该检索器固定返回的结果列表。
        """
        self._results: list[SearchResult] = results

    def retrieve(self, query: str, k: int = 5) -> list[SearchResult]:
        """返回构造时固定的结果列表，忽略查询与 ``k``。

        Args:
            query: 查询文本（忽略）。
            k: 返回数量（忽略）。

        Returns:
            构造时固定的结果列表。
        """
        return self._results


class TestVectorRetriever:
    """VectorRetriever 的单元测试。"""

    def test_retrieve_delegates_to_store_and_passes_k(self) -> None:
        """验证 retrieve 委托 store.query 并透传 k、原样返回结果。"""
        store: MagicMock = MagicMock()
        expected: list[SearchResult] = [_make_result("d1", "文本")]
        store.query.return_value = expected

        retriever = VectorRetriever(store)
        result: list[SearchResult] = retriever.retrieve("问题", k=3)

        store.query.assert_called_once_with("问题", 3)
        assert result is expected

    def test_retrieve_with_non_positive_k_raises(self) -> None:
        """验证 k=0 时在委托 store 之前抛 RetrievalError。"""
        store: MagicMock = MagicMock()
        retriever = VectorRetriever(store)

        with pytest.raises(RetrievalError):
            retriever.retrieve("问题", k=0)

        # 校验在委托前抛出，不应调用底层 store.query。
        store.query.assert_not_called()


class TestKeywordRetriever:
    """KeywordRetriever（手写 BM25 + 字符 bigram）的单元测试。"""

    def test_query_with_keyword_ranks_doc_with_more_terms_first(self) -> None:
        """验证含更多查询词项的文档分数最高、排第一。"""
        docs: list[Document] = [
            Document(text="苹果好吃", id="doc1"),
            Document(text="苹果苹果好吃", id="doc2"),
            Document(text="香蕉好吃", id="doc3"),
        ]
        retriever = KeywordRetriever(docs)
        results: list[SearchResult] = retriever.retrieve("苹果", k=5)

        # doc2 含「苹果」bigram 两次，分数应最高；doc3 无「苹果」应被过滤。
        assert len(results) == 2
        assert results[0].id == "doc2"
        assert results[1].id == "doc1"
        assert results[0].score > results[1].score

    def test_empty_corpus_returns_empty(self) -> None:
        """验证空语料检索返回空列表。"""
        retriever = KeywordRetriever([])
        assert retriever.retrieve("苹果", k=5) == []

    def test_query_without_common_bigram_returns_empty(self) -> None:
        """验证无共同 bigram 的查询返回空列表。"""
        docs: list[Document] = [
            Document(text="苹果好吃", id="doc1"),
            Document(text="香蕉好吃", id="doc2"),
        ]
        retriever = KeywordRetriever(docs)
        assert retriever.retrieve("葡萄", k=5) == []

    def test_bigrams_splits_chinese_text(self) -> None:
        """验证 _bigrams 把中文文本切分为正确的字符 bigram。"""
        assert _bigrams("苹果好吃") == ["苹果", "果好", "好吃"]

    def test_k1_zero_raises(self) -> None:
        """验证 k1=0 抛 RetrievalError（k1 必须为正数）。"""
        docs: list[Document] = [Document(text="苹果好吃", id="doc1")]
        with pytest.raises(RetrievalError):
            KeywordRetriever(docs, k1=0)

    def test_b_out_of_range_raises(self) -> None:
        """验证 b=1.5 抛 RetrievalError（b 必须在 [0.0, 1.0] 内）。"""
        docs: list[Document] = [Document(text="苹果好吃", id="doc1")]
        with pytest.raises(RetrievalError):
            KeywordRetriever(docs, b=1.5)

    def test_retrieve_with_non_positive_k_raises(self) -> None:
        """验证 k=0 抛 RetrievalError。"""
        docs: list[Document] = [Document(text="苹果好吃", id="doc1")]
        retriever = KeywordRetriever(docs)
        with pytest.raises(RetrievalError):
            retriever.retrieve("苹果", k=0)


class TestHybridRetriever:
    """HybridRetriever（RRF 融合）的单元测试。"""

    def test_rrf_fusion_accumulates_rank_across_retrievers(self) -> None:
        """验证 [A,B] 与 [B,A] 两路结果融合后文档分数被累加。"""
        a: SearchResult = _make_result("A", "A")
        b: SearchResult = _make_result("B", "B")
        retriever = HybridRetriever(
            [_FakeRetriever([a, b]), _FakeRetriever([b, a])], rrf_k=60
        )
        merged: list[SearchResult] = retriever.retrieve("q", k=5)

        # A、B 各在一条路排第 1、另一条路排第 2，RRF 分数应相等且都 > 0。
        assert {r.id for r in merged} == {"A", "B"}
        assert merged[0].score == pytest.approx(1 / 61 + 1 / 62)
        assert merged[1].score == merged[0].score

    def test_doc_first_in_multiple_retrievers_ranks_highest(self) -> None:
        """验证「多路都排第一」的文档融合后综合排名最高。"""
        a: SearchResult = _make_result("A", "A")
        b: SearchResult = _make_result("B", "B")
        c: SearchResult = _make_result("C", "C")
        retriever = HybridRetriever(
            [_FakeRetriever([a, b]), _FakeRetriever([a, c])], rrf_k=60
        )
        merged: list[SearchResult] = retriever.retrieve("q", k=5)

        # A 在两路都排第一，融合分数最高，应排在结果首位。
        assert merged[0].id == "A"
        assert merged[0].score > merged[1].score

    def test_empty_retrievers_raises(self) -> None:
        """验证空 retrievers 列表抛 RetrievalError。"""
        with pytest.raises(RetrievalError):
            HybridRetriever([])

    def test_rrf_k_zero_raises(self) -> None:
        """验证 rrf_k=0 抛 RetrievalError。"""
        with pytest.raises(RetrievalError):
            HybridRetriever([_FakeRetriever([])], rrf_k=0)

    def test_retrieve_with_non_positive_k_raises(self) -> None:
        """验证 k=0 抛 RetrievalError。"""
        retriever = HybridRetriever([_FakeRetriever([])])
        with pytest.raises(RetrievalError):
            retriever.retrieve("q", k=0)
