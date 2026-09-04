"""rag 包端到端编排链（RAGChain / RAGAnswer）的单元测试。

本模块使用 fake Retriever / fake ReRanker / fake LLMClient，在完全离线、不调用
真实向量库与 LLM 的前提下，验证 :class:`RAGChain.ask` 的完整链路：

- 端到端返回 :class:`RAGAnswer`，答案文本与引用映射表正确。
- reranker 可选：未注入时正常跳过，注入时被调用。
- 检索阶段异常原样传播，其余阶段异常包装为 :class:`RAGChainError`。
"""

from __future__ import annotations

import pytest

from rag import PromptBuilder, RAGAnswer, RAGChain, RAGChainError, RetrievalError
from vector_store.document import SearchResult


def _make_result(
    doc_id: str,
    text: str,
    metadata: dict[str, object],
) -> SearchResult:
    """构造一个用于测试的 SearchResult 实例。

    Args:
        doc_id: 命中结果的唯一标识。
        text: 命中结果的正文文本。
        metadata: 命中结果的元数据。

    Returns:
        带有默认 score / distance 的 SearchResult。
    """
    return SearchResult(id=doc_id, text=text, metadata=metadata, score=0.0, distance=0.0)


class _FakeRetriever:
    """返回固定结果列表的 fake 检索器。"""

    def __init__(self, results: list[SearchResult]) -> None:
        """初始化 fake 检索器。

        Args:
            results: 固定返回的结果列表。
        """
        self._results: list[SearchResult] = results

    def retrieve(self, query: str, k: int = 5) -> list[SearchResult]:
        """返回固定结果列表。

        Args:
            query: 查询文本（忽略）。
            k: 返回数量（忽略）。

        Returns:
            固定结果列表。
        """
        return self._results


class _RaisingRetriever:
    """retrieve 时抛出 RetrievalError 的 fake 检索器。"""

    def retrieve(self, query: str, k: int = 5) -> list[SearchResult]:
        """抛出检索异常。

        Args:
            query: 查询文本（忽略）。
            k: 返回数量（忽略）。

        Raises:
            RetrievalError: 固定抛出。
        """
        raise RetrievalError("检索失败")


class _FakeLLMClient:
    """返回固定 content 的 fake LLM 客户端。"""

    def __init__(self, content: str) -> None:
        """初始化 fake LLM 客户端。

        Args:
            content: 固定的回答内容。
        """
        self._content: str = content

    def chat_completion(self, messages: list[dict]) -> dict:
        """返回含固定 content 的字典。

        Args:
            messages: 消息列表（忽略）。

        Returns:
            含 ``content`` 字段的字典。
        """
        return {"content": self._content}


class _RaisingLLMClient:
    """chat_completion 时抛出 RuntimeError 的 fake LLM 客户端。"""

    def chat_completion(self, messages: list[dict]) -> dict:
        """抛出运行时异常。

        Args:
            messages: 消息列表（忽略）。

        Raises:
            RuntimeError: 固定抛出。
        """
        raise RuntimeError("LLM 调用失败")


class _FakeReRanker:
    """记录调用并原样返回结果的 fake 重排序器。"""

    def __init__(self) -> None:
        """初始化 fake 重排序器，初始化调用记录。"""
        self.calls: list[str] = []

    def rerank(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        """记录 query 并原样返回候选结果。

        Args:
            query: 查询文本。
            results: 候选结果列表。

        Returns:
            原样返回候选结果列表。
        """
        self.calls.append(query)
        return results


class TestRAGChain:
    """RAGChain 端到端编排的单元测试。"""

    def _chunks(self) -> list[SearchResult]:
        """返回固定的两条候选 chunk。

        Returns:
            两条带元数据的 SearchResult。
        """
        return [
            _make_result("d1", "片段一", {"source": "a.md", "page": 1}),
            _make_result("d2", "片段二", {"source": "b.md", "page": 2}),
        ]

    def test_ask_returns_answer_with_references(self) -> None:
        """验证 ask 端到端返回 RAGAnswer，答案与引用映射表正确。"""
        chunks: list[SearchResult] = self._chunks()
        chain = RAGChain(
            retriever=_FakeRetriever(chunks),
            prompt_builder=PromptBuilder(),
            llm_client=_FakeLLMClient("答案"),
        )
        answer: RAGAnswer = chain.ask("问题")

        assert isinstance(answer, RAGAnswer)
        assert answer.answer == "答案"
        assert answer.references == {
            1: {"source": "a.md", "page": 1, "text": "片段一"},
            2: {"source": "b.md", "page": 2, "text": "片段二"},
        }

    def test_ask_without_reranker_works(self) -> None:
        """验证不传 reranker 时链路正常运行。"""
        chain = RAGChain(
            retriever=_FakeRetriever(self._chunks()),
            prompt_builder=PromptBuilder(),
            llm_client=_FakeLLMClient("答案"),
        )
        assert chain.ask("问题").answer == "答案"

    def test_ask_calls_reranker_when_provided(self) -> None:
        """验证注入 reranker 时其 rerank 被调用一次。"""
        reranker = _FakeReRanker()
        chain = RAGChain(
            retriever=_FakeRetriever(self._chunks()),
            prompt_builder=PromptBuilder(),
            llm_client=_FakeLLMClient("答案"),
            reranker=reranker,
        )
        chain.ask("问题")

        assert reranker.calls == ["问题"]

    def test_ask_propagates_retrieval_error(self) -> None:
        """验证检索阶段异常原样传播（不包装）。"""
        chain = RAGChain(
            retriever=_RaisingRetriever(),
            prompt_builder=PromptBuilder(),
            llm_client=_FakeLLMClient("答案"),
        )
        with pytest.raises(RetrievalError):
            chain.ask("问题")

    def test_ask_wraps_llm_error_into_chain_error(self) -> None:
        """验证 LLM 调用异常被包装为 RAGChainError。"""
        chain = RAGChain(
            retriever=_FakeRetriever(self._chunks()),
            prompt_builder=PromptBuilder(),
            llm_client=_RaisingLLMClient(),
        )
        with pytest.raises(RAGChainError):
            chain.ask("问题")
