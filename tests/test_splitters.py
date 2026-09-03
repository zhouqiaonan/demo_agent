"""document_processor/splitters 的单元测试 —— 递归分割与语义分块。

本模块测试两类文本分割器的核心行为：

- :class:`RecursiveTextSplitter`：非法参数（``chunk_overlap >= chunk_size``、
  ``chunk_size <= 0``）抛错、长文本正确分块、``split_documents`` 保留源
  metadata 并追加局部 ``chunk_index``。
- :class:`SemanticSplitter`：单句 / 空文本的边界行为、非法阈值抛错，以及
  基于可控伪嵌入（:class:`_SemanticStubEmbedding`）的语义断句——语义相同的
  相邻句子相似度为 1.0 不断句、语义切换处相似度为 0.0 断句。

所有测试完全离线、确定，无需真实嵌入模型。
"""

from __future__ import annotations

import pytest

from document_processor import (
    DocumentSplitError,
    RecursiveTextSplitter,
    SemanticSplitter,
)
from vector_store.document import Document
from vector_store.embeddings.base import EmbeddingClient


# ---------------------------------------------------------------------------
# 语义分块用的可控伪嵌入
# ---------------------------------------------------------------------------


class _SemanticStubEmbedding(EmbeddingClient):
    """可控的语义伪嵌入：按 marker 关键字把句子映射为两组正交向量。

    包含 ``marker`` 关键字的句子返回 ``[1.0, 0.0]``，其余句子返回
    ``[0.0, 1.0]``。这样组内句子的余弦相似度为 1.0（语义相同），跨组句子的
    相似度为 0.0（语义不同），从而可精确验证语义断句行为。
    """

    def __init__(self, marker: str = "苹果") -> None:
        """初始化伪嵌入。

        Args:
            marker: 用于区分语义组的关键字，默认 ``"苹果"``。
        """
        self._marker: str = marker
        """用于区分语义组的关键字。"""

    @property
    def dimension(self) -> int:
        """返回输出向量的维度（固定为 2）。

        Returns:
            输出向量的维度（2）。
        """
        return 2

    def embed(self, texts: list[str]) -> list[list[float]]:
        """把一批句子映射为两组正交向量（顺序与输入一致）。

        Args:
            texts: 待嵌入的句子列表。

        Returns:
            与输入顺序一致的浮点向量列表；含 marker 的句子返回 ``[1.0, 0.0]``，
            其余返回 ``[0.0, 1.0]``。
        """
        vectors: list[list[float]] = []
        for text in texts:
            if self._marker in text:
                vectors.append([1.0, 0.0])
            else:
                vectors.append([0.0, 1.0])
        return vectors


# ---------------------------------------------------------------------------
# RecursiveTextSplitter
# ---------------------------------------------------------------------------


class TestRecursiveTextSplitter:
    """测试递归分割器的参数校验、分块与文档级分割。"""

    def test_chunk_overlap_ge_chunk_size_raises(self) -> None:
        """chunk_overlap >= chunk_size 时应抛出 DocumentSplitError。"""
        with pytest.raises(DocumentSplitError):
            RecursiveTextSplitter(chunk_size=100, chunk_overlap=100)

    @pytest.mark.parametrize("chunk_size", [0, -5])
    def test_chunk_size_le_zero_raises(self, chunk_size: int) -> None:
        """chunk_size <= 0 时应抛出 DocumentSplitError。"""
        with pytest.raises(DocumentSplitError):
            RecursiveTextSplitter(chunk_size=chunk_size, chunk_overlap=0)

    def test_split_text_splits_long_text(self) -> None:
        """长文本应被切分为多个长度不超过 chunk_size 的 chunk。"""
        splitter: RecursiveTextSplitter = RecursiveTextSplitter(
            chunk_size=50, chunk_overlap=0
        )
        long_text: str = " ".join(f"word{i}" for i in range(60))

        chunks: list[str] = splitter.split_text(long_text)

        assert len(chunks) > 1
        assert all(len(chunk) <= 50 for chunk in chunks)

    def test_split_documents_preserves_metadata_and_adds_chunk_index(self) -> None:
        """split_documents 应保留源 metadata 并按文档局部从 0 递增追加 chunk_index。"""
        splitter: RecursiveTextSplitter = RecursiveTextSplitter(
            chunk_size=5, chunk_overlap=0
        )
        docs: list[Document] = [
            Document(
                text="a b c d e f g h i j k l m n o p",
                metadata={"source": "x.pdf", "page": 0},
            ),
            Document(
                text="p q r s t u v w x y z",
                metadata={"source": "y.pdf", "page": 1},
            ),
        ]

        result: list[Document] = splitter.split_documents(docs)

        # 每个 chunk 都应保留源 metadata 并新增 chunk_index。
        assert all("chunk_index" in d.metadata for d in result)
        assert all("source" in d.metadata for d in result)
        assert all("page" in d.metadata for d in result)

        # chunk_index 仅在单个源文档内部局部递增（0 开始），不跨文档累加。
        first_chunks: list[Document] = [
            d for d in result if d.metadata["source"] == "x.pdf"
        ]
        second_chunks: list[Document] = [
            d for d in result if d.metadata["source"] == "y.pdf"
        ]
        assert [d.metadata["chunk_index"] for d in first_chunks] == list(
            range(len(first_chunks))
        )
        assert [d.metadata["chunk_index"] for d in second_chunks] == list(
            range(len(second_chunks))
        )

    @pytest.mark.parametrize("text", ["", "   ", "\n  \t"])
    def test_empty_input_returns_empty_list(self, text: str) -> None:
        """空/纯空白输入应返回空列表（空进空出），而非抛错。"""
        splitter: RecursiveTextSplitter = RecursiveTextSplitter()
        assert splitter.split_text(text) == []


# ---------------------------------------------------------------------------
# SemanticSplitter
# ---------------------------------------------------------------------------


class TestSemanticSplitter:
    """测试语义分块器的边界行为、参数校验与语义断句。"""

    def test_single_sentence_returns_original(self) -> None:
        """单句输入应原样返回 [text]，不触发嵌入。"""
        splitter: SemanticSplitter = SemanticSplitter(
            embedding=_SemanticStubEmbedding()
        )
        assert splitter.split_text("只有一句话。") == ["只有一句话。"]

    def test_empty_text_returns_empty_list(self) -> None:
        """空白文本应返回空列表。"""
        splitter: SemanticSplitter = SemanticSplitter(
            embedding=_SemanticStubEmbedding()
        )
        assert splitter.split_text("   \n  ") == []

    @pytest.mark.parametrize("threshold", [1.5, -0.1])
    def test_invalid_threshold_raises(self, threshold: float) -> None:
        """非法 breakpoint_threshold（>1 或 <0）应抛出 DocumentSplitError。"""
        with pytest.raises(DocumentSplitError):
            SemanticSplitter(
                embedding=_SemanticStubEmbedding(),
                breakpoint_threshold=threshold,
            )

    def test_buffer_size_lt_one_raises(self) -> None:
        """buffer_size < 1 时应抛出 DocumentSplitError。"""
        with pytest.raises(DocumentSplitError):
            SemanticSplitter(embedding=_SemanticStubEmbedding(), buffer_size=0)

    def test_semantic_breakpoint_splits(self) -> None:
        """语义相同处不断句、语义切换处断句：A A B 应得到 2 个 chunk。"""
        splitter: SemanticSplitter = SemanticSplitter(
            embedding=_SemanticStubEmbedding(),
            buffer_size=1,
            breakpoint_threshold=0.5,
        )

        chunks: list[str] = splitter.split_text(
            "苹果很好吃。苹果很甜。香蕉是黄色的。"
        )

        assert chunks == ["苹果很好吃。苹果很甜。", "香蕉是黄色的。"]

    def test_no_breakpoint_when_all_similar(self) -> None:
        """全部句子语义相同时相似度始终高于阈值，不应断句，得到单个 chunk。"""
        splitter: SemanticSplitter = SemanticSplitter(
            embedding=_SemanticStubEmbedding(),
            buffer_size=1,
            breakpoint_threshold=0.5,
        )

        chunks: list[str] = splitter.split_text("苹果很好吃。苹果很甜。苹果很脆。")

        assert chunks == ["苹果很好吃。苹果很甜。苹果很脆。"]
