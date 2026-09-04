"""RAG 端到端集成测试 —— 真实内存 ChromaDB + 真实检索组件 + mock LLM。

本模块用真实 ``chromadb.Client()``（进程内、内存存储、完全离线、无 docker）
配合真实检索组件（:class:`VectorRetriever` + :class:`KeywordRetriever` +
:class:`HybridRetriever`）、真实 :class:`PromptBuilder` 与 mock LLM，走完整的
:class:`RAGChain.ask` 流程，验证**答案引用来源的准确性**：

- 引用映射表确实命中了含目标关键词的文档，且 ``source`` 元数据正确；
- 引用编号是从 1 开始的连续整数，且每个引用文本都来自真实检索到的 chunk；
- 混合检索中关键词检索（BM25）能够命中「向量难以命中」的查询。

由于依赖真实 chromadb，本模块在顶部通过 ``pytest.importorskip("chromadb")``
保护：当 chromadb 未安装时，pytest 会优雅跳过本模块（skipped）而非报错。

说明：
    除 3 条主题明确的文档（苹果 / 香蕉 / 汽车）外，额外加入若干与主题无关的
    干扰文档。:class:`HashEmbedding` 是基于 SHA-256 的**非语义**嵌入，向量检索
    的排序近乎随机；加入干扰文档后，向量检索在 ``k=3`` 下很可能不把「香蕉」
    文档排入 Top-K，而关键词检索（BM25）通过字符 bigram 一定能精确命中，从而
    真正体现关键词检索在混合检索（RRF 融合）中的价值。
"""

from __future__ import annotations

from unittest.mock import MagicMock
from uuid import uuid4

import pytest

# 未安装 chromadb 时跳过整个模块，而非报错。
chromadb = pytest.importorskip("chromadb")

from rag import (
    HybridRetriever,
    KeywordRetriever,
    PromptBuilder,
    RAGAnswer,
    RAGChain,
    VectorRetriever,
)
from tests.fakes import HashEmbedding
from vector_store.chroma_store import ChromaStore
from vector_store.document import Document


# ---------------------------------------------------------------------------
# 标记：整个模块均为集成测试
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# 测试语料
# ---------------------------------------------------------------------------

# 3 条主题明确的文档：苹果 / 香蕉是水果，汽车是交通工具。
_CORE_DOCS: list[Document] = [
    Document(
        text="苹果是一种常见的水果，富含维生素和膳食纤维。",
        id="fruits-apple",
        metadata={"source": "fruits.txt"},
    ),
    Document(
        text="香蕉是一种常见的水果，富含钾元素，口感软糯。",
        id="fruits-banana",
        metadata={"source": "fruits.txt"},
    ),
    Document(
        text="汽车是一种常见的交通工具，用于日常出行。",
        id="vehicles-car",
        metadata={"source": "vehicles.txt"},
    ),
]

# 若干与主题无关的干扰文档：稀释向量检索结果，凸显关键词检索在混合检索中的价值。
_DISTRACTOR_DOCS: list[Document] = [
    Document(
        text=f"这是一篇与主题无关的干扰文档，编号为 {i}。",
        id=f"distractor-{i}",
        metadata={"source": "misc.txt"},
    )
    for i in range(6)
]

# 完整语料：3 条主题文档 + 干扰文档。
_ALL_DOCS: list[Document] = _CORE_DOCS + _DISTRACTOR_DOCS


# ---------------------------------------------------------------------------
# 构造辅助函数
# ---------------------------------------------------------------------------


def _build_chain(docs: list[Document], llm_content: str) -> RAGChain:
    """用真实内存 ChromaDB + 真实检索组件 + mock LLM 组装一条 RAG 链。

    流程：
    1. 构造一个使用唯一 collection 名称与 HashEmbedding 的 ChromaStore，并把
       ``docs`` 全部写入；
    2. 依次构造 VectorRetriever、KeywordRetriever 与 HybridRetriever；
    3. 用真实 PromptBuilder 与 mock LLM（``MagicMock``）组装 RAGChain。

    Args:
        docs: 待写入向量库并用于关键词索引的文档列表。
        llm_content: mock LLM 的 ``chat_completion`` 返回的 ``content``。

    Returns:
        已配置完毕的 :class:`RAGChain`，检索数量 ``k=3``。
    """
    store: ChromaStore = ChromaStore(
        embedding=HashEmbedding(dimension=64),
        collection_name=f"e2e_{uuid4().hex[:8]}",
        client=chromadb.Client(),
    )
    store.add_documents(docs)

    vector: VectorRetriever = VectorRetriever(store)
    keyword: KeywordRetriever = KeywordRetriever(docs)
    hybrid: HybridRetriever = HybridRetriever([vector, keyword])

    pb: PromptBuilder = PromptBuilder()

    llm: MagicMock = MagicMock()
    llm.chat_completion.return_value = {"content": llm_content}

    return RAGChain(hybrid, pb, llm, k=3)


# ---------------------------------------------------------------------------
# 端到端集成测试
# ---------------------------------------------------------------------------


class TestRagE2E:
    """用真实内存 ChromaDB 走完整 RAG 链路，验证引用来源准确性。"""

    def test_end_to_end_reference_accuracy(self) -> None:
        """端到端流程应命中含「苹果」的文档，且 source 元数据正确。

        查询「苹果是什么」，mock LLM 原样返回带引用标注的答案；断言：
        - 答案文本与 mock 返回一致；
        - 引用映射表中存在 text 含「苹果」的 chunk（检索确实命中该文档）；
        - 该 chunk 的 ``source`` 元数据为 ``"fruits.txt"``。
        """
        chain: RAGChain = _build_chain(_ALL_DOCS, "苹果是水果。[1]")

        answer: RAGAnswer = chain.ask("苹果是什么")

        # 答案文本应原样来自 mock LLM。
        assert answer.answer == "苹果是水果。[1]"

        # 引用准确性：检索应命中含「苹果」的文档，且它进入了引用映射表。
        apple_refs: list[dict] = [
            ref for ref in answer.references.values() if "苹果" in ref["text"]
        ]
        assert apple_refs, "检索结果中应包含含「苹果」的文档 chunk"

        # 该 chunk 的 source 元数据应为 fruits.txt。
        assert any(ref["source"] == "fruits.txt" for ref in apple_refs)

    def test_references_indices_are_contiguous(self) -> None:
        """引用编号应为从 1 开始的连续整数，且每个引用文本都来自真实 chunk。

        断言：
        - 引用映射表的键排序后恰为 ``1..len(references)`` 的连续整数；
        - 每个引用的 ``text`` 非空，且确实来自检索语料中的某条文档。
        """
        chain: RAGChain = _build_chain(_ALL_DOCS, "答案。")

        answer: RAGAnswer = chain.ask("苹果是什么")

        keys: list[int] = sorted(answer.references.keys())
        assert keys == list(range(1, len(answer.references) + 1))

        # 每个引用文本都应来自真实检索到的 chunk（非空且存在于语料中）。
        all_texts: set[str] = {doc.text for doc in _ALL_DOCS}
        for ref in answer.references.values():
            assert isinstance(ref["text"], str)
            assert ref["text"].strip() != ""
            assert ref["text"] in all_texts

    def test_hybrid_retrieval_hits_keyword_match(self) -> None:
        """混合检索应通过关键词检索命中「向量难命中」的「香蕉」查询。

        HashEmbedding 是哈希型非语义嵌入，向量检索对「香蕉」的排序近乎随机，
        且存在 6 条干扰文档，向量检索在 ``k=3`` 下很可能漏掉「香蕉」文档；
        而关键词检索（BM25 + 字符 bigram）一定能精确命中。断言混合检索最终
        的引用映射表中确实包含「香蕉」文档，体现关键词检索在 RRF 融合中的价值。
        """
        chain: RAGChain = _build_chain(_ALL_DOCS, "香蕉是水果。[1]")

        answer: RAGAnswer = chain.ask("香蕉")

        # 混合检索结果（经 references 的 text）中应确实包含「香蕉」文档。
        banana_refs: list[dict] = [
            ref for ref in answer.references.values() if "香蕉" in ref["text"]
        ]
        assert banana_refs, "混合检索应通过关键词检索命中「香蕉」文档"

        # 「香蕉」文档应来自水果文档，source 元数据为 fruits.txt。
        assert any(ref["source"] == "fruits.txt" for ref in banana_refs)
