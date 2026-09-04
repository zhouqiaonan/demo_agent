"""rag 包入口：RAG 核心流程编排层。

本包负责「用户问题 → 检索 → 重排序 → Prompt → 回答」的核心链路，
是 RAG 系统中将检索与生成串联起来的中枢环节。它计划提供四组核心能力：

- **多路检索**：:class:`Retriever`（抽象接口）、:class:`VectorRetriever`
  （向量检索）、:class:`KeywordRetriever`（BM25 关键词检索，已实现）与
  :class:`HybridRetriever`（RRF 融合检索）。
- **重排序**：:class:`ReRanker`（抽象接口）与 :class:`CrossEncoderReRanker`
  （基于交叉编码器的具体实现，懒加载重依赖）。
- **Prompt 构建**：:class:`PromptBuilder`（带引用片段构建最终 Prompt）。
- **端到端编排**：:class:`RAGChain`（串接检索、重排序、Prompt 构建与 LLM 调用）。

本包复用 :class:`vector_store` 的向量检索能力与 :class:`llm_client` 的
LLM 调用能力，从而避免重复实现底层基础设施。

本模块通过显式 ``__all__`` 导出公开 API，方便以
``from rag import ...`` 的形式直接引用。

.. note::
    rag 包已全部实现完毕，当前导出了完整的公开 API：异常体系
    （:class:`RAGError` / :class:`RetrievalError` / :class:`RerankError` /
    :class:`PromptBuildError` / :class:`RAGChainError`）、检索器抽象
    :class:`Retriever`、向量检索器 :class:`VectorRetriever`、关键词检索器
    :class:`KeywordRetriever`、混合检索器 :class:`HybridRetriever`、重排序器
    抽象 :class:`ReRanker` 与交叉编码器重排序器 :class:`CrossEncoderReRanker`、
    带引用来源的 Prompt 构建器 :class:`PromptBuilder` 与构建产物
    :class:`RAGPrompt`，以及端到端编排链 :class:`RAGChain` 与编排产物
    :class:`RAGAnswer`。
"""

from __future__ import annotations

from rag.chain import RAGAnswer, RAGChain
from rag.exceptions import (
    PromptBuildError,
    RAGChainError,
    RAGError,
    RerankError,
    RetrievalError,
)
from rag.prompt import PromptBuilder, RAGPrompt
from rag.rerankers import CrossEncoderReRanker, ReRanker
from rag.retrievers import (
    HybridRetriever,
    KeywordRetriever,
    Retriever,
    VectorRetriever,
)

__all__ = [
    "RAGError",
    "RetrievalError",
    "RerankError",
    "PromptBuildError",
    "RAGChainError",
    "Retriever",
    "VectorRetriever",
    "KeywordRetriever",
    "HybridRetriever",
    "ReRanker",
    "CrossEncoderReRanker",
    "PromptBuilder",
    "RAGPrompt",
    "RAGChain",
    "RAGAnswer",
]
