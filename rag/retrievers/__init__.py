"""rag.retrievers 子包：多路检索层。

本子包负责从向量存储中检索与用户问题相关的候选片段（chunk），
提供统一的检索抽象接口与具体的检索器实现：

- :class:`Retriever`：检索器的统一抽象接口，声明核心的 ``retrieve`` 方法。
- :class:`VectorRetriever`：基于向量相似度的检索器（**已实现**），通过依赖
  注入的向量存储后端完成语义检索。
- :class:`KeywordRetriever`：基于手写 BM25 与字符 bigram 分词的检索器
  （**已实现**），通过词频倒排索引完成关键词检索。
- :class:`HybridRetriever`：通过 RRF（Reciprocal Rank Fusion）融合多路检索
  结果的混合检索器（**已实现**）。

当前已实现 :class:`Retriever` 抽象接口、:class:`VectorRetriever` 向量检索器、
:class:`KeywordRetriever` 关键词检索器与 :class:`HybridRetriever` 混合检索器，
多路检索能力已全部落地。

本模块通过显式 ``__all__`` 导出公开 API，方便以
``from rag.retrievers import ...`` 的形式直接引用。
"""

from __future__ import annotations

from rag.retrievers.base import Retriever
from rag.retrievers.hybrid import HybridRetriever
from rag.retrievers.keyword import KeywordRetriever
from rag.retrievers.vector import VectorRetriever

__all__ = [
    "Retriever",
    "VectorRetriever",
    "KeywordRetriever",
    "HybridRetriever",
]
