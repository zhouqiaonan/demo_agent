"""检索器的抽象基类模块。

本模块定义了 rag 包中所有检索器的统一抽象接口 :class:`Retriever`。任何具体
的检索实现（向量检索、BM25 关键词检索、RRF 混合检索等）都应继承本类并实现
核心的 ``retrieve`` 方法，从而把「用户问题 → Top-K 相关 chunk」的检索动作
统一为对外契约，供下游的重排序、Prompt 构建与端到端编排环节使用。

设计意图：

- :class:`Retriever` 负责「给定查询返回 Top-K 相关 chunk」，是 RAG 链路中
  检索阶段的统一入口。
- 所有检索实现（向量 / 关键词 / 混合）都返回统一的
  :class:`vector_store.document.SearchResult` 列表，且按相关度降序排列
  （``score`` 越高越靠前），使上层业务无需关心底层检索后端即可无缝替换。
- 本抽象层统一了不同检索策略的对外契约，便于后续扩展
  :class:`KeywordRetriever` 与 :class:`HybridRetriever` 等实现。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from vector_store.document import SearchResult


class Retriever(ABC):
    """检索器的抽象基类。

    所有具体检索器（向量、关键词、混合等）都应继承本类，并实现抽象方法
    :meth:`retrieve`，把用户查询转化为按相关度降序排列的 Top-K 命中结果。

    契约约定：

    - :meth:`retrieve` 接收查询文本 ``query`` 与结果数量 ``k``，返回
      ``list[SearchResult]``。
    - 返回列表必须按相关度降序排列：即 ``SearchResult.score`` **越高越靠前**
      （或等价地 ``SearchResult.distance`` 越低越靠前）。
    - 检索过程中任何失败都应抛出
      :class:`rag.exceptions.RetrievalError`（或其子类），以便调用方统一捕获。
    """

    @abstractmethod
    def retrieve(self, query: str, k: int = 5) -> list[SearchResult]:
        """根据查询文本检索最相关的 ``k`` 条 chunk（抽象方法）。

        这是所有检索器的核心契约，子类必须实现。

        Args:
            query: 查询文本（通常为用户的自然语言问题）。
            k: 返回的最相关结果数量，默认 5。

        Returns:
            按相关度降序排列的 :class:`SearchResult` 列表，即 ``score`` 越高
            越靠前。

        Raises:
            RetrievalError: 检索失败（如向量存储不可用、索引缺失等）时抛出。
        """
        ...
