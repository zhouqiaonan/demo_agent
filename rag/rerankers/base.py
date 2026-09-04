"""重排序器的抽象基类模块。

本模块定义了 rag 包中所有重排序器的统一抽象接口 :class:`ReRanker`。任何具体
的重排序实现（交叉编码器 CrossEncoder、基于大模型的列表式排序等）都应继承
本类并实现核心的 ``rerank`` 方法，从而把「查询 + 初步检索结果 → 精细排序后的
结果」这一精排动作统一为对外契约。

设计意图：

- :class:`ReRanker` 负责「对多路检索得到的候选片段进行精排」，是 RAG 链路中
  检索阶段之后、Prompt 构建之前的统一入口。
- 所有重排序实现都接收并返回统一的
  :class:`vector_store.document.SearchResult` 列表：输入为初步检索的候选结果
  （通常按粗排分数降序），输出为按「query 与文档的精细相关性」重新排序后的
  结果列表。
- 与 :class:`rag.retrievers.base.Retriever` 不同，:class:`ReRanker` 的输入输出
  都是 :class:`SearchResult` 列表，允许实现**增删、重排**候选集，只需保证
  输出按相关度降序排列即可，使上层无需关心具体重排后端即可无缝替换。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from vector_store.document import SearchResult


class ReRanker(ABC):
    """重排序器的抽象基类。

    所有具体重排序器（交叉编码器、基于大模型的列表式排序等）都应继承本类，
    并实现抽象方法 :meth:`rerank`，把初步检索结果按「query 与文档的精细
    相关性」重新排序。

    契约约定：

    - :meth:`rerank` 接收查询文本 ``query`` 与候选结果 ``results``，返回
      ``list[SearchResult]``。
    - 输入输出均为 :class:`SearchResult` 列表，实现可对候选集做增删、重排，
      但返回列表必须按相关度降序排列：即 ``SearchResult.score`` **越高越靠前**
      （或等价地 ``SearchResult.distance`` 越低越靠前）。
    - 重排序过程中任何失败都应抛出
      :class:`rag.exceptions.RerankError`（或其子类），以便调用方统一捕获。
    """

    @abstractmethod
    def rerank(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        """对初步检索结果按查询精细相关性重新排序（抽象方法）。

        这是所有重排序器的核心契约，子类必须实现。

        Args:
            query: 查询文本（通常为用户的自然语言问题）。
            results: 初步检索得到的候选结果列表，通常按粗排分数降序。

        Returns:
            按「query 与文档精细相关性」降序排列的 :class:`SearchResult`
            列表，即 ``score`` 越高越靠前。

        Raises:
            RerankError: 重排序失败（如模型加载失败、推理异常等）时抛出。
        """
        ...
