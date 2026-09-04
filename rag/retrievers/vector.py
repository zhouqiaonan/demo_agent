"""基于向量相似度的检索器实现。

本模块提供 :class:`VectorRetriever`，是 :class:`Retriever` 抽象接口的一个具体
实现：它通过依赖注入的方式持有一个向量存储后端（:class:`ChromaStore`），把
「给定查询返回 Top-K 相关 chunk」的动作直接委托给后端的向量相似度检索能力。

设计要点：

- **依赖注入**：构造时注入任意实现了 ``query(text, k) -> list[SearchResult]``
  契约的向量存储实例，检索逻辑与具体后端解耦，便于测试时替换为 mock。
- **纯委托**：:meth:`retrieve` 不自行实现向量化或排序，而是直接复用
  ``store.query`` 的既有能力（``ChromaStore.query`` 已返回按相似度降序排列的
  :class:`SearchResult` 列表）。
- **懒依赖**：本模块仅持有 ``ChromaStore`` 的类型引用，不触发 ChromaDB 等
  重依赖的导入。
"""

from __future__ import annotations

from vector_store.chroma_store import ChromaStore
from vector_store.document import SearchResult

from rag.exceptions import RetrievalError
from rag.retrievers.base import Retriever


class VectorRetriever(Retriever):
    """基于向量相似度的检索器实现。

    本类是对 :class:`Retriever` 抽象接口的向量检索实现：将查询文本交由注入的
    向量存储后端做语义相似度检索，返回按相似度降序排列的 Top-K 命中结果。

    说明：
        这里依赖注入的 ``store`` 类型标注为 :class:`ChromaStore`，但只要满足
        ``query(text, k) -> list[SearchResult]`` 契约，任意 ``VectorStore`` 实现
        都可以作为检索后端传入，从而在需要时无缝替换不同的向量数据库。

    Attributes:
        _store: 注入的向量存储后端，负责实际的「文本 → Top-K 结果」检索。
    """

    def __init__(self, store: ChromaStore) -> None:
        """初始化向量检索器。

        Args:
            store: 向量存储后端实例（如 :class:`ChromaStore`）。检索时会把
                查询文本委托给该实例的 ``query`` 方法。
        """
        self._store: ChromaStore = store
        """注入的向量存储后端，负责实际的向量相似度检索。"""

    def retrieve(self, query: str, k: int = 5) -> list[SearchResult]:
        """根据查询文本执行向量相似度检索，返回最相关的 ``k`` 条结果。

        本方法不自行实现向量化与排序，而是直接委托 ``self._store.query`` 完成
        检索（``ChromaStore.query`` 内部会先向量化查询文本，再返回按相似度
        分数降序排列的 :class:`SearchResult` 列表）。

        Args:
            query: 查询文本（通常为用户的自然语言问题）。
            k: 返回的最相关结果数量，默认 5。

        Returns:
            按相似度分数降序排列的 :class:`SearchResult` 列表，即 ``score``
            越高越靠前。

        Raises:
            RetrievalError: ``k`` 非正数时抛出（k 必须为正整数）；或底层向量存储
                检索失败时抛出（由 store 内部上抛）。
        """
        if k <= 0:
            raise RetrievalError(f"检索数量 k 必须为正整数，当前传入 {k}。")

        return self._store.query(query, k)
