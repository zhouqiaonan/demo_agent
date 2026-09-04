"""基于 RRF（Reciprocal Rank Fusion）的混合检索器实现。

本模块提供 :class:`HybridRetriever`，是 :class:`Retriever` 抽象接口的一个具体
实现：它不自行执行任何检索，而是持有多个检索器（如 :class:`VectorRetriever` 与
:class:`KeywordRetriever`），分别让它们对同一查询各检索一次，再用 **RRF（倒数
排名融合）** 把多路结果合并成一路按融合分数降序排列的 Top-K 命中列表。

RRF 算法原理简述：

    对文档 d，其融合分数定义为各检索器贡献之和：

        RRF(d) = Σ_{r ∈ retrievers} 1 / (k + rank_d(r))

    其中 ``rank_d(r)`` 是文档 d 在检索器 r 的结果列表中的**排名**（1-based，
    即第 1 名的 rank=1），``k`` 是平滑常数（默认 60）。

    该算法之所以只看「排名」而**不看各检索器原始分数**，关键在于：

    - **量纲无关**：向量检索的余弦相似度、关键词检索的 BM25 分数、甚至其它
      检索器的 TF-IDF 分数，它们的取值范围与分布截然不同，无法直接相加或比较。
      而「排名」是一个统一的、无量纲的序数，天然可跨检索器融合。
    - **k 的作用**：``k`` 作为平滑常数加在排名上，避免排名极端小（如第 1 名
      贡献 ``1/1 = 1.0``）时主导融合结果；``k`` 越大，各排名的贡献越接近，
      排名靠后的文档也有机会进入最终列表；``k`` 越小，头部排名的话语权越强。
    - **多路认可加分**：同一个 chunk（以 ``id`` 标识）可能同时被多个检索器
      命中——例如既被向量检索命中又被关键词检索命中。此时它的 RRF 分数会把
      各检索器的贡献**累加**，从而「多路都认可」的文档排名天然更高，这正是
      混合检索提升召回质量的核心机制。

设计要点：

- **纯组合模式**：本类对注入的检索器只依赖 ``retrieve(query, k) -> list[
  SearchResult]`` 契约，不关心每个检索器内部的实现细节，因而任意
  :class:`Retriever` 子类（向量 / 关键词 / 自定义）都可无侵入地参与融合。
- **距离语义映射**：RRF 分数是「越大越相关」，与 ``SearchResult.score`` 语义
  一致；为保持 ``SearchResult.distance``「越低越近」的约定，命中结果统一用
  ``distance = -score`` 填充。
- **文档信息来源**：以 ``id`` 为键保留首次命中的 ``SearchResult`` 作为文档
  信息（text / metadata）来源，融合阶段只重算分数，不改写文档内容。
"""

from __future__ import annotations

from vector_store.document import SearchResult

from rag.exceptions import RetrievalError
from rag.retrievers.base import Retriever


class HybridRetriever(Retriever):
    """基于 RRF（Reciprocal Rank Fusion）的混合检索器实现。

    本类是对 :class:`Retriever` 抽象接口的混合检索实现：持有一组检索器，对同一
    查询分别检索后，按「倒数排名融合」公式把多路结果的排名转换为统一的融合
    分数，返回融合分数降序排列的 Top-K 命中结果。

    核心价值在于把「向量检索 + 关键词检索」这类**异构、量纲不同的分数**统一到
    「排名」这一无量纲维度上做公平融合：一方面避免某一检索器因分数天然偏大而
    主导结果，另一方面让「被多路同时认可」的文档因分数累加而排名靠前。

    Attributes:
        _retrievers: 参与融合的检索器列表（至少一个）。
        _rrf_k: RRF 平滑常数，加在排名上以避免头部排名主导，默认 60。
    """

    def __init__(self, retrievers: list[Retriever], rrf_k: int = 60) -> None:
        """初始化混合检索器。

        Args:
            retrievers: 参与融合的检索器列表，不能为空。
            rrf_k: RRF 平滑常数，必须为正整数，默认 60。``k`` 越大，各排名的
                贡献越接近（更平滑）；``k`` 越小，头部排名话语权越强。

        Raises:
            RetrievalError: ``retrievers`` 为空，或 ``rrf_k`` 非正数时抛出。
        """
        if not retrievers:
            raise RetrievalError("混合检索器至少需要一个子检索器，不能为空列表。")
        if rrf_k <= 0:
            raise RetrievalError(
                f"RRF 平滑常数 k 必须为正整数，当前传入 {rrf_k}。"
            )

        self._retrievers: list[Retriever] = retrievers
        """参与融合的检索器列表。"""

        self._rrf_k: int = rrf_k
        """RRF 平滑常数，加在排名上以避免头部排名主导。"""

    def retrieve(self, query: str, k: int = 5) -> list[SearchResult]:
        """对多个检索器的结果做 RRF 融合，返回最相关的 ``k`` 条结果。

        流程：

        1. 依次调用每个检索器的 ``retrieve(query, k)``，得到各自的 Top-K 结果。
        2. 用两个 dict 做聚合：``scores``（``id → 累计 RRF 分数``）累加每个文档
           在各检索器中的倒数排名贡献；``by_id``（``id → SearchResult``）保留
           文档首次命中的 text / metadata，作为最终构造的信息来源。
        3. 遍历每个检索器的结果，``enumerate(results, start=1)`` 得到 1-based
           排名 ``rank``，累加 ``scores[rid] += 1.0 / (k + rank)``；同一文档被
           多路命中时分数自然累加，实现「多路认可加分」。
        4. 按 RRF 分数降序排序，取前 ``k`` 个。
        5. 每个命中构造新的 :class:`SearchResult`，其中 ``score=rrf_score``、
           ``distance=-rrf_score``，以保持「distance 越低越近」的语义一致。

        Args:
            query: 查询文本（通常为用户的自然语言问题）。
            k: 返回的最相关结果数量，默认 5。该 ``k`` 同时用作每个子检索器的
                候选数量，即每个子检索器各自返回 Top-``k`` 后再融合。

        Returns:
            按 RRF 融合分数降序排列的 :class:`SearchResult` 列表（分数越高越靠
            前）。若所有子检索器均未命中（结果为空），返回空列表。

        Raises:
            RetrievalError: ``k`` 非正数时抛出（k 必须为正整数）；或任一子检索器
                检索失败时由其内部上抛（本方法不额外捕获），以保留失败发生的
                具体阶段信息。
        """
        if k <= 0:
            raise RetrievalError(f"检索数量 k 必须为正整数，当前传入 {k}。")

        scores: dict[str, float] = {}
        """文档 id → 累计 RRF 融合分数。"""

        by_id: dict[str, SearchResult] = {}
        """文档 id → 首次命中的 SearchResult，用于最终构造时取 text/metadata。"""

        for retriever in self._retrievers:
            results: list[SearchResult] = retriever.retrieve(query, k)
            for rank, result in enumerate(results, start=1):
                rid: str = result.id
                scores[rid] = scores.get(rid, 0.0) + 1.0 / (self._rrf_k + rank)
                by_id.setdefault(rid, result)

        if not scores:
            return []

        # 按 RRF 分数降序排序；分数相同时按 id 稳定排序，保证结果可复现。
        ranked: list[tuple[str, float]] = sorted(
            scores.items(), key=lambda item: (-item[1], item[0])
        )

        merged: list[SearchResult] = []
        for rid, rrf_score in ranked[:k]:
            document: SearchResult = by_id[rid]
            merged.append(
                SearchResult(
                    id=rid,
                    text=document.text,
                    metadata=document.metadata,
                    score=rrf_score,
                    distance=-rrf_score,
                )
            )
        return merged
