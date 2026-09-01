"""vector_store 包的向量存储核心模块：抽象接口 + ChromaDB 实现。

本模块同时定义了向量数据库的两个关键角色：

- :class:`VectorStore` —— 面向调用方的**统一抽象接口**，声明了向量存储
  必须具备的五个核心操作（写入、文本检索、向量检索、删除、计数）。
  上层业务只需面向 ``VectorStore`` 编程，即可无缝替换不同的后端实现。
- :class:`ChromaStore` —— 基于 ChromaDB 的**具体实现**，通过「懒初始化 +
  懒导入 + 同步门面 + 异步核心」的设计，兼顾易用性与性能。

整体设计要点：

1. **懒导入 chromadb（关键）**：``chromadb`` 依赖体积较大，且并非所有
   使用场景都需要真正连接 ChromaDB（例如单元测试中注入 mock client）。
   因此本模块**不在模块顶部 import chromadb**，而是把 ``import chromadb``
   语句延迟到 :meth:`ChromaStore._ensure_client` 方法体内执行。这样：
   - 仅 ``import vector_store.chroma_store`` 或实例化 ``ChromaStore``
     （不触发客户端构建）不会因缺少 chromadb 而报错。
   - 测试注入 mock ``client`` 后，``_ensure_client`` 直接返回注入对象，
     **完全不会触发** chromadb 导入，从而无需安装 chromadb 即可单测。

2. **懒初始化**：Chromadb 客户端与 collection 都延迟到首次真正读写时才
   创建（见 :meth:`ChromaStore._ensure_client` 与
   :meth:`ChromaStore._ensure_collection`），并在创建后缓存复用。

3. **同步门面 + 异步核心**：对外暴露同步方法（如 ``add_documents`` /
   ``query``），内部通过 :meth:`ChromaStore.aadd_documents` 等异步实现驱动；
   写入流程使用 ``asyncio.Semaphore`` 限制并发，最大化嵌入吞吐。

4. **维度校验**：在创建 collection 时，若集合中已存在数据，则校验既有向量
   维度是否与嵌入客户端声明的维度一致，不一致抛出
   :class:`DimensionMismatchError`，避免「先写后查」维度错配的隐蔽错误。
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any, cast

from vector_store.distance import DistanceMetric, distance_to_score
from vector_store.document import Document, SearchResult
from vector_store.embeddings.base import EmbeddingClient
from vector_store.exceptions import DimensionMismatchError

__all__ = ["VectorStore", "ChromaStore"]


# ============================================================================
# VectorStore —— 抽象接口
# ============================================================================


class VectorStore(ABC):
    """向量存储的抽象基类。

    声明所有向量存储后端必须实现的五个核心操作，作为上层业务「面向接口
    编程」的统一契约。具体后端（如 :class:`ChromaStore`）继承本类并实现
    全部抽象方法。

    契约约定：

    - ``add_documents`` 负责把一批 :class:`Document` 向量化后写入后端，
      返回按输入顺序排列的文档 id 列表。
    - ``query`` 与 ``query_by_vector`` 分别按文本、按原始向量检索最近邻，
      返回统一结构的 :class:`SearchResult` 列表。
    - ``delete`` 删除指定 id 的文档；``count`` 返回当前文档总数。
    """

    @abstractmethod
    def add_documents(
        self,
        documents: list[Document],
        chunk_size: int = 100,
        concurrency: int = 4,
    ) -> list[str]:
        """将一批文档向量化后写入向量存储（同步）。

        实现应当把 ``documents`` 按 ``chunk_size`` 分片，逐片向量化后写入
        后端，并通过 ``concurrency`` 限制并发的分片数量。

        Args:
            documents: 待写入的文档列表。
            chunk_size: 每次向量化的分片大小（文档条数），默认 100。
            concurrency: 并发处理分片的最大数量，默认 4。

        Returns:
            按输入顺序排列的、所有已写入文档的 id 列表。
        """
        ...

    @abstractmethod
    def query(
        self,
        text: str,
        k: int = 5,
        where: dict | None = None,
    ) -> list[SearchResult]:
        """按文本查询与给定文本最相似的 ``k`` 条文档。

        实现通常先将 ``text`` 向量化，再转调 ``query_by_vector``。

        Args:
            text: 查询文本。
            k: 返回的最相似结果数量，默认 5。
            where: 可选的元数据过滤条件；为 ``None`` 表示不过滤。

        Returns:
            按相似度分数降序排列的 :class:`SearchResult` 列表。
        """
        ...

    @abstractmethod
    def query_by_vector(
        self,
        vector: list[float],
        k: int = 5,
        where: dict | None = None,
    ) -> list[SearchResult]:
        """按原始向量查询与之最相似的 ``k`` 条文档。

        Args:
            vector: 查询向量。
            k: 返回的最相似结果数量，默认 5。
            where: 可选的元数据过滤条件；为 ``None`` 表示不过滤。

        Returns:
            按相似度分数降序排列的 :class:`SearchResult` 列表。
        """
        ...

    @abstractmethod
    def delete(self, ids: list[str]) -> None:
        """删除指定 id 的文档。

        Args:
            ids: 待删除的文档 id 列表。
        """
        ...

    @abstractmethod
    def count(self) -> int:
        """返回当前向量存储中的文档总数。

        Returns:
            已写入的文档数量。
        """
        ...


# ============================================================================
# ChromaStore —— ChromaDB 实现
# ============================================================================


class ChromaStore(VectorStore):
    """基于 ChromaDB 的向量存储实现。

    本类通过「懒导入 + 懒初始化」降低外部依赖成本，并通过「同步门面 +
    异步核心」提供并发友好的写入路径。

    设计说明：

    - **懒导入**：``chromadb`` 的导入延迟到 :meth:`_ensure_client` 内部执行，
      未真正使用 ChromaDB 的场景（如注入 mock client 的单测）不会触发导入。
    - **懒初始化**：客户端与 collection 在首次读写时才创建并缓存。
    - **可注入客户端**：构造时可传入 ``client``（如测试用的 mock 对象），
      此时 ``_ensure_client`` 直接返回注入对象，跳过 chromadb 导入与实例化。
    - **持久化**：传入 ``persist_directory`` 时使用
      ``chromadb.PersistentClient``，否则使用内存版 ``chromadb.Client``。

    Attributes:
        _embedding: 关联的嵌入客户端，负责「文本 → 向量」转换。
        _collection_name: ChromaDB collection 的名称。
        _metric: 向量空间的距离度量（决定检索时的相似度映射方式）。
        _client: 注入或懒创建的 chromadb 客户端；为 ``None`` 表示尚未初始化。
        _persist_directory: 持久化目录；为 ``None`` 表示使用内存客户端。
        _collection: 懒创建的 chromadb collection；为 ``None`` 表示尚未创建。
    """

    def __init__(
        self,
        embedding: EmbeddingClient,
        collection_name: str = "default",
        metric: DistanceMetric = DistanceMetric.COSINE,
        client: Any | None = None,
        persist_directory: str | None = None,
    ) -> None:
        """初始化 ChromaStore。

        本构造方法**不会**导入 chromadb 或创建任何后端连接（懒初始化），
        因此无论是否安装 chromadb 均可安全构造实例。

        Args:
            embedding: 嵌入客户端，负责把文本转换为向量。
            collection_name: ChromaDB collection 名称，默认 ``"default"``。
            metric: 向量空间的距离度量，默认
                :attr:`DistanceMetric.COSINE`。
            client: 可注入的 chromadb 客户端。若提供（如测试 mock），则
                ``_ensure_client`` 直接返回该对象，**不会**触发 chromadb 导入。
            persist_directory: 持久化目录路径。若提供，懒初始化时使用
                ``chromadb.PersistentClient``；否则使用内存版
                ``chromadb.Client``。
        """
        self._embedding: EmbeddingClient = embedding
        """关联的嵌入客户端。"""

        self._collection_name: str = collection_name
        """ChromaDB collection 名称。"""

        self._metric: DistanceMetric = metric
        """向量空间的距离度量。"""

        self._client: Any | None = client
        """注入或懒创建的 chromadb 客户端；``None`` 表示尚未初始化。"""

        self._persist_directory: str | None = persist_directory
        """持久化目录；``None`` 表示使用内存客户端。"""

        self._collection: Any | None = None
        """懒创建的 chromadb collection；``None`` 表示尚未创建。"""

    # ------------------------------------------------------------------
    # 懒初始化
    # ------------------------------------------------------------------

    def _ensure_client(self) -> Any:
        """确保 chromadb 客户端已就绪，并返回该客户端。

        逻辑：
        1. 若构造时注入了 ``client``，直接返回该对象（**不导入 chromadb**）。
        2. 否则懒导入 ``chromadb``（首次调用才 import）。
        3. 依据 ``persist_directory`` 选择 ``PersistentClient``（持久化）或
           ``Client``（内存版）。
        4. 将结果缓存到 ``self._client`` 复用。

        Returns:
            已就绪的 chromadb 客户端实例。

        Raises:
            ImportError: 未安装 chromadb 且未注入 client 时抛出。
        """
        if self._client is not None:
            return self._client

        import chromadb

        if self._persist_directory is not None:
            self._client = chromadb.PersistentClient(path=self._persist_directory)
        else:
            self._client = chromadb.Client()

        return self._client

    def _ensure_collection(self) -> Any:
        """确保 chromadb collection 已就绪，并返回该 collection。

        逻辑：
        1. 若已创建，直接返回缓存。
        2. 调用 :meth:`_ensure_client` 获取客户端，再用
           ``get_or_create_collection`` 创建（或复用）collection，并在
           ``metadata`` 中写入距离度量 ``{"hnsw:space": metric.value}``。
        3. **维度校验**：若该 collection 已有数据（``count() > 0``），通过
           ``get(limit=1, include=["embeddings"])`` 取一条既有向量，比较其
           长度与 ``embedding.dimension``；不一致则抛出
           :class:`DimensionMismatchError`。空 collection 跳过校验（维度由
           首次 ``add`` 决定，chromadb 会兜底校验）。
        4. 缓存到 ``self._collection`` 复用。

        Returns:
            已就绪的 chromadb collection 实例。

        Raises:
            DimensionMismatchError: 既有数据维度与嵌入客户端维度不一致时抛出。
        """
        if self._collection is not None:
            return self._collection

        client: Any = self._ensure_client()
        collection: Any = client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": self._metric.value},
        )

        # ---- 维度校验：仅当集合中已有数据时执行 ----
        if collection.count() > 0:
            existing: dict[str, Any] = collection.get(
                limit=1, include=["embeddings"]
            )
            existing_embeddings: list[Any] = existing.get("embeddings") or []
            if existing_embeddings:
                actual_dim: int = len(existing_embeddings[0])
                expected_dim: int = self._embedding.dimension
                if actual_dim != expected_dim:
                    raise DimensionMismatchError(
                        expected_dim=expected_dim,
                        actual_dim=actual_dim,
                    )

        self._collection = collection
        return self._collection

    # ------------------------------------------------------------------
    # 写入：同步门面 + 异步核心
    # ------------------------------------------------------------------

    async def aadd_documents(
        self,
        documents: list[Document],
        chunk_size: int = 100,
        concurrency: int = 4,
    ) -> list[str]:
        """将一批文档向量化后写入 ChromaDB（异步核心）。

        处理流程：
        1. 先 :meth:`_ensure_client` 与 :meth:`_ensure_collection`。
        2. 将 ``documents`` 按 ``chunk_size`` 分片。
        3. 用 ``asyncio.Semaphore(concurrency)`` 限制并发，对每个分片并发调用
           ``embedding.aembed(chunk_texts)`` 生成向量。
        4. 每个分片组装 ``ids`` / ``embeddings`` / ``documents`` /
           ``metadatas`` 后调用 ``collection.add`` 写入。其中 ``metadatas``
           中若某文档 ``metadata`` 为空 dict，则该位置置为 ``None``，避免
           chromadb 对空 dict 的兼容问题。
        5. 按输入顺序汇总所有分片的 id 列表并返回。

        Args:
            documents: 待写入的文档列表。
            chunk_size: 每次向量化的分片大小（文档条数），默认 100。
            concurrency: 并发处理分片的最大数量，默认 4。

        Returns:
            按输入顺序排列的、所有已写入文档的 id 列表。

        Raises:
            ValueError: 当 ``chunk_size`` 或 ``concurrency`` 非正整数时抛出。
        """
        if chunk_size <= 0:
            raise ValueError(f"chunk_size 必须为正整数，实际为 {chunk_size!r}")
        if concurrency <= 0:
            raise ValueError(f"concurrency 必须为正整数，实际为 {concurrency!r}")

        self._ensure_client()
        collection: Any = self._ensure_collection()

        # ---- 分片 ----
        chunks: list[list[Document]] = [
            documents[i : i + chunk_size]
            for i in range(0, len(documents), chunk_size)
        ]

        semaphore: asyncio.Semaphore = asyncio.Semaphore(concurrency)

        async def _process_chunk(chunk: list[Document]) -> list[str]:
            """并发处理单个分片：向量化后写入 collection。

            Args:
                chunk: 单个分片内的文档列表。

            Returns:
                该分片内文档的 id 列表（按输入顺序）。
            """
            async with semaphore:
                chunk_texts: list[str] = [d.text for d in chunk]
                embeddings: list[list[float]] = await self._embedding.aembed(
                    chunk_texts
                )

                # Document.__post_init__ 保证 id 恒为非 None，这里用 cast 收窄类型。
                ids: list[str] = [cast(str, d.id) for d in chunk]
                documents_texts: list[str] = [d.text for d in chunk]
                # 空 dict 的 metadata 用 None 占位，规避 chromadb 兼容问题。
                metadatas: list[dict[str, Any] | None] = [
                    d.metadata if d.metadata else None for d in chunk
                ]

                collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    documents=documents_texts,
                    metadatas=metadatas,
                )
                return ids

        # ---- 并发执行所有分片 ----
        chunk_id_lists: list[list[str]] = await asyncio.gather(
            *(_process_chunk(chunk) for chunk in chunks)
        )

        # ---- 按顺序汇总 ----
        all_ids: list[str] = []
        for chunk_ids in chunk_id_lists:
            all_ids.extend(chunk_ids)
        return all_ids

    def add_documents(
        self,
        documents: list[Document],
        chunk_size: int = 100,
        concurrency: int = 4,
    ) -> list[str]:
        """将一批文档向量化后写入 ChromaDB（同步门面）。

        本方法是异步核心 :meth:`aadd_documents` 的同步包装，内部通过
        ``asyncio.run`` 运行。

        注意：若在**已有运行中的事件循环**（如 Jupyter Notebook 或异步框架
        如 FastAPI 的路由内）里调用，``asyncio.run`` 会抛出 ``RuntimeError``。
        此种场景请改用 ``await aadd_documents(...)``。

        Args:
            documents: 待写入的文档列表。
            chunk_size: 每次向量化的分片大小（文档条数），默认 100。
            concurrency: 并发处理分片的最大数量，默认 4。

        Returns:
            按输入顺序排列的、所有已写入文档的 id 列表。

        Raises:
            RuntimeError: 在已存在事件循环的环境中调用时抛出。
        """
        return asyncio.run(
            self.aadd_documents(documents, chunk_size, concurrency)
        )

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------

    def query(
        self,
        text: str,
        k: int = 5,
        where: dict | None = None,
    ) -> list[SearchResult]:
        """按文本查询与给定文本最相似的 ``k`` 条文档。

        实现：先将 ``text`` 用 ``embedding.embed`` 向量化（取第一个向量），
        再转调 :meth:`query_by_vector` 完成检索。

        Args:
            text: 查询文本。
            k: 返回的最相似结果数量，默认 5。
            where: 可选的元数据过滤条件；为 ``None`` 表示不过滤。

        Returns:
            按相似度分数降序排列的 :class:`SearchResult` 列表。
        """
        vector: list[float] = self._embedding.embed([text])[0]
        return self.query_by_vector(vector, k, where)

    def query_by_vector(
        self,
        vector: list[float],
        k: int = 5,
        where: dict | None = None,
    ) -> list[SearchResult]:
        """按原始向量查询与之最相似的 ``k`` 条文档。

        实现：先 :meth:`_ensure_collection`，再调用
        ``collection.query(query_embeddings=[vector], n_results=k, ...)``。
        ChromaDB 返回的 ``ids`` / ``documents`` / ``metadatas`` /
        ``distances`` 为四个并列的二维列表（一维为单条查询，这里仅查询一个
        向量，故取下标 0），随后用 :func:`distance_to_score` 把距离映射为
        「越大越相似」的分数，构造 :class:`SearchResult` 返回。

        注意：``where=None`` 时**不向 chromadb 传递** ``where`` 参数，避免
        chromadb 对 ``None`` 的语义差异（可能被当作无过滤的显式参数处理）。

        Args:
            vector: 查询向量。
            k: 返回的最相似结果数量，默认 5。
            where: 可选的元数据过滤条件；为 ``None`` 表示不过滤。

        Returns:
            按相似度分数降序排列的 :class:`SearchResult` 列表。
        """
        collection: Any = self._ensure_collection()

        kwargs: dict[str, Any] = {
            "query_embeddings": [vector],
            "n_results": k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where is not None:
            kwargs["where"] = where

        result: dict[str, Any] = collection.query(**kwargs)

        raw_ids: list[list[Any]] = result.get("ids") or [[]]
        raw_documents: list[list[Any]] = result.get("documents") or [[]]
        raw_metadatas: list[list[Any]] = result.get("metadatas") or [[]]
        raw_distances: list[list[Any]] = result.get("distances") or [[]]

        ids: list[Any] = raw_ids[0]
        documents: list[Any] = raw_documents[0]
        metadatas: list[Any] = raw_metadatas[0]
        distances: list[Any] = raw_distances[0]

        search_results: list[SearchResult] = []
        for doc_id, doc_text, doc_meta, distance in zip(
            ids, documents, metadatas, distances
        ):
            score: float = distance_to_score(self._metric, float(distance))
            search_results.append(
                SearchResult(
                    id=str(doc_id),
                    text=doc_text if doc_text is not None else "",
                    metadata=doc_meta if doc_meta is not None else {},
                    score=score,
                    distance=float(distance),
                )
            )
        return search_results

    # ------------------------------------------------------------------
    # 删除与计数
    # ------------------------------------------------------------------

    def delete(self, ids: list[str]) -> None:
        """删除指定 id 的文档。

        先 :meth:`_ensure_collection` 确保集合就绪，再调用
        ``collection.delete(ids=ids)``。

        Args:
            ids: 待删除的文档 id 列表。
        """
        collection: Any = self._ensure_collection()
        collection.delete(ids=ids)

    def count(self) -> int:
        """返回当前 ChromaDB collection 中的文档总数。

        先 :meth:`_ensure_collection` 确保集合就绪，再返回
        ``collection.count()``。

        Returns:
            已写入的文档数量。
        """
        collection: Any = self._ensure_collection()
        return int(collection.count())
