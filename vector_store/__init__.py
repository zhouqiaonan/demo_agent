"""vector_store 包入口：向量数据库抽象层。

本包提供与后端无关的向量数据库能力，覆盖「文本 → 向量 → 存储 → 检索」
的完整链路，定位为一套可插拔、可替换后端的向量存储抽象层。核心组成如下：

- **存储层**：:class:`VectorStore`（统一抽象接口）与 :class:`ChromaStore`
  （基于 ChromaDB 的具体实现，懒导入 + 懒初始化，无需 chromadb 也能安全
  实例化）。
- **嵌入层**：:class:`EmbeddingClient`（嵌入客户端抽象接口）及其两种后端
  实现 :class:`OpenAIEmbedding`（云端）与 :class:`SentenceTransformerEmbedding`
  （本地，懒加载，需自装 torch）。
- **数据模型**：:class:`Document`（待向量化文档）与 :class:`SearchResult`
  （检索命中结果）。
- **距离度量**：:class:`DistanceMetric`（余弦 / L2 / 内积度量枚举）。
- **异常体系**：:class:`VectorStoreError`（基类）、:class:`EmbeddingError`
  （嵌入阶段错误）、:class:`DimensionMismatchError`（维度不匹配错误）。

本模块通过显式 ``__all__`` 导出公开 API，方便以
``from vector_store import ...`` 的形式直接引用。
"""

from __future__ import annotations

from vector_store.chroma_store import ChromaStore, VectorStore
from vector_store.distance import DistanceMetric
from vector_store.document import Document, SearchResult
from vector_store.embeddings.base import EmbeddingClient
from vector_store.embeddings.openai_embedding import OpenAIEmbedding
from vector_store.embeddings.sentence_transformer import SentenceTransformerEmbedding
from vector_store.exceptions import (
    DimensionMismatchError,
    EmbeddingError,
    VectorStoreError,
)

__all__ = [
    "VectorStore",
    "ChromaStore",
    "EmbeddingClient",
    "OpenAIEmbedding",
    "SentenceTransformerEmbedding",
    "Document",
    "SearchResult",
    "DistanceMetric",
    "VectorStoreError",
    "EmbeddingError",
    "DimensionMismatchError",
]
