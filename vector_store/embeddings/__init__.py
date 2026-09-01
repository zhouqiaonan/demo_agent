"""vector_store 包的嵌入（Embedding）子包入口。

本子包负责「文本 → 向量」的转换，是向量数据库写入与检索流程的上游环节。
它定义统一的嵌入客户端抽象接口 :class:`EmbeddingClient`，并提供两种具体实现：

- :class:`OpenAIEmbedding`：基于 OpenAI API 的云端嵌入（同步 + 原生异步）。
- :class:`SentenceTransformerEmbedding`：基于 sentence-transformers 的本地
  嵌入（懒加载，需自装 torch / sentence-transformers）。

所有嵌入客户端都遵循同一契约：``embed`` 同步为核心，``aembed`` 提供异步
能力；输入一批文本，输出**与输入顺序一致**的浮点向量列表。

本模块通过显式 ``__all__`` 导出公开 API，方便以
``from vector_store.embeddings import ...`` 的形式直接引用。
"""

from vector_store.embeddings.base import EmbeddingClient
from vector_store.embeddings.openai_embedding import OpenAIEmbedding
from vector_store.embeddings.sentence_transformer import SentenceTransformerEmbedding

__all__ = [
    "EmbeddingClient",
    "OpenAIEmbedding",
    "SentenceTransformerEmbedding",
]
