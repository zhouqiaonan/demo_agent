"""测试用的确定性伪嵌入客户端。

本模块提供 :class:`HashEmbedding`，它实现了 :class:`EmbeddingClient` 抽象
接口，用于在单元测试中替代真实的嵌入模型（如 OpenAI 或 sentence-transformers），
从而让测试完全离线、确定且无需安装任何外部依赖。

核心思想：用 ``hashlib.sha256`` 把每个文本映射为确定性的 32 字节摘要，再把
这些字节映射到 ``[-1.0, 1.0]`` 区间的 ``dimension`` 个浮点。这样：
- 同一个文本永远得到同一个向量（确定性）；
- 不同文本（几乎必然）得到不同的向量；
- 输出向量顺序与输入文本顺序严格一致。
"""

from __future__ import annotations

import hashlib

from vector_store.embeddings.base import EmbeddingClient


class HashEmbedding(EmbeddingClient):
    """基于 SHA-256 哈希的确定性伪嵌入客户端。

    用于单元测试中替代真实嵌入模型，保证结果离线、确定、可复现：

    - ``embed`` 对每个文本计算 ``sha256(text.encode("utf-8")).digest()``，
      再把摘要中的字节映射为 ``[-1.0, 1.0]`` 区间的 ``dimension`` 个浮点。
    - ``dimension`` 属性返回构造时指定的维度。
    - ``aembed`` 复用基类 ``asyncio.to_thread`` 的默认实现。

    Attributes:
        _dimension: 每个输出向量的维度。
    """

    def __init__(self, dimension: int = 16) -> None:
        """初始化伪嵌入客户端。

        Args:
            dimension: 每个输出向量的维度，默认 16。
        """
        self._dimension: int = dimension
        """每个输出向量的维度。"""

    @property
    def dimension(self) -> int:
        """返回输出向量的维度。

        Returns:
            每个嵌入向量的维度（正整数）。
        """
        return self._dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        """将一批文本映射为确定性的浮点向量。

        Args:
            texts: 待嵌入的文本列表。

        Returns:
            与输入顺序一致的浮点向量列表，每个向量长度为 ``dimension``。
        """
        vectors: list[list[float]] = []
        for text in texts:
            digest: bytes = hashlib.sha256(text.encode("utf-8")).digest()
            # 将字节（0~255）线性映射到 [-1.0, 1.0]：0 -> -1.0，255 -> 1.0。
            vector: list[float] = [
                (digest[i % len(digest)] / 127.5) - 1.0
                for i in range(self._dimension)
            ]
            vectors.append(vector)
        return vectors
