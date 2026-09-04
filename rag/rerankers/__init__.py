"""rag.rerankers 子包：检索结果重排序层。

本子包负责对多路检索得到的候选片段进行精排，以提升最终上下文中
与问题最相关片段的排序质量，提供：

- :class:`ReRanker`：重排序器抽象接口（**已实现**），声明核心的 ``rerank``
  方法，统一「初步检索结果 → 精细排序」的对外契约。
- :class:`CrossEncoderReRanker`：基于交叉编码器（CrossEncoder）的具体实现
  （**已实现**），重依赖采用懒加载方式按需引入，适合对粗排 Top-K 做精排。

本模块通过显式 ``__all__`` 导出公开 API，方便以
``from rag.rerankers import ...`` 的形式直接引用。
"""

from __future__ import annotations

from rag.rerankers.base import ReRanker
from rag.rerankers.cross_encoder import CrossEncoderReRanker

__all__ = [
    "ReRanker",
    "CrossEncoderReRanker",
]
