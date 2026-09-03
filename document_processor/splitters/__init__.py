"""document_processor 包的文本分割（splitters）子包入口。

本子包负责「文本 → chunk」的分割，提供统一的文本分割抽象接口
:class:`TextSplitter`，以及具体实现：

- :class:`RecursiveTextSplitter`：基于 ``langchain_text_splitters`` 的递归
  分割器，按一组中英混合友好的递归分隔符把长文本切分为长度可控的 chunk。
- :class:`SemanticSplitter`：语义分块分割器，基于 embedding 相似度变化在
  语义边界处断句，可注入任意 :class:`EmbeddingClient` 嵌入后端。

本模块通过显式 ``__all__`` 导出公开 API，方便以
``from document_processor.splitters import ...`` 的形式直接引用。
"""

from __future__ import annotations

from document_processor.splitters.base import TextSplitter
from document_processor.splitters.recursive import RecursiveTextSplitter
from document_processor.splitters.semantic import SemanticSplitter

__all__ = [
    "TextSplitter",
    "RecursiveTextSplitter",
    "SemanticSplitter",
]
