"""document_processor 包的文档加载（loaders）子包入口。

本子包负责「原始文件 → 文本」的加载，支持 PDF / Word 等常见文档格式。
它提供统一的文档加载抽象接口 :class:`DocumentLoader`，以及具体的格式
实现：:class:`PdfLoader`（基于 ``pypdf`` 的 PDF 加载器）与
:class:`WordLoader`（基于 ``python-docx`` 的 Word 加载器）。

本模块通过显式 ``__all__`` 导出公开 API，方便以
``from document_processor.loaders import ...`` 的形式直接引用。
"""

from __future__ import annotations

from document_processor.loaders.base import DocumentLoader
from document_processor.loaders.pdf_loader import PdfLoader
from document_processor.loaders.word_loader import WordLoader

__all__ = [
    "DocumentLoader",
    "PdfLoader",
    "WordLoader",
]
