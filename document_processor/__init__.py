"""document_processor 包入口：文档上游处理层。

本包负责「原始文件 → 文本 chunk」的上游处理，是 RAG 链路中
向量化与检索之前的环节。它提供三组核心能力：

- **文档加载**：:class:`DocumentLoader`（抽象接口）、:class:`PdfLoader`
  （PDF 具体实现，懒导入 pypdf）与 :class:`WordLoader`（Word 具体实现，
  懒导入 python-docx）。
- **文本分割**：:class:`TextSplitter`（抽象接口）、
  :class:`RecursiveTextSplitter`（递归分割具体实现）与
  :class:`SemanticSplitter`（语义分块具体实现，基于 embedding 相似度变化）。
- **元数据提取**：:class:`MetadataExtractor`（抽象接口）、
  :class:`BasicMetadataExtractor`（基础实现，提取文件名 / 类型 / 大小 /
  时间 / 页码 / 章节等溯源信息）与 :class:`UnstructuredMetadataExtractor`
  （基于 unstructured 的深度实现，懒导入重依赖，提取标题 / 章节 / 页码 /
  元素统计等更丰富结构信息）。

本包复用 :class:`vector_store.Document` 作为文档数据模型，并复用
:class:`vector_store.embeddings.EmbeddingClient` 作为嵌入能力，
从而与下游向量存储无缝衔接。

本模块通过显式 ``__all__`` 导出公开 API，方便以
``from document_processor import ...`` 的形式直接引用。

.. note::
    loader（:class:`DocumentLoader` / :class:`PdfLoader` /
    :class:`WordLoader`）、splitter（:class:`TextSplitter` /
    :class:`RecursiveTextSplitter` / :class:`SemanticSplitter`）与
    metadata（:class:`MetadataExtractor` / :class:`BasicMetadataExtractor` /
    :class:`UnstructuredMetadataExtractor`）已全部实现并导出。
"""

from __future__ import annotations

from document_processor.exceptions import (
    DocumentLoadError,
    DocumentProcessorError,
    DocumentSplitError,
    MetadataExtractionError,
)
from document_processor.loaders import DocumentLoader, PdfLoader, WordLoader
from document_processor.metadata import (
    BasicMetadataExtractor,
    MetadataExtractor,
    UnstructuredMetadataExtractor,
)
from document_processor.splitters import (
    RecursiveTextSplitter,
    SemanticSplitter,
    TextSplitter,
)

__all__ = [
    "DocumentProcessorError",
    "DocumentLoadError",
    "DocumentSplitError",
    "MetadataExtractionError",
    "DocumentLoader",
    "PdfLoader",
    "WordLoader",
    "TextSplitter",
    "RecursiveTextSplitter",
    "SemanticSplitter",
    "MetadataExtractor",
    "BasicMetadataExtractor",
    "UnstructuredMetadataExtractor",
]
