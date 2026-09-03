"""document_processor 包的元数据提取（metadata）子包入口。

本子包负责从文档中提取结构化元数据，包括文件名、类型、大小、时间、
页码、章节、时间戳等溯源信息。它提供统一的元数据提取抽象接口
:class:`MetadataExtractor`，以及两类具体实现：

- :class:`BasicMetadataExtractor`：基础实现，提取文件级通用字段，并按文件
  类型追加 PDF 页码 / Word 章节等结构化字段（零额外配置）。
- :class:`UnstructuredMetadataExtractor`：基于 unstructured 的深度实现，
  懒导入重依赖 ``unstructured``，能提取标题 / 章节 / 页码 / 元素统计等
  更丰富的结构信息（可选后端）。

本模块通过显式 ``__all__`` 导出公开 API，方便以
``from document_processor.metadata import ...`` 的形式直接引用。
"""

from __future__ import annotations

from document_processor.metadata.base import MetadataExtractor
from document_processor.metadata.basic import BasicMetadataExtractor
from document_processor.metadata.unstructured_extractor import (
    UnstructuredMetadataExtractor,
)

__all__ = [
    "MetadataExtractor",
    "BasicMetadataExtractor",
    "UnstructuredMetadataExtractor",
]
