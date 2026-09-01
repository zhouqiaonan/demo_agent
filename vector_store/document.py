"""vector_store 包的数据模型模块。

本模块定义了向量数据库包中的两个核心数据结构：

- ``Document`` —— 一条待向量化的原始文档，包含正文文本、元数据与唯一标识。
- ``SearchResult`` —— 一次向量检索命中的结果，包含命中文档的标识、正文、
  元数据，以及相似度分数和距离两个度量指标。

字段语义约定：

- ``Document.text`` 是文档的原始正文，将由嵌入模型转换为向量。
- ``Document.metadata`` 是附着在文档上的任意键值对，用于过滤、溯源或展示。
- ``Document.id`` 是文档的唯一标识；若调用方未显式提供，则在初始化时自动
  使用 ``uuid.uuid4().hex`` 生成一个 32 位（无连字符）十六进制字符串。
- ``SearchResult.score`` 表示相似度分数，**值越高越相似**（如余弦相似度）。
- ``SearchResult.distance`` 表示距离度量，**值越低越近**（如欧氏距离或
  余弦距离，与 ``score`` 单调反向）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4


@dataclass
class Document:
    """一条待向量化的文档。

    本类是向量存储写入流程中的最小数据单元：调用方将原始文本与可选元数据
    封装为 ``Document``，随后由嵌入模块将其转换为向量并写入存储后端。

    若未显式提供 ``id``，则会在初始化时自动生成一个无连字符的 32 位十六进制
    字符串作为唯一标识，便于后续检索结果反查对应的文档。

    Attributes:
        text: 文档正文文本，将由嵌入模型转换为向量。
        metadata: 附着在文档上的元数据（任意键值对），默认空字典。
        id: 文档唯一标识。默认 ``None``，在 ``__post_init__`` 中自动生成。

    Example:
        >>> doc = Document(text="你好")
        >>> len(doc.id)
        32
        >>> doc.metadata
        {}
        >>> doc2 = Document(text="带元数据", metadata={"source": "wiki"}, id="doc-1")
        >>> doc2.id
        'doc-1'
    """

    # ---- 字段定义 ---------------------------------------------------------

    text: str
    """文档正文文本。"""

    metadata: dict[str, Any] = field(default_factory=dict)
    """附着在文档上的元数据，默认空字典。"""

    id: str | None = None
    """文档唯一标识；为 ``None`` 时在 ``__post_init__`` 中自动生成。"""

    # ---- 校验与初始化 -----------------------------------------------------

    def __post_init__(self) -> None:
        """在初始化完成后为缺失的 ``id`` 自动生成唯一标识。

        仅当调用方未显式提供 ``id``（即 ``id is None``）时，才使用
        ``uuid.uuid4().hex`` 生成一个 32 位无连字符的十六进制字符串。
        """
        if self.id is None:
            self.id = uuid4().hex


@dataclass
class SearchResult:
    """一次向量检索命中的结果。

    本类封装检索后端返回的单个命中项，用于向调用方同时提供命中文档的内容
    与匹配度量信息。

    Attributes:
        id: 命中文档的唯一标识，与 ``Document.id`` 对应。
        text: 命中文档的正文文本。
        metadata: 命中文档的元数据。
        score: 相似度分数，**值越高越相似**（如余弦相似度）。
        distance: 距离度量，**值越低越近**（如欧氏距离或余弦距离）。

    Example:
        >>> result = SearchResult(
        ...     id="doc-1", text="你好", metadata={}, score=0.95, distance=0.05
        ... )
        >>> result.score
        0.95
        >>> result.distance
        0.05
    """

    # ---- 字段定义 ---------------------------------------------------------

    id: str
    """命中文档的唯一标识。"""

    text: str
    """命中文档的正文文本。"""

    metadata: dict[str, Any]
    """命中文档的元数据。"""

    score: float
    """相似度分数，值越高表示越相似。"""

    distance: float
    """距离度量，值越低表示越接近。"""
