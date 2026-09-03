"""元数据提取器的抽象基类模块。

本模块定义了 document_processor 包中所有元数据提取器的统一抽象接口
:class:`MetadataExtractor`。任何具体的元数据提取实现（如基础元数据提取、
基于 unstructured 的深度提取等）都应继承本类并实现核心的 ``extract`` 方法，
从而把外部文件的溯源信息（文件名 / 类型 / 大小 / 时间 / 页码 / 章节等）
统一解析为键值对字典，供下游构建 :class:`vector_store.document.Document`
的 ``metadata`` 字段使用。

设计意图：

- ``MetadataExtractor`` 负责「文件路径 → 结构化元数据字典」的转换，
  与 :class:`~document_processor.loaders.base.DocumentLoader` 互补：后者
  负责抽取正文内容，本抽象层负责抽取文件级 / 结构级的溯源信息。
- 本抽象层统一了不同提取器的对外契约，上层业务只需面向
  ``MetadataExtractor`` 编程，即可无缝替换不同的元数据提取后端。
- 具体实现应遵循「懒导入」原则，将重依赖（如 ``pypdf``、``python-docx``
  等）延迟到 ``extract`` 方法体内 import，从而在仅引用本抽象基类时无需
  安装对应依赖。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class MetadataExtractor(ABC):
    """元数据提取器的抽象基类。

    所有具体的元数据提取实现（基础提取、unstructured 深度提取等）都应继承
    本类，并实现抽象方法 :meth:`extract`，把外部文件解析为一组键值对形式的
    元数据字典。

    契约约定：

    - :meth:`extract` 返回一个 ``dict[str, Any]``，键为元数据字段名（如
      ``"filename"``、``"file_type"``、``"size"`` 等），值为对应字段的取值。
    - 不同文件类型可能返回不同的字段集合：所有文件都具备的基础字段之外，
      具体实现还可按文件类型追加 ``page_count``（PDF 页码）、``sections``
      （Word 章节）等类型专属字段。
    - 提取过程中任何失败（文件不存在、格式不支持、内容损坏、字段缺失或
      非法等）都应抛出
      :class:`document_processor.exceptions.MetadataExtractionError`，以便
      调用方统一捕获。
    """

    @abstractmethod
    def extract(self, path: str | Path) -> dict[str, Any]:
        """从指定文件提取元数据并返回键值对字典（抽象方法）。

        这是所有元数据提取器的核心契约，子类必须实现。返回的字典应至少包含
        ``filename``（文件名）与 ``file_type``（文件类型）等基础溯源字段，
        并根据文件类型按需追加页码、章节、时间戳等更细粒度的字段。

        Args:
            path: 待提取元数据的文件路径（字符串或 :class:`pathlib.Path`）。

        Returns:
            以键值对形式组织的元数据字典；键为字段名（``str``），值为对应
            字段的取值（``Any``）。

        Raises:
            MetadataExtractionError: 文件不存在、不是文件、格式不支持或解析
                失败时抛出。
        """
        ...
