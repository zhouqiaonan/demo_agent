"""基于 unstructured 的元数据提取器模块。

本模块实现 :class:`UnstructuredMetadataExtractor`，作为 document_processor
元数据提取能力的一个**可选后端**：它依赖第三方库 ``unstructured``，能够
从文档中提取比 :class:`BasicMetadataExtractor` 更丰富的结构信息（标题 /
章节 / 页码 / 元素统计）。设计要点：

- **懒导入重依赖**：``unstructured`` 是重量级依赖，且通常要求系统级
  依赖（``libmagic`` / ``poppler`` / ``tesseract``）配合才能完整工作。
  因此本模块**不在模块顶部 import** 该库，而是延迟到 :meth:`extract`
  方法体内执行；只有用户显式实例化并调用时才真正触发 import。这样在未
  安装 ``unstructured`` 的环境中，仅 ``import`` 本模块不会报错。

- **统一异常契约**：文件不存在、``unstructured`` 未安装、文档解析失败等
  任何错误都会统一捕获并转抛为
  :class:`document_processor.exceptions.MetadataExtractionError`，并保留
  原始异常链（``raise ... from exc``），便于调用方定位根因。其中「未安装
  unstructured」会给出清晰的安装提示。

- **安全访问跨版本结构**：unstructured 不同版本的元素（Element）及其
  元数据结构存在差异，因此本实现一律使用 :func:`getattr` 安全访问
  ``category`` / ``text`` / ``metadata.page_number`` 等字段，并提供合理
  的默认值（``""`` / ``None``），避免因字段缺失而崩溃。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from document_processor._utils import _DEFAULT_MAX_FILE_SIZE
from document_processor.exceptions import MetadataExtractionError
from document_processor.metadata.base import MetadataExtractor


class UnstructuredMetadataExtractor(MetadataExtractor):
    """基于 unstructured 的深度元数据提取器（可选后端）。

    本类通过 ``unstructured.partition.auto.partition`` 解析文档，提取比
    :class:`BasicMetadataExtractor` 更丰富的结构信息：文件名、文件类型、
    元素总数、标题列表、章节标题列表以及去重升序的页码列表。

    相比 :class:`BasicMetadataExtractor` 的优势：

    - 能够识别并分类文档中的结构元素（标题 / 章节标题 / 正文等），
      而不仅限于 PDF 页码或 Word 章节。
    - 基于文件内容推断（``partition`` 自动检测格式），支持 PDF、Word、
      HTML、Markdown、纯文本等多种格式，而不受限于少量扩展名。

    注意：本类是**可选后端**，``unstructured`` 为重量级依赖（通常需系统
    级 ``libmagic`` / ``poppler`` / ``tesseract``），因此采用**懒导入**——
    仅在调用 :meth:`extract` 时才 ``import``，未安装时抛出
    :class:`MetadataExtractionError` 并给出安装提示。

    Example:
        >>> extractor = UnstructuredMetadataExtractor()
        >>> extractor.extract("docs/report.pdf")
        {'filename': 'report.pdf', 'file_type': 'pdf', 'elements_count': 42,
         'titles': ['年度报告'], 'headers': ['第一章', '1.1 背景'],
         'page_numbers': [1, 2, 3]}
    """

    def __init__(self, max_file_size: int = _DEFAULT_MAX_FILE_SIZE) -> None:
        """初始化基于 unstructured 的元数据提取器。

        Args:
            max_file_size: 允许的最大文件字节数，默认 50 MiB（防解压炸弹）。
                提取前会校验实际文件大小，超过该值将抛出
                :class:`MetadataExtractionError`。
        """
        self._max_file_size: int = max_file_size
        """允许的最大文件字节数（防解压炸弹）。"""
    def extract(self, path: str | Path) -> dict[str, Any]:
        """从指定文件提取元数据并返回键值对字典。

        提取流程：

        1. 将 ``path`` 规范化为 :class:`pathlib.Path`，校验其为存在的
           文件，否则抛出 :class:`MetadataExtractionError`。
        2. 校验文件大小不超过 ``self._max_file_size``（防解压炸弹），超限
           抛出 :class:`MetadataExtractionError`。
        3. **懒导入** ``from unstructured.partition.auto import partition``；
           若捕获 :class:`ImportError`，说明当前环境未安装 unstructured，
           抛出带有安装提示（``pip install unstructured`` 及系统依赖
           ``libmagic`` / ``poppler`` / ``tesseract``）的
           :class:`MetadataExtractionError`。
        4. 调用 ``partition(filename=str(p))`` 解析文档得到元素列表
           ``elements``；解析失败时捕获异常并转抛
           :class:`MetadataExtractionError`（``raise ... from exc``）。
        5. 从 ``elements`` 提取元数据，全程使用 :func:`getattr` 安全访问
           跨版本字段：

           - ``"filename"``：文件名（``p.name``）。
           - ``"file_type"``：去掉点号的小写后缀（``p.suffix``）。
           - ``"elements_count"``：元素总数（``len(elements)``，``int``）。
           - ``"titles"``：``list[str]``，筛选 ``category == "Title"`` 且
             文本非空的元素文本（保序）。
           - ``"headers"``：``list[str]``，筛选 ``category == "Header"``
             的元素文本（章节标题，保序）。
           - ``"page_numbers"``：``list[int]``，收集元素元数据中
             ``page_number`` 非 ``None`` 的页码，**去重并升序排序**。

        6. 返回最终元数据字典。

        Args:
            path: 待提取元数据的文件路径（字符串或 :class:`pathlib.Path`）。

        Returns:
            键值对形式的元数据字典，包含 ``filename`` / ``file_type`` /
            ``elements_count`` / ``titles`` / ``headers`` / ``page_numbers``
            等字段。

        Raises:
            MetadataExtractionError: 文件不存在、不是文件、unstructured 未
                安装或文档解析失败时抛出。
        """
        p: Path = Path(path)

        if not p.is_file():
            raise MetadataExtractionError(f"目标路径不是有效文件：{p}")

        # 提取前校验文件大小，防止解压炸弹。
        file_size: int = p.stat().st_size
        if file_size > self._max_file_size:
            raise MetadataExtractionError(
                f"文件大小超限：{p}（{file_size} 字节）"
                f"超过允许的最大值 {self._max_file_size} 字节。"
            )

        try:
            from unstructured.partition.auto import partition
        except ImportError as exc:
            raise MetadataExtractionError(
                "未安装 unstructured，请先安装：pip install unstructured"
                "（需系统依赖 libmagic / poppler / tesseract）"
            ) from exc

        try:
            elements = partition(filename=str(p))
        except MetadataExtractionError:
            raise
        except Exception as exc:
            raise MetadataExtractionError(
                f"使用 unstructured 解析文件失败：{p}，原因：{exc}"
            ) from exc

        titles: list[str] = []
        headers: list[str] = []
        page_numbers_set: set[int] = set()

        for element in elements:
            category: str = str(getattr(element, "category", "") or "")
            text: str = str(getattr(element, "text", "") or "").strip()
            if category == "Title" and text:
                titles.append(text)
            elif category == "Header" and text:
                headers.append(text)

            metadata = getattr(element, "metadata", None)
            page_number = getattr(metadata, "page_number", None)
            if page_number is not None:
                page_numbers_set.add(int(page_number))

        page_numbers: list[int] = sorted(page_numbers_set)

        return {
            "filename": p.name,
            "file_type": p.suffix.lstrip(".").lower(),
            "elements_count": len(elements),
            "titles": titles,
            "headers": headers,
            "page_numbers": page_numbers,
        }
