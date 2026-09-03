"""document_processor 包的共享工具与常量模块。

本模块提供跨子包（loaders / metadata）复用的工具函数与常量，避免各实现
重复内联相同逻辑：

- :data:`_DEFAULT_MAX_FILE_SIZE`：解析前文件大小上限的默认值（50 MiB），
  用于在解析文档前校验文件大小，防止解压炸弹（zip bomb）。
- :func:`is_heading_style`：判断 Word（``python-docx``）段落样式名是否为
  标题样式（``"Heading"`` 前缀，或 ``"Title"`` / ``"Subtitle"``），供
  :class:`~document_processor.loaders.word_loader.WordLoader` 与
  :class:`~document_processor.metadata.basic.BasicMetadataExtractor` 复用，
  统一标题识别逻辑。
"""

from __future__ import annotations

# 默认最大文件大小：50 MiB（解析前校验，防解压炸弹）。
_DEFAULT_MAX_FILE_SIZE: int = 50 * 1024 * 1024


def is_heading_style(style_name: str | None) -> bool:
    """判断 Word 段落样式名是否为标题样式。

    标题样式定义为：样式名以 ``"Heading"`` 开头，或恰好等于 ``"Title"`` /
    ``"Subtitle"``。用于识别 Word 文档中的标题段落，进而追踪章节归属或
    收集章节标题列表。

    Args:
        style_name: 段落的样式名；可能为 ``None``（段落未设置样式）。

    Returns:
        若样式名属于标题样式返回 ``True``，否则（包括 ``None``）返回
        ``False``。
    """
    if style_name is None:
        return False
    return style_name.startswith("Heading") or style_name in ("Title", "Subtitle")
