"""document_processor/metadata 的单元测试 —— 基础提取器与 unstructured 后端。

本模块测试两类元数据提取器的核心行为：

- :class:`BasicMetadataExtractor`：PDF 的 ``filename`` / ``file_type`` /
  ``page_count`` 字段、docx 的 ``sections`` 字段（保序去重）、非文件路径抛错。
- :class:`UnstructuredMetadataExtractor`：未安装 ``unstructured`` 时抛出含
  「未安装」提示的 :class:`MetadataExtractionError`（若当前环境已安装则跳过）。

所有测试通过 :func:`_write_blank_pdf`（``pypdf.PdfWriter`` 构造空白 PDF）与
:func:`_write_docx`（``python-docx`` 构造含标题段落的 docx）在 ``tmp_path``
临时目录内完成，完全离线、无需真实文件。
"""

from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path

import docx
import pytest
from pypdf import PdfWriter

from document_processor import (
    BasicMetadataExtractor,
    MetadataExtractionError,
    UnstructuredMetadataExtractor,
)


# ---------------------------------------------------------------------------
# 构造辅助函数
# ---------------------------------------------------------------------------


def _write_blank_pdf(path: Path, page_count: int = 1) -> None:
    """构造一个不含任何文本的多页空白 PDF 文件（供测试使用）。

    Args:
        path: 目标 PDF 文件写入路径。
        page_count: 空白页数量，默认 1。
    """
    writer: PdfWriter = PdfWriter()
    for _ in range(page_count):
        writer.add_blank_page(width=72, height=72)
    with open(path, "wb") as f:
        writer.write(f)


def _write_docx(path: Path, blocks: list[tuple[str, str]]) -> None:
    """构造 docx 文件，按 ``(样式名, 文本)`` 二元组顺序写入段落。

    Args:
        path: 目标 docx 文件写入路径。
        blocks: 段落列表，每项为 ``(样式名, 文本)`` 二元组。
    """
    document: docx.Document = docx.Document()
    for style_name, text in blocks:
        document.add_paragraph(text, style=style_name)
    document.save(str(path))


# ---------------------------------------------------------------------------
# BasicMetadataExtractor
# ---------------------------------------------------------------------------


class TestBasicMetadataExtractor:
    """测试基础元数据提取器的字段提取与错误处理。"""

    def test_pdf_metadata_fields(self, tmp_path: Path) -> None:
        """PDF 应提取 filename / file_type / size 等基础字段与 page_count。"""
        path: Path = tmp_path / "report.pdf"
        _write_blank_pdf(path, page_count=3)

        metadata = BasicMetadataExtractor().extract(str(path))

        assert metadata["filename"] == "report.pdf"
        assert metadata["file_type"] == "pdf"
        assert metadata["page_count"] == 3
        assert metadata["size"] > 0
        assert "modified_time" in metadata

    def test_docx_sections_dedup_and_ordered(self, tmp_path: Path) -> None:
        """docx 的 sections 应保序去重地收集标题段落文本。"""
        path: Path = tmp_path / "notes.docx"
        _write_docx(
            path,
            [
                ("Heading 1", "Chapter One"),
                ("Normal", "Intro body."),
                ("Heading 2", "Section 1.1"),
                ("Normal", "Detail body."),
                ("Heading 1", "Chapter One"),
            ],
        )

        metadata = BasicMetadataExtractor().extract(str(path))

        assert metadata["filename"] == "notes.docx"
        assert metadata["file_type"] == "docx"
        # 去重后仅保留首次出现，且保持出现顺序。
        assert metadata["sections"] == ["Chapter One", "Section 1.1"]

    def test_non_file_path_raises(self, tmp_path: Path) -> None:
        """不存在的文件路径应抛出 MetadataExtractionError。"""
        with pytest.raises(MetadataExtractionError):
            BasicMetadataExtractor().extract(str(tmp_path / "missing.pdf"))

    def test_file_too_large_raises(self, tmp_path: Path) -> None:
        """文件大小超过 max_file_size 时应抛出 MetadataExtractionError。"""
        path: Path = tmp_path / "large.pdf"
        _write_blank_pdf(path)

        with pytest.raises(MetadataExtractionError, match="大小超限"):
            BasicMetadataExtractor(max_file_size=1).extract(str(path))


# ---------------------------------------------------------------------------
# UnstructuredMetadataExtractor
# ---------------------------------------------------------------------------


class TestUnstructuredMetadataExtractor:
    """测试 unstructured 后端的未安装分支。"""

    def test_unstructured_not_installed_raises(self, tmp_path: Path) -> None:
        """未安装 unstructured 时应抛出含「未安装」提示的 MetadataExtractionError。"""
        if find_spec("unstructured") is not None:
            pytest.skip("当前环境已安装 unstructured，跳过「未安装」分支测试。")

        path: Path = tmp_path / "report.pdf"
        _write_blank_pdf(path)

        with pytest.raises(MetadataExtractionError, match="未安装"):
            UnstructuredMetadataExtractor().extract(str(path))

    def test_file_too_large_raises(self, tmp_path: Path) -> None:
        """文件大小超过 max_file_size 时应抛出 MetadataExtractionError。

        大小校验先于 unstructured 懒导入，因此无论当前环境是否安装
        unstructured，都会抛出带「大小超限」提示的错误。
        """
        path: Path = tmp_path / "large.pdf"
        _write_blank_pdf(path)

        with pytest.raises(MetadataExtractionError, match="大小超限"):
            UnstructuredMetadataExtractor(max_file_size=1).extract(str(path))
