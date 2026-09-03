"""document_processor/loaders 的单元测试 —— PdfLoader 与 WordLoader。

本模块测试 PDF / Word 文档加载器的核心行为：

- :class:`PdfLoader`：文件不存在 / 目录路径抛错、空白页跳过、含文本 PDF
  正确抽取正文与 ``source`` / ``page``（0-based）元数据。
- :class:`WordLoader`：段落按序加载并带 ``paragraph_index``、标题样式正确
  填充后续段落的 ``section``、文件不存在抛错。

所有测试均通过 :func:`_write_text_pdf`（手工构造含文本的最小 PDF）、
:func:`_write_blank_pdf`（``pypdf.PdfWriter`` 构造空白 PDF）与
:func:`_write_docx`（``python-docx`` 构造含标题段落的 docx）在
``tmp_path`` 临时目录内完成，完全离线、无需真实文件。
"""

from __future__ import annotations

from pathlib import Path

import docx
import pytest
from pypdf import PdfWriter

from document_processor import DocumentLoadError, PdfLoader, WordLoader


# ---------------------------------------------------------------------------
# 构造辅助函数
# ---------------------------------------------------------------------------


def _write_text_pdf(path: Path, text: str) -> None:
    """手工构造一个含单段文本的最小 PDF 文件（供测试使用）。

    通过手写 PDF 对象、xref 表与 trailer 生成合法的单页 PDF，页内容为一段
    使用 Helvetica 字体渲染的英文文本。注意：正文经 ``latin-1`` 编码，中文
    会被替换为 ``??``，因此测试文本请使用英文。

    Args:
        path: 目标 PDF 文件写入路径。
        text: 要写入 PDF 的英文文本内容。
    """
    escaped: str = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    content: bytes = ("BT /F1 12 Tf 72 720 Td (" + escaped + ") Tj ET").encode(
        "latin-1", "replace"
    )
    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length "
        + str(len(content)).encode()
        + b" >>\nstream\n"
        + content
        + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    pdf: bytearray = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf += (str(i) + " 0 obj\n").encode() + obj + b"\nendobj\n"
    xref: int = len(pdf)
    pdf += ("xref\n0 " + str(len(objects) + 1) + "\n").encode()
    pdf += b"0000000000 65535 f \n"
    for off in offsets:
        pdf += ("{:010d} 00000 n \n".format(off)).encode()
    pdf += (
        "trailer\n<< /Size "
        + str(len(objects) + 1)
        + " /Root 1 0 R >>\nstartxref\n"
        + str(xref)
        + "\n%%EOF\n"
    ).encode()
    path.write_bytes(bytes(pdf))


def _write_blank_pdf(path: Path) -> None:
    """构造一个不含任何文本的单页空白 PDF 文件（供测试使用）。

    Args:
        path: 目标 PDF 文件写入路径。
    """
    writer: PdfWriter = PdfWriter()
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
# PdfLoader
# ---------------------------------------------------------------------------


class TestPdfLoader:
    """测试 PdfLoader 的文件校验、空白页跳过与文本抽取。"""

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        """文件不存在时应抛出 DocumentLoadError。"""
        with pytest.raises(DocumentLoadError):
            PdfLoader(str(tmp_path / "missing.pdf")).load()

    def test_directory_path_raises(self, tmp_path: Path) -> None:
        """目录路径（不是文件）时应抛出 DocumentLoadError。"""
        with pytest.raises(DocumentLoadError):
            PdfLoader(str(tmp_path)).load()

    def test_blank_pdf_returns_empty(self, tmp_path: Path) -> None:
        """空白 PDF（无文本页）应返回空列表（跳过空白页）。"""
        path: Path = tmp_path / "blank.pdf"
        _write_blank_pdf(path)
        assert PdfLoader(str(path)).load() == []

    def test_text_pdf_extracts_text_and_metadata(self, tmp_path: Path) -> None:
        """含文本 PDF 应正确抽取正文与 source / page（0-based）元数据。"""
        path: Path = tmp_path / "sample.pdf"
        _write_text_pdf(path, "Hello world from a PDF document.")

        docs = PdfLoader(str(path)).load()

        assert len(docs) == 1
        assert docs[0].text == "Hello world from a PDF document."
        assert docs[0].metadata["source"] == str(path)
        assert docs[0].metadata["page"] == 0

    def test_file_too_large_raises(self, tmp_path: Path) -> None:
        """文件大小超过 max_file_size 时应抛出 DocumentLoadError。"""
        path: Path = tmp_path / "large.pdf"
        _write_text_pdf(path, "Hello world from a PDF document.")

        with pytest.raises(DocumentLoadError, match="大小超限"):
            PdfLoader(str(path), max_file_size=1).load()


# ---------------------------------------------------------------------------
# WordLoader
# ---------------------------------------------------------------------------


class TestWordLoader:
    """测试 WordLoader 的段落加载、章节追踪与文件校验。"""

    def test_load_paragraphs_with_index_and_section(self, tmp_path: Path) -> None:
        """段落应按序加载并带 paragraph_index；标题样式正确填充 section。"""
        path: Path = tmp_path / "sample.docx"
        _write_docx(
            path,
            [
                ("Heading 1", "Chapter One"),
                ("Normal", "First body paragraph."),
                ("Heading 2", "Section 1.1"),
                ("Normal", "Second body paragraph."),
            ],
        )

        docs = WordLoader(str(path)).load()

        assert len(docs) == 4
        # 所有段落都应携带 source 元数据。
        assert all(d.metadata["source"] == str(path) for d in docs)

        # 标题段落本身作为内容产出，其 section 为自身标题。
        assert docs[0].text == "Chapter One"
        assert docs[0].metadata["paragraph_index"] == 0
        assert docs[0].metadata["section"] == "Chapter One"

        # 正文段落继承最近一个标题的 section，paragraph_index 为 0-based。
        assert docs[1].text == "First body paragraph."
        assert docs[1].metadata["paragraph_index"] == 1
        assert docs[1].metadata["section"] == "Chapter One"

        # 遇到新的标题后，后续段落切换 section。
        assert docs[2].text == "Section 1.1"
        assert docs[2].metadata["paragraph_index"] == 2
        assert docs[2].metadata["section"] == "Section 1.1"

        assert docs[3].text == "Second body paragraph."
        assert docs[3].metadata["paragraph_index"] == 3
        assert docs[3].metadata["section"] == "Section 1.1"

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        """文件不存在时应抛出 DocumentLoadError。"""
        with pytest.raises(DocumentLoadError):
            WordLoader(str(tmp_path / "missing.docx")).load()

    def test_file_too_large_raises(self, tmp_path: Path) -> None:
        """文件大小超过 max_file_size 时应抛出 DocumentLoadError。"""
        path: Path = tmp_path / "large.docx"
        _write_docx(path, [("Normal", "Some text.")])

        with pytest.raises(DocumentLoadError, match="大小超限"):
            WordLoader(str(path), max_file_size=1).load()

    def test_empty_heading_does_not_pollute_section(self, tmp_path: Path) -> None:
        """空白标题段落不应把 section 污染为空字符串。"""
        path: Path = tmp_path / "empty_heading.docx"
        _write_docx(
            path,
            [
                ("Heading 1", ""),
                ("Normal", "Body after empty heading."),
            ],
        )

        docs = WordLoader(str(path)).load()

        # 空白标题段落被跳过；正文段落不携带 section。
        assert len(docs) == 1
        assert docs[0].text == "Body after empty heading."
        assert "section" not in docs[0].metadata
