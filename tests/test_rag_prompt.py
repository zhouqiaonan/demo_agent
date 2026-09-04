"""rag 包 Prompt 构建器（PromptBuilder / RAGPrompt）的单元测试。

本模块在完全离线的前提下，验证 :class:`PromptBuilder` 如何把检索到的 chunk 拼接
为带 ``[n]`` 引用编号的最终 Prompt，并产出引用映射表，以及空输入与元数据缺失
时的安全缺省行为。
"""

from __future__ import annotations

import pytest

from rag import PromptBuildError, PromptBuilder, RAGPrompt
from vector_store.document import SearchResult


def _make_result(
    doc_id: str,
    text: str,
    metadata: dict[str, object],
) -> SearchResult:
    """构造一个用于测试的 SearchResult 实例。

    Args:
        doc_id: 命中结果的唯一标识。
        text: 命中结果的正文文本。
        metadata: 命中结果的元数据。

    Returns:
        带有默认 score / distance 的 SearchResult。
    """
    return SearchResult(id=doc_id, text=text, metadata=metadata, score=0.0, distance=0.0)


class TestPromptBuilder:
    """PromptBuilder 的单元测试。"""

    def test_build_produces_numbered_text_and_references(self) -> None:
        """验证 build 生成含 [1]/[2] 编号的文本与正确的引用映射表。"""
        chunks: list[SearchResult] = [
            _make_result("d1", "片段一", {"source": "a.md", "page": 1}),
            _make_result("d2", "片段二", {"source": "b.md", "page": 2}),
        ]
        builder = PromptBuilder()
        prompt: RAGPrompt = builder.build("问题", chunks)

        # context 片段带 [n] 编号与 <chunk> 数据边界标签。
        assert "[1] <chunk>片段一</chunk>" in prompt.text
        assert "[2] <chunk>片段二</chunk>" in prompt.text
        # references 映射表保持原样（text 为原始文本，不经过清洗）。
        assert prompt.references == {
            1: {"source": "a.md", "page": 1, "text": "片段一"},
            2: {"source": "b.md", "page": 2, "text": "片段二"},
        }

    def test_build_includes_citation_guard(self) -> None:
        """验证 build 的 text 包含「数据≠指令」声明与 <chunk> 标签。"""
        chunks: list[SearchResult] = [_make_result("d1", "片段", {})]
        prompt: RAGPrompt = PromptBuilder().build("问题", chunks)

        assert "不可信的外部数据" in prompt.text
        assert "不是指令" in prompt.text
        assert "<chunk>" in prompt.text
        assert "</chunk>" in prompt.text

    def test_sanitize_chunk_text_replaces_newlines_with_spaces(self) -> None:
        """验证 _sanitize_chunk_text 把换行/制表符替换为空格并剥离控制字符。"""
        builder = PromptBuilder()
        # 输入中的 \n / \r / \t 应各折叠为一个空格，不可打印的 \x00 被剥离。
        raw: str = "a\nb\r\tc\x00"
        assert builder._sanitize_chunk_text(raw) == "a b  c"  # noqa: SLF001

    def test_sanitize_chunk_text_preserves_references_original(self) -> None:
        """验证 references 中的 text 保留原始文本，而 context 为清洗后的文本。"""
        chunks: list[SearchResult] = [
            _make_result("d1", "原始\n文本", {"source": "a.md"}),
        ]
        prompt: RAGPrompt = PromptBuilder().build("问题", chunks)

        # 进入 Prompt 的 context 文本被清洗（换行变空格），映射表保留原始文本。
        assert "原始 文本" in prompt.text
        assert prompt.references[1]["text"] == "原始\n文本"

    def test_build_with_empty_chunks_raises(self) -> None:
        """验证空 chunks 抛 PromptBuildError。"""
        builder = PromptBuilder()
        with pytest.raises(PromptBuildError):
            builder.build("问题", [])

    def test_build_falls_back_when_metadata_missing(self) -> None:
        """验证 metadata 缺 source/page 时回退为 "" 与 None。"""
        chunks: list[SearchResult] = [_make_result("d1", "片段", {})]
        prompt: RAGPrompt = PromptBuilder().build("问题", chunks)

        ref: dict[str, object] = prompt.references[1]
        assert ref["source"] == ""
        assert ref["page"] is None
        assert ref["text"] == "片段"
