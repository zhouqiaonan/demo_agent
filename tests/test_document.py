"""Tests for vector_store/document.py — Document 与 SearchResult 数据模型。"""

from __future__ import annotations

from vector_store.document import Document, SearchResult


class TestDocument:
    """测试 Document 数据模型。"""

    def test_auto_generates_32_hex_id(self) -> None:
        """未显式提供 id 时应自动生成 32 位十六进制字符串。"""
        doc = Document(text="x")
        assert len(doc.id) == 32
        # 若不是合法十六进制字符串，int(doc.id, 16) 会抛出 ValueError。
        int(doc.id, 16)

    def test_two_documents_have_distinct_ids(self) -> None:
        """两次构造的 Document 应具有不同的自动生成 id。"""
        doc1 = Document(text="x")
        doc2 = Document(text="x")
        assert doc1.id != doc2.id

    def test_explicit_id_preserved(self) -> None:
        """显式传入 id 时应原样保留。"""
        doc = Document(text="x", id="abc")
        assert doc.id == "abc"

    def test_metadata_defaults_to_empty_dict(self) -> None:
        """未提供 metadata 时默认为空字典。"""
        doc = Document(text="x")
        assert doc.metadata == {}

    def test_metadata_preserved(self) -> None:
        """显式传入 metadata 时应原样保留。"""
        doc = Document(text="x", metadata={"source": "wiki"})
        assert doc.metadata == {"source": "wiki"}


class TestSearchResult:
    """测试 SearchResult 数据模型。"""

    def test_fields(self) -> None:
        """各字段应正确赋值。"""
        result = SearchResult(
            id="doc-1",
            text="你好",
            metadata={"a": 1},
            score=0.95,
            distance=0.05,
        )
        assert result.id == "doc-1"
        assert result.text == "你好"
        assert result.metadata == {"a": 1}
        assert result.score == 0.95
        assert result.distance == 0.05
