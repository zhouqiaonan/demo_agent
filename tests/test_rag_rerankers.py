"""rag 包重排序器（ReRanker / CrossEncoderReRanker）的单元测试。

本模块在完全离线、不安装 sentence-transformers 的前提下，验证：

- :class:`ReRanker` 抽象基类不可直接实例化。
- :class:`CrossEncoderReRanker` 的空输入短路行为（不触发懒加载）。
- 懒加载失败时正确转抛 :class:`RerankError`。
"""

from __future__ import annotations

import sys
import types
from unittest.mock import patch

import pytest

from rag import CrossEncoderReRanker, ReRanker, RerankError
from vector_store.document import SearchResult


def _make_result(doc_id: str, text: str) -> SearchResult:
    """构造一个用于测试的 SearchResult 实例。

    Args:
        doc_id: 命中结果的唯一标识。
        text: 命中结果的正文文本。

    Returns:
        带有默认 score / distance 的 SearchResult。
    """
    return SearchResult(id=doc_id, text=text, metadata={}, score=0.0, distance=0.0)


class TestReRanker:
    """ReRanker 抽象基类的单元测试。"""

    def test_cannot_instantiate_abstract_base(self) -> None:
        """验证 ReRanker 抽象基类不可直接实例化。"""
        with pytest.raises(TypeError):
            ReRanker()


class TestCrossEncoderReRanker:
    """CrossEncoderReRanker 的单元测试。"""

    def test_empty_results_returns_empty_without_lazy_load(self) -> None:
        """验证空结果直接返回 [] 且不触发模型懒加载。"""
        reranker = CrossEncoderReRanker()
        result: list[SearchResult] = reranker.rerank("问题", [])
        assert result == []
        # 懒加载仅在非空输入时触发，因此模型实例应保持为 None。
        assert reranker._model is None  # noqa: SLF001 - 测试需访问内部状态

    def test_lazy_load_failure_raises_rerank_error(self) -> None:
        """验证懒加载模型失败时转抛 RerankError，且 message 含「模型」。"""
        # 构造一个 CrossEncoder 构造即抛异常的假模块并注入 sys.modules，使
        # rerank 内部的 `from sentence_transformers import CrossEncoder` 命中它，
        # 从而在不安装重依赖、不联网的前提下模拟模型加载失败。
        class _FailingCrossEncoder:
            def __init__(self, model_name: str) -> None:
                raise RuntimeError("模型权重下载失败")

        fake_module: types.ModuleType = types.ModuleType("sentence_transformers")
        setattr(fake_module, "CrossEncoder", _FailingCrossEncoder)

        reranker = CrossEncoderReRanker(model_name="fake/model")
        with patch.dict(sys.modules, {"sentence_transformers": fake_module}):
            with pytest.raises(RerankError) as exc_info:
                reranker.rerank("问题", [_make_result("d1", "文本")])

        assert "模型" in str(exc_info.value)

    def test_predict_failure_raises_rerank_error(self) -> None:
        """验证模型推理（predict）异常被转抛为 RerankError。

        该测试覆盖 CrossEncoder 成功路径（模型已加载后进入推理环节），通过向
        ``reranker._model`` 注入一个 ``predict`` 抛异常的假模型，验证推理异常
        不会裸抛、也不会被上层误包装为 RAGChainError。
        """

        class _FakeModel:
            """predict 抛异常的假模型，用于模拟推理阶段失败。"""

            def predict(self, pairs: list[list[str]]) -> list[float]:
                """模拟推理失败：直接抛异常。

                Args:
                    pairs: (query, document) 对列表（忽略）。

                Returns:
                    永不返回（始终抛异常）。
                """
                raise RuntimeError("推理失败")

        reranker = CrossEncoderReRanker(model_name="fake/model")
        reranker._model = _FakeModel()  # noqa: SLF001 - 测试需注入假模型

        with pytest.raises(RerankError) as exc_info:
            reranker.rerank("问题", [_make_result("d1", "文本")])

        assert "推理" in str(exc_info.value)
