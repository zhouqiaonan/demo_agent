"""Tests for vector_store/embeddings — 嵌入客户端抽象与两种实现。

所有测试离线运行：OpenAI 客户端仅测试属性与构造（不发起网络请求）；
sentence-transformers 通过注入 fake module 验证懒加载与委托。
"""

from __future__ import annotations

import asyncio
import os
import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from tests.fakes import HashEmbedding
from vector_store.embeddings.base import EmbeddingClient
from vector_store.embeddings.openai_embedding import OpenAIEmbedding
from vector_store.embeddings.sentence_transformer import SentenceTransformerEmbedding


class TestEmbeddingClientABC:
    """测试 EmbeddingClient 抽象基类契约。"""

    def test_abstract_cannot_instantiate(self) -> None:
        """抽象基类不可直接实例化，应抛出 TypeError。"""
        with pytest.raises(TypeError):
            EmbeddingClient()  # type: ignore[abstract]

    def test_aembed_matches_embed(self) -> None:
        """基类默认 aembed 应复用 embed（经 asyncio.to_thread）。"""
        emb = HashEmbedding(dimension=8)
        texts: list[str] = ["你好", "world", "foo"]
        expected: list[list[float]] = emb.embed(texts)
        actual: list[list[float]] = asyncio.run(emb.aembed(texts))
        assert actual == expected


class TestOpenAIEmbedding:
    """测试 OpenAIEmbedding 的属性与构造（离线，不发起请求）。"""

    def test_dimension_default(self) -> None:
        """未指定 dimensions 时维度默认 1536。"""
        assert OpenAIEmbedding().dimension == 1536

    def test_dimension_explicit(self) -> None:
        """显式指定 dimensions 时返回该值。"""
        assert OpenAIEmbedding(dimensions=256).dimension == 256

    def test_api_key_explicit_does_not_read_env(self) -> None:
        """显式传入 api_key 时不应读取环境变量。"""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "env-secret"}):
            emb = OpenAIEmbedding(api_key="explicit-secret")
            assert emb._api_key == "explicit-secret"

    def test_api_key_falls_back_to_env(self) -> None:
        """未显式传入 api_key 时应回退到环境变量。"""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "env-secret"}):
            emb = OpenAIEmbedding()
            assert emb._api_key == "env-secret"


class _FakeArray:
    """模拟 NumPy 数组，仅提供 ``tolist()`` 方法供 embed 使用。"""

    def __init__(self, data: list[list[float]]) -> None:
        """初始化伪造数组。

        Args:
            data: 内部保存的二维浮点列表。
        """
        self._data: list[list[float]] = data
        """内部数据。"""

    def tolist(self) -> list[list[float]]:
        """返回内部保存的二维浮点列表。"""
        return self._data


class TestSentenceTransformerEmbedding:
    """测试 SentenceTransformerEmbedding 的懒加载与委托。"""

    def test_not_loaded_on_construction(self) -> None:
        """构造后不应触发 sentence_transformers 导入或模型加载。"""
        emb = SentenceTransformerEmbedding()
        assert emb.model_name == "all-MiniLM-L6-v2"
        assert emb._model is None

    def test_lazy_load_dimension_and_embed(self) -> None:
        """首次使用时应懒加载模型，dimension/embed 委托给模型。"""
        emb = SentenceTransformerEmbedding(model_name="fake-model")

        fake_model = MagicMock()
        fake_model.get_sentence_embedding_dimension.return_value = 384

        sentence_transformer_cls = MagicMock(return_value=fake_model)
        fake_module = ModuleType("sentence_transformers")
        setattr(fake_module, "SentenceTransformer", sentence_transformer_cls)

        with patch.dict(sys.modules, {"sentence_transformers": fake_module}):
            # 访问 dimension 触发懒加载，并委托给模型的维度查询。
            assert emb.dimension == 384
            sentence_transformer_cls.assert_called_once_with(
                "fake-model", trust_remote_code=False
            )

            # embed 委托给模型 encode，并把返回的数组转为 list。
            fake_model.encode.return_value = _FakeArray([[0.1, 0.2], [0.3, 0.4]])
            result = emb.embed(["你好", "世界"])
            assert result == [[0.1, 0.2], [0.3, 0.4]]
            fake_model.encode.assert_called_once_with(
                ["你好", "世界"],
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
