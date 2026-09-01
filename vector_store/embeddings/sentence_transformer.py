"""基于 sentence-transformers 的本地文本嵌入客户端模块。

本模块封装 HuggingFace 的 sentence-transformers 库，提供完全离线的本地
嵌入能力，无需调用外部 API。

**懒导入与懒加载设计（重要）**：

- sentence-transformers 依赖 PyTorch（torch），其安装体积巨大（约 2GB），
  且并非所有用户都需要本地嵌入能力。因此本模块**不在模块顶部导入**
  ``sentence_transformers``，而是把导入语句放在 ``_load_model`` 方法内部。
- 同样地，模型本身的下载与加载也在**首次调用** :meth:`embed` 或访问
  :attr:`dimension` 时才触发。

这样做的收益是：未安装 torch / sentence-transformers 的用户执行
``import vector_store.embeddings.sentence_transformer`` 或实例化
``SentenceTransformerEmbedding`` 都不会报错；只有真正调用 ``embed``
触发模型加载时，才会导入依赖并可能抛出 ``ImportError`` / ``OSError``。
"""

from __future__ import annotations

from typing import Any

from vector_store.embeddings.base import EmbeddingClient


class SentenceTransformerEmbedding(EmbeddingClient):
    """基于 sentence-transformers 的本地文本嵌入客户端。

    默认使用 ``all-MiniLM-L6-v2`` 模型（输出 384 维向量），并通过
    ``normalize_embeddings`` 控制是否对输出向量做 L2 归一化。

    设计说明：

    - **懒导入 + 懒加载**：``sentence_transformers`` 的导入与模型实例化都
      延迟到首次真正使用时才触发，避免未安装 torch 的用户在 import 阶段报错。
      详见模块级 docstring。
    - **空输入处理**：传入空列表 ``[]`` 时直接返回 ``[]``（不触发模型加载）；
      传入的任意字符串若为空字符串 ``""`` 则抛出 :class:`ValueError`。

    Attributes:
        model_name: 使用的模型名称（用于内部加载，也作为便捷公共属性暴露）。
        _normalize_embeddings: 是否对输出向量做 L2 归一化。
        _model: 懒加载的 sentence-transformers 模型实例；为 ``None`` 表示
            尚未加载。
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        normalize_embeddings: bool = True,
    ) -> None:
        """初始化 sentence-transformers 嵌入客户端。

        注意：本构造方法**不会**导入或加载模型（懒加载），因此无需安装
        torch 也能安全构造实例。

        Args:
            model_name: sentence-transformers 模型名称，默认
                ``"all-MiniLM-L6-v2"``。
            normalize_embeddings: 是否对输出向量做 L2 归一化，默认 ``True``。
                归一化后向量模长为 1，适合使用余弦相似度 / 内积检索。
        """
        self.model_name: str = model_name
        """使用的 sentence-transformers 模型名称。"""

        self._normalize_embeddings: bool = normalize_embeddings
        """是否对输出向量做 L2 归一化。"""

        self._model: Any | None = None
        """懒加载的模型实例；为 ``None`` 表示尚未加载。"""

    def _load_model(self) -> Any:
        """懒加载 sentence-transformers 模型实例。

        首次调用时才执行 ``import sentence_transformers`` 并实例化
        ``SentenceTransformer``，随后缓存到 ``self._model`` 复用。实例化时
        显式传入 ``trust_remote_code=False``，禁止执行模型仓库中的远程代码，
        避免加载不可信模型时潜在的安全风险。

        Returns:
            已加载的 sentence-transformers 模型实例。

        Raises:
            ImportError: 未安装 sentence-transformers 时抛出。
        """
        if self._model is None:
            import sentence_transformers

            self._model = sentence_transformers.SentenceTransformer(
                self.model_name,
                trust_remote_code=False,
            )
        return self._model

    @property
    def dimension(self) -> int:
        """返回模型的输出向量维度。

        该值由 sentence-transformers 模型自身决定（懒加载后查询），
        例如 ``all-MiniLM-L6-v2`` 返回 384。

        Returns:
            模型输出向量的维度（正整数）。
        """
        return int(self._load_model().get_sentence_embedding_dimension())

    def embed(self, texts: list[str]) -> list[list[float]]:
        """将一批文本转换为本地嵌入向量（同步）。

        懒加载模型后调用 ``SentenceTransformer.encode`` 进行编码，并把返回的
        二维 NumPy 数组转换为 ``list[list[float]]``。

        空输入处理约定：

        - ``texts`` 为空列表 ``[]`` 时，直接返回空列表 ``[]``，不触发模型加载。
        - ``texts`` 中包含空字符串 ``""`` 时，抛出 :class:`ValueError`。

        Args:
            texts: 待嵌入的文本列表。

        Returns:
            与输入顺序一致的浮点向量列表，每个元素为 ``list[float]``。

        Raises:
            ValueError: 当 ``texts`` 不为空但包含空字符串时抛出。
        """
        if not texts:
            return []
        if any(text == "" for text in texts):
            raise ValueError("待嵌入文本不能包含空字符串")

        model = self._load_model()
        vectors = model.encode(
            texts,
            normalize_embeddings=self._normalize_embeddings,
            convert_to_numpy=True,
        )
        return vectors.tolist()
