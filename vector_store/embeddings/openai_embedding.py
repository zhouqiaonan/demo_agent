"""OpenAI 文本嵌入客户端模块。

本模块基于 OpenAI 官方 Python SDK 实现 :class:`EmbeddingClient`，支持
``text-embedding-3-small`` / ``text-embedding-3-large`` 等嵌入模型，并同时
提供同步与原生异步两种调用方式。

设计要点：

- **懒导入 openai**：``openai`` 的导入延迟到 :attr:`_client` /
  :attr:`_async_client` 首次被访问时才执行，因此未安装 ``openai`` 时也能
  正常 ``import vector_store``（与 chromadb、sentence-transformers 的懒导入
  策略一致）。
- **懒创建客户端**：同步客户端 :class:`openai.OpenAI` 与异步客户端
  :class:`openai.AsyncOpenAI` 都在首次使用时才实例化。这样即使构造
  ``OpenAIEmbedding`` 时尚未配置 API Key，也不会在 ``__init__`` 阶段报错，
  而是在真正发起嵌入请求时才触发客户端创建与鉴权。
- **同步 + 原生异步**：``embed`` 走同步 SDK，``aembed`` 覆盖基类的通用
  ``asyncio.to_thread`` 实现、改走 ``AsyncOpenAI``，从而在批量场景下获得
  真正的异步并发能力。
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from vector_store.embeddings.base import EmbeddingClient

if TYPE_CHECKING:
    from openai import AsyncOpenAI, OpenAI


# 已知 OpenAI 嵌入模型对应的输出维度映射表。
_MODEL_DIMENSIONS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}
"""已知 OpenAI 嵌入模型名到输出维度的映射。"""


class OpenAIEmbedding(EmbeddingClient):
    """基于 OpenAI API 的文本嵌入客户端。

    默认使用 ``text-embedding-3-small`` 模型。可通过 ``dimensions`` 参数
    显式指定输出向量的维度（OpenAI 的 ``text-embedding-3`` 系列支持缩短
    维度）。

    设计说明：

    - **懒创建客户端**：同步客户端 ``OpenAI`` 与异步客户端 ``AsyncOpenAI``
      都在首次调用 ``embed`` / ``aembed`` 时才创建，避免未配置 API Key 时
      构造阶段就抛出异常。
    - **空输入处理**：传入空列表 ``[]`` 时直接返回 ``[]``（不发起请求）；
      传入的任意字符串若为空字符串 ``""`` 则抛出 :class:`ValueError`，因为
      OpenAI 嵌入接口不接受空字符串。

    Attributes:
        model: 使用的嵌入模型名称。
        _api_key: 解析后的 API Key（构造时若 ``api_key`` 为空则回退到
            环境变量 ``OPENAI_API_KEY``）。
        _dimensions: 显式指定的输出维度；为 ``None`` 时使用模型默认维度。
    """

    # 回退维度：未知模型的默认维度，同时也是 text-embedding-3-small 的默认维度。
    _DEFAULT_DIMENSION: int = 1536
    """未知模型的回退维度，同时也是 ``text-embedding-3-small`` 的默认维度。"""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "text-embedding-3-small",
        dimensions: int | None = None,
    ) -> None:
        """初始化 OpenAI 嵌入客户端。

        Args:
            api_key: OpenAI API Key；为 ``None`` 或空字符串时回退到环境变量
                ``OPENAI_API_KEY``。若二者都为空，客户端会在首次真正请求时
                抛出鉴权错误（构造阶段不报错）。
            model: 使用的嵌入模型名称，默认 ``"text-embedding-3-small"``。
            dimensions: 显式指定的输出向量维度；为 ``None`` 时使用模型默认
                维度（``text-embedding-3-small`` 为 1536）。
        """
        self.model: str = model
        """使用的嵌入模型名称。"""

        self._api_key: str = api_key or os.environ.get("OPENAI_API_KEY", "")
        """解析后的 API Key（空字符串表示未配置）。"""

        self._dimensions: int | None = dimensions
        """显式指定的输出向量维度；为 ``None`` 时使用模型默认维度。"""

        self._sync_client: OpenAI | None = None
        """懒创建的同步 OpenAI 客户端，首次调用 ``embed`` 时实例化。"""

        self._async_client_impl: AsyncOpenAI | None = None
        """懒创建的异步 OpenAI 客户端，首次调用 ``aembed`` 时实例化。"""

    @property
    def dimension(self) -> int:
        """返回输出向量的维度。

        若构造时通过 ``dimensions`` 参数显式指定了维度，则返回该值；否则
        依据 ``self.model`` 在 :data:`_MODEL_DIMENSIONS` 中查找已知模型
        的维度；未知模型回退到 1536（即 ``text-embedding-3-small`` 的默认维度）。

        Returns:
            输出向量的维度。
        """
        if self._dimensions is not None:
            return self._dimensions
        return _MODEL_DIMENSIONS.get(self.model, self._DEFAULT_DIMENSION)

    @property
    def _client(self) -> OpenAI:
        """同步 OpenAI 客户端（懒创建）。

        首次访问时用 ``self._api_key`` 实例化 :class:`openai.OpenAI` 并缓存，
        后续访问复用同一实例。

        Returns:
            同步 OpenAI 客户端实例。
        """
        if self._sync_client is None:
            from openai import OpenAI

            self._sync_client = OpenAI(api_key=self._api_key)
        return self._sync_client

    @property
    def _async_client(self) -> AsyncOpenAI:
        """异步 OpenAI 客户端（懒创建）。

        首次访问时用 ``self._api_key`` 实例化 :class:`openai.AsyncOpenAI`
        并缓存，后续访问复用同一实例。

        Returns:
            异步 OpenAI 客户端实例。
        """
        if self._async_client_impl is None:
            from openai import AsyncOpenAI

            self._async_client_impl = AsyncOpenAI(api_key=self._api_key)
        return self._async_client_impl

    def embed(self, texts: list[str]) -> list[list[float]]:
        """将一批文本转换为 OpenAI 嵌入向量（同步）。

        空输入处理约定：

        - ``texts`` 为空列表 ``[]`` 时，直接返回空列表 ``[]``，不发起网络请求。
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

        response = self._client.embeddings.create(
            input=texts,
            model=self.model,
            **({"dimensions": self._dimensions} if self._dimensions else {}),
        )
        return [item.embedding for item in response.data]

    async def aembed(self, texts: list[str]) -> list[list[float]]:
        """将一批文本转换为 OpenAI 嵌入向量（异步，原生异步 SDK）。

        本方法**覆盖**基类的通用 ``asyncio.to_thread`` 实现，改用
        :class:`openai.AsyncOpenAI` 进行真正的异步调用，从而在批量嵌入场景下
        获得更高的并发性能。

        空输入处理约定与 :meth:`embed` 一致。

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

        response = await self._async_client.embeddings.create(
            input=texts,
            model=self.model,
            **({"dimensions": self._dimensions} if self._dimensions else {}),
        )
        return [item.embedding for item in response.data]
