"""向量嵌入客户端抽象基类模块。

本模块定义了向量数据库包中所有嵌入客户端的统一抽象接口
:class:`EmbeddingClient`。任何具体的嵌入实现（如 OpenAI 嵌入、
sentence-transformers 本地模型等）都应继承本类并实现核心的同步
``embed`` 方法，从而在向量存储的上层流程中做到「面向接口编程」、
可无缝替换不同的嵌入后端。

设计意图（同步为核心 + 通用异步包装）：

- ``embed`` 是**同步**核心方法，是子类必须实现的最小契约，负责把一批
  文本转换为对应的浮点向量。
- ``aembed`` 是**异步**方法，本基类提供**通用默认实现**：内部通过
  ``asyncio.to_thread`` 把同步 ``embed`` 投递到线程池执行，从而无需子类
  额外实现即可获得异步能力。对于拥有原生异步 SDK 的后端（如 OpenAI），
  子类可以按需**覆盖** ``aembed``，改用真正的异步调用以提升并发性能。
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod


class EmbeddingClient(ABC):
    """向量嵌入客户端的抽象基类。

    所有具体的嵌入后端（OpenAI、sentence-transformers 等）都应继承本类，
    并实现同步核心方法 :meth:`embed` 与抽象属性 :attr:`dimension`。
    异步方法 :meth:`aembed` 由本类提供通用默认实现，子类可选择性覆盖。

    契约约定：

    - :meth:`embed` 与 :meth:`aembed` 都接收一批文本，返回**与输入顺序严格
      一致**的向量列表，即 ``len(result) == len(texts)``，且 ``result[i]``
      对应 ``texts[i]``。
    - 每个向量都是 ``list[float]``，其长度应等于 :attr:`dimension`。
    """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """向量维度（抽象属性，子类必须实现）。

        表示该嵌入客户端产出的每个向量的长度。上层向量存储通常依赖此值
        做写入前的维度校验。

        Returns:
            每个嵌入向量的维度（正整数）。
        """
        ...

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """将一批文本转换为对应的嵌入向量（同步，抽象方法）。

        这是所有嵌入客户端的核心同步契约，子类必须实现。返回的向量列表
        顺序必须与 ``texts`` 输入顺序严格一致。

        Args:
            texts: 待嵌入的文本列表。

        Returns:
            与输入顺序一致的浮点向量列表，每个元素为 ``list[float]``。
        """
        ...

    async def aembed(self, texts: list[str]) -> list[list[float]]:
        """将一批文本转换为对应的嵌入向量（异步）。

        本方法由基类提供**通用默认实现**：借助 ``asyncio.to_thread`` 将同步
        的 :meth:`embed` 投递到线程池执行，从而在不阻塞事件循环的前提下复用
        同步实现。拥有原生异步 SDK 的子类（如 OpenAI 的 ``AsyncOpenAI``）
        可覆盖本方法以获得真正的异步并发能力。

        Args:
            texts: 待嵌入的文本列表。

        Returns:
            与输入顺序一致的浮点向量列表，每个元素为 ``list[float]``。
        """
        return await asyncio.to_thread(self.embed, texts)
