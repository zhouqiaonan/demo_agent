"""vector_store 包的自定义异常体系。

本模块定义了 vector_store 包中所有异常的基类和具体异常类型，
用于在向量嵌入、向量存储读写、维度校验等流程中统一错误处理策略。
"""

from __future__ import annotations

from typing import Any


class VectorStoreError(Exception):
    """Vector Store 包所有异常的基类。

    所有 vector_store 包内抛出的异常都应继承自本类，
    以便调用方可以统一捕获和处理向量存储相关的错误。
    """


class EmbeddingError(VectorStoreError):
    """向量嵌入（Embedding）阶段的错误。

    适用于以下场景：
    - 调用嵌入模型（如 sentence-transformers）时失败
    - 嵌入服务不可用或返回异常
    - 文本输入格式非法导致无法生成向量

    调用方应捕获本异常并向用户报告嵌入失败的原因，
    不应将本异常视为向量存储后端本身的错误。
    """


class DimensionMismatchError(VectorStoreError):
    """向量维度不匹配异常，表示实际向量维度与期望维度不一致。

    适用于以下场景：
    - 写入向量存储时，向量维度与集合（collection）定义不符
    - 查询向量与已存储向量的维度不一致
    - 嵌入模型输出维度与预期配置不匹配

    Attributes:
        expected_dim: 期望的向量维度。
        actual_dim: 实际的向量维度。
    """

    def __init__(self, expected_dim: int, actual_dim: int, *args: Any) -> None:
        """初始化 DimensionMismatchError。

        Args:
            expected_dim: 期望的向量维度（通常来自集合定义或配置）。
            actual_dim: 实际的向量维度（通常来自嵌入结果或待写入向量）。
            *args: 传递给父类 Exception 的额外位置参数。
        """
        self.expected_dim: int = expected_dim
        """期望的向量维度。"""

        self.actual_dim: int = actual_dim
        """实际的向量维度。"""

        message = self._build_message()
        super().__init__(message, *args)

    def _build_message(self) -> str:
        """根据期望维度与实际维度构建异常的描述字符串。

        格式示例:
            "向量维度不匹配：期望 384 维，实际 768 维"

        Returns:
            格式化后的异常描述字符串。
        """
        return (
            f"向量维度不匹配：期望 {self.expected_dim} 维，"
            f"实际 {self.actual_dim} 维"
        )
