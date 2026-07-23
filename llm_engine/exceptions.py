"""llm_engine 包的自定义异常体系。

本模块定义了 llm_engine 包中所有异常的基类和具体异常类型，
用于在 LLM 调用重试、路由切换等流程中统一错误处理策略。
"""

from __future__ import annotations

from typing import Any


class LLMEngineError(Exception):
    """LLM Engine 包所有异常的基类。

    所有 llm_engine 包内抛出的异常都应继承自本类，
    以便调用方可以统一捕获和处理。
    """


class TransientError(LLMEngineError):
    """瞬态错误，表示该请求应该被重试。

    适用于以下场景：
    - 网络超时（timeout）
    - HTTP 429 速率限制（rate limit）
    - HTTP 5xx 服务端错误
    - 临时的服务不可用

    调用方应捕获本异常并实现退避重试策略，
    不应将本异常视为最终失败。
    """


class NonRetryableError(LLMEngineError):
    """不可重试错误，表示请求应直接失败而不重试。

    适用于以下场景：
    - HTTP 401 认证失败（无效的 API Key）
    - HTTP 400 请求参数错误
    - HTTP 402 账户余额不足
    - HTTP 403 权限不足或模型未授权

    调用方应捕获本异常并向用户报告明确的失败原因，
    不应尝试重试。
    """


class AllModelsExhaustedError(LLMEngineError):
    """所有模型（含备用模型）均已尝试且全部失败的异常。

    当 LLM Engine 遍历了所有可用模型（包括主模型和备用模型），
    且每个模型都返回了错误时，抛出本异常。

    Attributes:
        failed_models: 一个列表，每个元素是一个字典，
            包含 ``model``（模型名称）和 ``error``（错误信息字符串）两个键。
    """

    def __init__(self, failed_models: list[dict[str, str]], *args: Any) -> None:
        """初始化 AllModelsExhaustedError。

        Args:
            failed_models: 每个模型失败信息的列表。每个元素为包含 ``model`` 和
                ``error`` 键的字典，分别记录模型名称和错误描述。
            *args: 传递给父类 Exception 的额外位置参数。
        """
        self.failed_models: list[dict[str, str]] = failed_models
        message = self._build_message()
        super().__init__(message, *args)

    def _build_message(self) -> str:
        """根据 failed_models 构建异常的描述字符串。

        格式示例:
            "所有模型均已尝试且全部失败: gpt-4o (timeout), deepseek-chat (rate limit)"

        Returns:
            格式化后的异常描述字符串。
        """
        if not self.failed_models:
            return "所有模型均已尝试且全部失败（无失败详情）"

        details = ", ".join(
            f"{item['model']} ({item['error']})" for item in self.failed_models
        )
        return f"所有模型均已尝试且全部失败: {details}"
