"""LLM Engine 降级（Fallback）管理器模块。

本模块提供 ``FallbackManager`` 类，按顺序管理一组 ``BaseLLMClient`` 实例。
当主模型调用失败时，自动切换到下一个备用模型，每个模型内部使用
``RetryConfig`` 配置的 tenacity 重试策略。

核心流程::

    clients = [OpenAIClient(...), DeepSeekClient(...)]
    fm = FallbackManager(clients, RetryConfig(max_attempts=3))
    result = fm.execute([{"role": "user", "content": "Hello"}])

如果第一个客户端在重试耗尽后仍失败，会自动尝试第二个客户端。
当所有客户端都失败时，抛出 ``AllModelsExhaustedError``。
"""

from __future__ import annotations

import re
from typing import Any, Callable

from tenacity import Retrying, retry_if_exception_type

from llm_client.base import BaseLLMClient
from llm_engine.exceptions import AllModelsExhaustedError, NonRetryableError
from llm_engine.retry import RetryConfig

# ------------------------------------------------------------------
# 敏感信息过滤
# ------------------------------------------------------------------

_SENSITIVE_PATTERNS: list[tuple[str, str]] = [
    (r"sk-[A-Za-z0-9]{32,}", "[REDACTED_API_KEY]"),
    (r"Bearer\s+\S+", "Bearer [REDACTED]"),
]
"""敏感信息正则模式列表，用于清理异常消息中的 API 密钥等敏感数据。"""


def _sanitize_error_message(error: Exception) -> str:
    """从异常消息中移除潜在敏感信息。

    对 API 密钥（如 ``sk-...`` 格式）及 Bearer token 进行替换脱敏，
    确保记录到 ``_failures`` 中的错误描述不包含可用于攻击的凭据。

    Args:
        error: 原始异常对象。

    Returns:
        脱敏后的错误描述字符串，格式为 ``"ExceptionType: sanitized_detail"``。
    """
    msg: str = type(error).__name__
    try:
        detail: str = str(error)
    except Exception:
        return msg
    for pattern, replacement in _SENSITIVE_PATTERNS:
        detail = re.sub(pattern, replacement, detail, flags=re.IGNORECASE)
    return f"{msg}: {detail}" if detail else msg


class FallbackManager:
    """LLM 调用降级管理器，按客户端顺序进行容错调用。

    本类直接管理一个 ``BaseLLMClient`` 列表，不依赖 ModelRouter。
    每个客户端内部使用 tenacity ``Retrying`` 进行配置化重试，
    客户端之间按列表顺序依次尝试。

    Attributes:
        _clients: 内部持有的客户端列表（不可变副本）。
        _retry_config: 每个客户端内部使用的重试配置。
        _failures: 最近一次 ``execute`` 调用中累积的失败记录列表。
            每个元素为 ``{"model": str, "error": str}`` 格式。

    Example:
        >>> from unittest.mock import MagicMock
        >>> client = MagicMock()
        >>> client.model_name = "test-model"
        >>> client.chat_completion.return_value = {"role": "assistant", "content": "hi"}
        >>> fm = FallbackManager([client], RetryConfig(max_attempts=1))
        >>> result = fm.execute([{"role": "user", "content": "hello"}])
        >>> result["content"]
        'hi'
    """

    def __init__(
        self,
        clients: list[BaseLLMClient],
        retry_config: RetryConfig | None = None,
    ) -> None:
        """初始化 FallbackManager。

        Args:
            clients: ``BaseLLMClient`` 实例列表，按顺序作为主模型和备用模型。
                调用 ``execute`` 时将按此顺序依次尝试。
            retry_config: 每个客户端内部使用的重试配置。若为 ``None``，
                则使用默认 ``RetryConfig()``（最多 3 次尝试，指数退避，
                仅重试 ``TransientError``）。

        Raises:
            ValueError: 如果 ``clients`` 列表为空。
        """
        if not clients:
            raise ValueError("clients 列表不能为空，至少需要提供一个 LLM 客户端")

        self._clients: list[BaseLLMClient] = list(clients)
        """内部客户端列表（存储副本以保证不可变性）。"""

        self._retry_config: RetryConfig = (
            retry_config if retry_config is not None else RetryConfig()
        )
        """每个客户端内部使用的重试配置。"""

        self._failures: list[dict[str, str]] = []
        """最近一次调用中累积的失败记录。"""

    # ------------------------------------------------------------------
    # 只读属性
    # ------------------------------------------------------------------

    @property
    def clients(self) -> tuple[BaseLLMClient, ...]:
        """返回客户端列表的不可变副本。

        Returns:
            包含所有已注册 ``BaseLLMClient`` 实例的元组。
        """
        return tuple(self._clients)

    @property
    def retry_config(self) -> RetryConfig:
        """返回当前使用的重试配置。

        Returns:
            ``RetryConfig`` 实例。
        """
        return self._retry_config

    @property
    def failure_history(self) -> list[dict[str, str]]:
        """返回最近一次 ``execute`` 调用中记录的失败信息。

        列表中每个元素为 ``{"model": str, "error": str}`` 格式的字典，
        分别记录模型名称和错误描述。若最近一次调用全部成功，则返回空列表。

        返回的是内部 ``_failures`` 的防御性副本，外部修改不影响内部状态。

        Returns:
            失败记录列表的浅拷贝。
        """
        return list(self._failures)

    # ------------------------------------------------------------------
    # 核心执行方法
    # ------------------------------------------------------------------

    def execute(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        """按客户端顺序执行 LLM 调用，失败时自动降级到下一个客户端。

        本方法委托给 ``execute_with_callbacks``，将所有回调设为 ``None``，
        以消除与带回调版本的代码重复。

        Args:
            messages: 符合 OpenAI Chat Completions 格式的消息列表，
                每个元素为 ``{"role": str, "content": str}`` 格式。
            **kwargs: 透传给 ``BaseLLMClient.chat_completion`` 的额外参数，
                如 ``temperature``、``max_tokens`` 等。

        Returns:
            LLM 响应字典，格式由具体 ``BaseLLMClient`` 实现决定，
            通常包含 ``"content"`` 和 ``"role"`` 键。

        Raises:
            NonRetryableError: 任意客户端返回不可重试错误时立即抛出。
            AllModelsExhaustedError: 所有客户端均在重试耗尽后失败。
        """
        return self.execute_with_callbacks(
            messages=messages,
            on_success=None,
            on_failure=None,
            on_switch=None,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # 带回调的执行方法
    # ------------------------------------------------------------------

    def execute_with_callbacks(
        self,
        messages: list[dict[str, Any]],
        on_success: Callable[
            [BaseLLMClient, dict[str, Any], int], None
        ] | None = None,
        on_failure: Callable[
            [BaseLLMClient, Exception, int], None
        ] | None = None,
        on_switch: Callable[
            [BaseLLMClient, BaseLLMClient, str], None
        ] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """带回调钩子的执行方法，供上层 LLMEngine 注入 metrics/cost 逻辑。

        本方法与 ``execute`` 的核心逻辑相同，但额外提供三个回调钩子：

        - ``on_success(client, response, attempt_count)``：
          单次调用成功时触发，可在其中记录 cost、延迟等指标。
        - ``on_failure(client, error, attempt_count)``：
          每次尝试失败时触发（含重试循环内的每一次失败），
          可用于记录单次错误的详细日志。
        - ``on_switch(from_client, to_client, reason)``：
          从一个客户端切换到下一个客户端时触发，可用于告警或审计。

        Args:
            messages: 符合 OpenAI Chat Completions 格式的消息列表。
            on_success: 成功回调，接收当前客户端、响应字典和尝试次数。
            on_failure: 失败回调，接收当前客户端、异常对象和尝试次数。
            on_switch: 切换回调，接收上一个客户端、下一个客户端和切换原因描述。
            **kwargs: 透传给 ``BaseLLMClient.chat_completion`` 的额外参数。

        Returns:
            LLM 响应字典。

        Raises:
            NonRetryableError: 任意客户端返回不可重试错误时立即抛出。
            AllModelsExhaustedError: 所有客户端均在重试耗尽后失败。
        """
        self._failures = []

        for i, client in enumerate(self._clients):
            # ---- 触发切换回调（非首个客户端时）----
            if i > 0 and on_switch is not None:
                previous_client: BaseLLMClient = self._clients[i - 1]
                reason: str = (
                    f"模型 {previous_client.model_name} 调用失败，"
                    f"切换到备用模型 {client.model_name}"
                )
                on_switch(previous_client, client, reason)

            retrying: Retrying = Retrying(
                stop=self._retry_config.to_tenacity_stop(),
                wait=self._retry_config.to_tenacity_wait(),
                retry=retry_if_exception_type(
                    *self._retry_config.retryable_exceptions
                ),
                reraise=True,
            )

            try:
                for attempt in retrying:
                    with attempt:
                        try:
                            result: dict[str, Any] = client.chat_completion(
                                messages, **kwargs
                            )
                        except NonRetryableError as exc:
                            # 不可重试错误：通知回调后向上传播
                            if on_failure is not None:
                                on_failure(
                                    client,
                                    exc,
                                    attempt.retry_state.attempt_number,
                                )
                            raise
                        except Exception as exc:
                            # 单次尝试失败（可重试或未知异常）
                            if on_failure is not None:
                                on_failure(
                                    client,
                                    exc,
                                    attempt.retry_state.attempt_number,
                                )
                            raise
                        else:
                            # 调用成功
                            if on_success is not None:
                                on_success(
                                    client,
                                    result,
                                    attempt.retry_state.attempt_number,
                                )
                            return result
            except NonRetryableError:
                # 不可重试错误：直接向上传播，不切换到下一个客户端
                raise
            except Exception as exc:
                # 当前客户端重试耗尽
                self._failures.append(
                    {"model": client.model_name, "error": _sanitize_error_message(exc)}
                )
                continue

        # 所有客户端均已耗尽
        raise AllModelsExhaustedError(self._failures)
