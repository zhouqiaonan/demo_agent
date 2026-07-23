"""LLM Engine 重试策略配置模块。

本模块提供 ``RetryConfig`` 配置类，封装 tenacity 库的声明式重试策略，
将 stop（停止条件）、wait（等待策略）、retry（重试条件）三大维度统一管理。
同时提供 ``create_retry_decorator`` 函数，展示装饰器模式下的声明式重试用法。

典型用法（编程式）::

    config = RetryConfig(max_attempts=3, backoff="exponential")
    retrying = config.build_retrying()
    result = retrying(lambda: call_llm_api(prompt), prompt)

典型用法（装饰器式）::

    config = RetryConfig(max_attempts=5, backoff="linear", min_wait=2.0)
    retry_decorator = create_retry_decorator(config)

    @retry_decorator
    def call_llm(prompt: str) -> str:
        ...

排除不可重试异常的自定义用法::

    from tenacity import retry_if_exception_type, retry_if_not_exception_type

    # 仅重试 TransientError，拒绝 NonRetryableError
    config = RetryConfig(
        max_attempts=4,
        backoff="exponential",
        # 扩大可重试异常集
        retryable_exceptions=(TransientError, ConnectionError, TimeoutError),
    )

    # 构建时额外排除 NonRetryableError —— 即使它被包含在 retryable_exceptions 中
    @retry(
        stop=config.to_tenacity_stop(),
        wait=config.to_tenacity_wait(),
        retry=retry_if_exception_type(*config.retryable_exceptions)
        & retry_if_not_exception_type(NonRetryableError),
    )
    def safe_call_llm(prompt: str) -> str:
        ...
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from tenacity import (
    Retrying,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_fixed,
    wait_incrementing,
)
from tenacity.stop import stop_base
from tenacity.wait import wait_base

from llm_engine.exceptions import NonRetryableError, TransientError

# ---------------------------------------------------------------------------
# 合法的 backoff 策略名称集合
# ---------------------------------------------------------------------------

_VALID_BACKOFF_VALUES: frozenset[str] = frozenset({"exponential", "fixed", "linear"})


# ---------------------------------------------------------------------------
# RetryConfig
# ---------------------------------------------------------------------------


@dataclass
class RetryConfig:
    """重试策略配置，封装 tenacity 的 stop、wait、retry 策略。

    本类作为 LLM Engine 重试行为的声明式配置载体，提供三个维度：

    - **stop** —— 何时停止重试（如尝试次数上限）
    - **wait** —— 两次重试之间等待多长时间（固定 / 线性递增 / 指数退避）
    - **retry** —— 什么异常类型触发重试（默认为 ``TransientError``）

    各配置方法的返回值均为 tenacity 原生对象，可直接传入
    ``tenacity.retry`` 装饰器或 ``tenacity.Retrying`` 构造函数。

    Attributes:
        max_attempts: 最大尝试次数（包含首次调用），默认 3。
        backoff: 退避策略名称，支持 ``"exponential"`` | ``"fixed"`` | ``"linear"``，
            默认 ``"exponential"``。
        min_wait: 最小等待时间（秒），默认 1.0。
        max_wait: 最大等待时间（秒），默认 60.0。
        retryable_exceptions: 应触发重试的异常类型元组，默认仅 ``TransientError``。
            ``NonRetryableError`` 不在其中，因此默认不会对它进行重试。

    Example:
        >>> config = RetryConfig(max_attempts=3, backoff="exponential")
        >>> config.to_tenacity_stop()
        <tenacity.stop.stop_after_attempt object at ...>
        >>> config.to_tenacity_wait()
        <tenacity.wait.wait_exponential object at ...>
        >>> retrying = config.build_retrying()
        >>> retrying(lambda x: x / 0, 1)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        ZeroDivisionError: division by zero
    """

    # ---- 字段定义 ---------------------------------------------------------

    max_attempts: int = 3
    """最大尝试次数（包含首次调用）。"""

    backoff: str = "exponential"
    """退避策略名称。合法值: ``"exponential"`` | ``"fixed"`` | ``"linear"``。"""

    min_wait: float = 1.0
    """最小等待时间（秒），用于限制退避下界。"""

    max_wait: float = 60.0
    """最大等待时间（秒），用于限制退避上界。"""

    retryable_exceptions: tuple[type[Exception], ...] = (TransientError,)
    """应触发重试的异常类型元组。排除型条件可通过组合 tenacity 谓词实现。"""

    # ---- 校验 ------------------------------------------------------------

    def __post_init__(self) -> None:
        """校验配置参数的合法性。

        Raises:
            ValueError: 如果 ``backoff`` 不在合法策略集合内。
        """
        if self.backoff not in _VALID_BACKOFF_VALUES:
            raise ValueError(
                f"非法的 backoff 策略值: {self.backoff!r}，"
                f"合法值为: {', '.join(sorted(_VALID_BACKOFF_VALUES))}"
            )

    # ---- tenacity wait 转换 ---------------------------------------------

    def to_tenacity_wait(self) -> wait_base:
        """根据 ``backoff`` 字段返回对应的 tenacity wait 策略对象。

        三种策略的含义：

        - **exponential** —— 指数退避：
          第 n 次重试等待 ``2^(n-1) * min_wait`` 秒，最小不低于 ``min_wait``，
          最大不超过 ``max_wait``。
        - **fixed** —— 固定等待：
          每次重试固定等待 ``min_wait`` 秒。
        - **linear** —— 线性递增：
          第 n 次重试等待 ``n * min_wait`` 秒，最大不超过 ``max_wait``。

        Returns:
            tenacity ``wait_base`` 子类实例，可直接传入 ``Retrying`` 或 ``retry``
            装饰器。

        Example:
            >>> config = RetryConfig(backoff="exponential", min_wait=2.0, max_wait=30.0)
            >>> w = config.to_tenacity_wait()
            >>> isinstance(w, wait_base)
            True
        """
        if self.backoff == "exponential":
            return wait_exponential(
                multiplier=1,
                min=self.min_wait,  # type: ignore[arg-type]  # tenacity 接受 float
                max=self.max_wait,  # type: ignore[arg-type]
            )
        elif self.backoff == "fixed":
            return wait_fixed(self.min_wait)  # type: ignore[arg-type]
        else:  # linear
            return wait_incrementing(
                start=self.min_wait,  # type: ignore[arg-type]
                increment=self.min_wait,  # type: ignore[arg-type]
                max=self.max_wait,  # type: ignore[arg-type]
            )

    # ---- tenacity stop 转换 ----------------------------------------------

    def to_tenacity_stop(self) -> stop_base:
        """返回基于 ``max_attempts`` 的 tenacity stop 策略。

        使用 ``tenacity.stop_after_attempt``，表示最多尝试 ``max_attempts`` 次
        （含首次调用）后停止重试。

        Returns:
            tenacity ``stop_base`` 子类实例。

        Example:
            >>> config = RetryConfig(max_attempts=5)
            >>> s = config.to_tenacity_stop()
            >>> isinstance(s, stop_base)
            True
        """
        return stop_after_attempt(self.max_attempts)

    # ---- 构建 Retrying 对象（编程式用法）-------------------------------

    def build_retrying(self) -> Retrying:
        """构建配置好的 ``tenacity.Retrying`` 对象，用于编程式重试调用。

        返回的 ``Retrying`` 对象包含：

        - **stop**: ``stop_after_attempt(max_attempts)``
        - **wait**: 由 ``to_tenacity_wait()`` 返回的对应等待策略
        - **retry**: ``retry_if_exception_type(*retryable_exceptions)``
          （默认仅重试 ``TransientError``，自动忽略 ``NonRetryableError``）

        编程式用法示例::

            config = RetryConfig(max_attempts=3, backoff="exponential")
            retrying = config.build_retrying()

            def call_api(prompt: str) -> str:
                ...

            result = retrying(call_api, prompt)

        如需排除 ``NonRetryableError`` 等异常，可自行组合 tenacity 谓词::

            from tenacity import retry_if_not_exception_type

            custom_retry = retry_if_exception_type(*config.retryable_exceptions) \\
                & retry_if_not_exception_type(NonRetryableError)

            retrying = Retrying(
                stop=config.to_tenacity_stop(),
                wait=config.to_tenacity_wait(),
                retry=custom_retry,
            )

        Returns:
            配置完毕的 ``tenacity.Retrying`` 实例，可直接用于 ``__call__`` 执行
            附带重试逻辑的任意可调用对象。

        Raises:
            ValueError: 如果 ``backoff`` 字段不合法（在 ``__post_init__`` 中校验）。
        """
        return Retrying(
            stop=self.to_tenacity_stop(),
            wait=self.to_tenacity_wait(),
            retry=retry_if_exception_type(*self.retryable_exceptions),
        )


# ---------------------------------------------------------------------------
# 模块级工具函数
# ---------------------------------------------------------------------------


def create_retry_decorator(
    config: RetryConfig | None = None,
) -> Callable[..., Callable[..., object]]:
    """返回一个基于 ``RetryConfig`` 配置的 tenacity ``retry`` 装饰器。

    本函数展示声明式（装饰器）重试模式：将重试策略组装为可装饰任意函数的
    装饰器对象，使业务代码与重试逻辑完全解耦。

    生成的装饰器内部使用：
    - ``stop_after_attempt`` 控制最大尝试次数
    - ``wait_*`` 控制重试间隔策略
    - ``retry_if_exception_type`` 控制触发重试的异常类型

    Args:
        config: ``RetryConfig`` 实例。若为 ``None``，则使用默认配置
            （最多 3 次尝试、指数退避、仅重试 ``TransientError``）。

    Returns:
        一个 tenacity ``retry`` 装饰器实例，可直接用于装饰函数。

    Example:
        >>> # 默认配置的装饰器
        >>> decorator = create_retry_decorator()
        >>> @decorator
        ... def might_fail() -> str:
        ...     raise TransientError("临时错误")
        >>> might_fail()  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        TransientError: 临时错误

        >>> # 自定义配置的装饰器
        >>> config = RetryConfig(max_attempts=2, backoff="fixed", min_wait=0.01)
        >>> decorator = create_retry_decorator(config)
        >>> call_count = 0
        >>> @decorator
        ... def flaky() -> str:
        ...     nonlocal call_count
        ...     call_count += 1
        ...     if call_count < 2:
        ...         raise TransientError("重试中...")
        ...     return "success"
        >>> flaky()
        'success'

        >>> # NonRetryableError 不会被重试
        >>> config = RetryConfig(max_attempts=5, backoff="fixed", min_wait=0.01)
        >>> decorator = create_retry_decorator(config)
        >>> @decorator
        ... def bad_auth() -> str:
        ...     raise NonRetryableError("无效 API Key")
        >>> bad_auth()  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        NonRetryableError: 无效 API Key
    """
    if config is None:
        config = RetryConfig()

    return retry(
        stop=config.to_tenacity_stop(),
        wait=config.to_tenacity_wait(),
        retry=retry_if_exception_type(*config.retryable_exceptions),
        reraise=True,
    )
