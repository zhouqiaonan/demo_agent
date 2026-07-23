"""Tests for llm_engine/retry.py — RetryConfig and create_retry_decorator."""

from unittest.mock import MagicMock

import pytest
from tenacity import RetryError, Retrying
from tenacity.retry import retry_base
from tenacity.stop import stop_after_attempt, stop_base
from tenacity.wait import wait_base, wait_exponential, wait_fixed, wait_incrementing

from llm_engine.exceptions import NonRetryableError, TransientError
from llm_engine.retry import RetryConfig, create_retry_decorator


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_config() -> RetryConfig:
    """返回默认配置的 RetryConfig 实例。"""
    return RetryConfig()


@pytest.fixture
def fast_retry_config() -> RetryConfig:
    """返回用于测试的快速重试配置（避免测试过慢）。"""
    return RetryConfig(max_attempts=3, backoff="fixed", min_wait=0.001, max_wait=0.01)


# ---------------------------------------------------------------------------
# TestRetryConfigDefaults
# ---------------------------------------------------------------------------


class TestRetryConfigDefaults:
    """测试 RetryConfig 默认值。"""

    def test_default_max_attempts(self, default_config: RetryConfig) -> None:
        """默认 max_attempts 应为 3。"""
        assert default_config.max_attempts == 3

    def test_default_backoff(self, default_config: RetryConfig) -> None:
        """默认 backoff 应为 'exponential'。"""
        assert default_config.backoff == "exponential"

    def test_default_min_wait(self, default_config: RetryConfig) -> None:
        """默认 min_wait 应为 1.0。"""
        assert default_config.min_wait == 1.0

    def test_default_max_wait(self, default_config: RetryConfig) -> None:
        """默认 max_wait 应为 60.0。"""
        assert default_config.max_wait == 60.0

    def test_default_retryable_exceptions(self, default_config: RetryConfig) -> None:
        """默认仅重试 TransientError。"""
        assert default_config.retryable_exceptions == (TransientError,)


# ---------------------------------------------------------------------------
# TestRetryConfigCustomValues
# ---------------------------------------------------------------------------


class TestRetryConfigCustomValues:
    """测试 RetryConfig 自定义构造。"""

    def test_custom_max_attempts(self) -> None:
        """自定义 max_attempts=5 应生效。"""
        config = RetryConfig(max_attempts=5)
        assert config.max_attempts == 5

    def test_custom_backoff_fixed(self) -> None:
        """自定义 backoff='fixed' 应生效。"""
        config = RetryConfig(backoff="fixed")
        assert config.backoff == "fixed"

    def test_custom_backoff_linear(self) -> None:
        """自定义 backoff='linear' 应生效。"""
        config = RetryConfig(backoff="linear")
        assert config.backoff == "linear"

    def test_custom_min_and_max_wait(self) -> None:
        """自定义 min_wait=2.5, max_wait=30.0 应生效。"""
        config = RetryConfig(min_wait=2.5, max_wait=30.0)
        assert config.min_wait == 2.5
        assert config.max_wait == 30.0


# ---------------------------------------------------------------------------
# TestToTenacityWait
# ---------------------------------------------------------------------------


class TestToTenacityWait:
    """测试 to_tenacity_wait() 三种退避策略。"""

    def test_exponential_backoff_returns_wait_exponential(self, default_config: RetryConfig) -> None:
        """exponential 退避策略应返回 wait_exponential 实例。"""
        wait = default_config.to_tenacity_wait()
        assert isinstance(wait, wait_base)
        assert isinstance(wait, wait_exponential)

    def test_fixed_backoff_returns_wait_fixed(self) -> None:
        """fixed 退避策略应返回 wait_fixed 实例。"""
        config = RetryConfig(backoff="fixed")
        wait = config.to_tenacity_wait()
        assert isinstance(wait, wait_fixed)

    def test_linear_backoff_returns_wait_incrementing(self) -> None:
        """linear 退避策略应返回 wait_incrementing 实例。"""
        config = RetryConfig(backoff="linear")
        wait = config.to_tenacity_wait()
        assert isinstance(wait, wait_incrementing)

    def test_exponential_uses_min_and_max_wait(self) -> None:
        """exponential 策略应使用 min_wait 和 max_wait 参数。"""
        config = RetryConfig(backoff="exponential", min_wait=2.0, max_wait=30.0)
        wait = config.to_tenacity_wait()
        assert isinstance(wait, wait_exponential)

    def test_fixed_uses_min_wait(self) -> None:
        """fixed 策略的等待时间应等于 min_wait。"""
        config = RetryConfig(backoff="fixed", min_wait=0.5)
        wait = config.to_tenacity_wait()
        assert isinstance(wait, wait_fixed)


# ---------------------------------------------------------------------------
# TestToTenacityStop
# ---------------------------------------------------------------------------


class TestToTenacityStop:
    """测试 to_tenacity_stop()。"""

    def test_returns_stop_after_attempt(self, default_config: RetryConfig) -> None:
        """应返回 stop_after_attempt 实例。"""
        stop = default_config.to_tenacity_stop()
        assert isinstance(stop, stop_base)
        assert isinstance(stop, stop_after_attempt)

    def test_max_attempts_matches_config(self) -> None:
        """max_attempts=5 时 stop 策略应反映此值。"""
        config = RetryConfig(max_attempts=5)
        stop = config.to_tenacity_stop()
        assert stop.max_attempt_number == 5  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# TestBuildRetrying
# ---------------------------------------------------------------------------


class TestBuildRetrying:
    """测试 build_retrying() 返回 Retrying 对象。"""

    def test_returns_retrying(self, default_config: RetryConfig) -> None:
        """build_retrying() 应返回 Retrying 实例。"""
        retrying = default_config.build_retrying()
        assert isinstance(retrying, Retrying)

    def test_retrying_has_stop_strategy(self, default_config: RetryConfig) -> None:
        """返回的 Retrying 对象应包含 stop 策略。"""
        retrying = default_config.build_retrying()
        assert retrying.stop is not None

    def test_retrying_has_wait_strategy(self, default_config: RetryConfig) -> None:
        """返回的 Retrying 对象应包含 wait 策略。"""
        retrying = default_config.build_retrying()
        assert retrying.wait is not None


# ---------------------------------------------------------------------------
# TestInvalidBackoff
# ---------------------------------------------------------------------------


class TestInvalidBackoff:
    """测试非法 backoff 值抛出 ValueError。"""

    def test_invalid_backoff_raises_value_error(self) -> None:
        """非法的 backoff 值应在 __post_init__ 中抛出 ValueError。"""
        with pytest.raises(ValueError, match="非法的 backoff 策略值"):
            RetryConfig(backoff="invalid")


# ---------------------------------------------------------------------------
# TestCreateRetryDecorator
# ---------------------------------------------------------------------------


class TestCreateRetryDecorator:
    """测试 create_retry_decorator() 返回可用的装饰器。"""

    def test_returns_callable(self) -> None:
        """create_retry_decorator 应返回可调用对象（装饰器）。"""
        decorator = create_retry_decorator()
        assert callable(decorator)

    def test_decorator_with_default_config(self) -> None:
        """使用默认配置的装饰器装饰函数应正常工作。"""
        decorator = create_retry_decorator()

        call_count: list[int] = [0]

        @decorator
        def flaky() -> str:
            call_count[0] += 1
            if call_count[0] < 2:
                raise TransientError("临时错误")
            return "success"

        result = flaky()
        assert result == "success"
        assert call_count[0] == 2

    def test_decorator_with_custom_config(self) -> None:
        """使用自定义配置的装饰器应反映配置参数。"""
        config = RetryConfig(max_attempts=5, backoff="fixed", min_wait=0.001)
        decorator = create_retry_decorator(config)

        call_count: list[int] = [0]

        @decorator
        def flaky() -> str:
            call_count[0] += 1
            if call_count[0] < 3:
                raise TransientError("重试中...")
            return "done"

        result = flaky()
        assert result == "done"
        assert call_count[0] == 3


# ---------------------------------------------------------------------------
# TestRetryBehavior
# ---------------------------------------------------------------------------


class TestRetryBehavior:
    """测试 tenacity 实际重试行为。"""

    def test_retry_on_transient_error(self, fast_retry_config: RetryConfig) -> None:
        """TransientError 会触发重试。"""
        mock_client = MagicMock()
        mock_client.chat_completion.side_effect = [
            TransientError("timeout"),
            TransientError("rate limit"),
            {"content": "finally works"},
        ]

        result: dict | None = None
        for attempt in fast_retry_config.build_retrying():
            with attempt:
                result = mock_client.chat_completion()

        assert result == {"content": "finally works"}
        assert mock_client.chat_completion.call_count == 3

    def test_no_retry_on_non_retryable_error(self, fast_retry_config: RetryConfig) -> None:
        """NonRetryableError 不会触发重试。"""
        mock_client = MagicMock()
        mock_client.chat_completion.side_effect = NonRetryableError("invalid api key")

        with pytest.raises(NonRetryableError):
            for attempt in fast_retry_config.build_retrying():
                with attempt:
                    mock_client.chat_completion()

        assert mock_client.chat_completion.call_count == 1

    def test_retry_stops_after_max_attempts(self, fast_retry_config: RetryConfig) -> None:
        """重试应在达到 max_attempts 后停止，抛出 RetryError。"""
        mock_client = MagicMock()
        mock_client.chat_completion.side_effect = TransientError("always fail")

        with pytest.raises(RetryError):
            for attempt in fast_retry_config.build_retrying():
                with attempt:
                    mock_client.chat_completion()

        assert mock_client.chat_completion.call_count == fast_retry_config.max_attempts

    def test_first_attempt_success_no_retry(self, fast_retry_config: RetryConfig) -> None:
        """首次调用成功时不应发生重试。"""
        mock_client = MagicMock()
        mock_client.chat_completion.return_value = {"content": "ok"}

        for attempt in fast_retry_config.build_retrying():
            with attempt:
                result = mock_client.chat_completion()

        assert result == {"content": "ok"}
        assert mock_client.chat_completion.call_count == 1
