"""Tests for llm_engine/fallback.py — FallbackManager."""

from unittest.mock import MagicMock

import pytest

from llm_engine.exceptions import AllModelsExhaustedError, NonRetryableError, TransientError
from llm_engine.fallback import FallbackManager
from llm_engine.retry import RetryConfig


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_mock_client(name: str) -> MagicMock:
    """创建一个具有 model_name 和 chat_completion 属性的 mock 客户端。

    Args:
        name: 模型名称。

    Returns:
        配置好的 MagicMock 实例。
    """
    client = MagicMock()
    client.model_name = name
    return client


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fast_retry() -> RetryConfig:
    """返回用于测试的快速重试配置（避免测试过慢）。"""
    return RetryConfig(max_attempts=2, backoff="fixed", min_wait=0.001, max_wait=0.01)


@pytest.fixture
def primary() -> MagicMock:
    """主模型 mock 客户端。"""
    return _make_mock_client("primary-model")


@pytest.fixture
def backup() -> MagicMock:
    """备用模型 mock 客户端。"""
    return _make_mock_client("backup-model")


# ---------------------------------------------------------------------------
# TestFallbackManagerConstruction
# ---------------------------------------------------------------------------


class TestFallbackManagerConstruction:
    """测试 FallbackManager 构造和属性。"""

    def test_empty_clients_raises_value_error(self) -> None:
        """空 clients 列表应抛出 ValueError。"""
        with pytest.raises(ValueError, match="clients 列表不能为空"):
            FallbackManager([])

    def test_clients_property(self, primary: MagicMock, fast_retry: RetryConfig) -> None:
        """clients 属性应返回所有注册客户端的元组。"""
        mgr = FallbackManager([primary], fast_retry)
        assert isinstance(mgr.clients, tuple)
        assert len(mgr.clients) == 1
        assert mgr.clients[0] is primary

    def test_clients_property_immutable(self, primary: MagicMock, fast_retry: RetryConfig) -> None:
        """clients 属性返回的元组不可修改。"""
        mgr = FallbackManager([primary], fast_retry)
        with pytest.raises(TypeError):
            mgr.clients[0] = _make_mock_client("other")  # type: ignore[index]

    def test_retry_config_property(self, primary: MagicMock, fast_retry: RetryConfig) -> None:
        """retry_config 属性应返回传入的 RetryConfig 实例。"""
        mgr = FallbackManager([primary], fast_retry)
        assert mgr.retry_config is fast_retry

    def test_default_retry_config(self, primary: MagicMock) -> None:
        """未提供 retry_config 时应使用默认配置。"""
        mgr = FallbackManager([primary])
        assert mgr.retry_config.max_attempts == 3
        assert mgr.retry_config.backoff == "exponential"

    def test_failure_history_initially_empty(self, primary: MagicMock, fast_retry: RetryConfig) -> None:
        """初始状态下 failure_history 应为空列表。"""
        mgr = FallbackManager([primary], fast_retry)
        assert mgr.failure_history == []


# ---------------------------------------------------------------------------
# TestFallbackManagerExecuteSuccess
# ---------------------------------------------------------------------------


class TestFallbackManagerExecuteSuccess:
    """测试 execute() 成功场景。"""

    def test_primary_succeeds(self, primary: MagicMock, fast_retry: RetryConfig) -> None:
        """主模型成功时，应直接返回其响应。"""
        primary.chat_completion.return_value = {"role": "assistant", "content": "hi"}
        mgr = FallbackManager([primary], fast_retry)

        result = mgr.execute([{"role": "user", "content": "hello"}])

        assert result == {"role": "assistant", "content": "hi"}
        primary.chat_completion.assert_called_once()

    def test_execute_passes_kwargs(self, primary: MagicMock, fast_retry: RetryConfig) -> None:
        """execute() 应将 **kwargs 透传给 chat_completion。"""
        primary.chat_completion.return_value = {"content": "ok"}
        mgr = FallbackManager([primary], fast_retry)

        mgr.execute(
            [{"role": "user", "content": "hello"}],
            temperature=0.5,
            max_tokens=100,
        )

        call_kwargs = primary.chat_completion.call_args[1]
        assert call_kwargs.get("temperature") == 0.5
        assert call_kwargs.get("max_tokens") == 100

    def test_primary_fails_backup_succeeds(
        self, primary: MagicMock, backup: MagicMock, fast_retry: RetryConfig
    ) -> None:
        """主模型失败后，应自动切换到备用模型并返回备用模型的响应。"""
        primary.chat_completion.side_effect = TransientError("timeout")
        backup.chat_completion.return_value = {"role": "assistant", "content": "from backup"}

        mgr = FallbackManager([primary, backup], fast_retry)

        result = mgr.execute([{"role": "user", "content": "hello"}])

        assert result == {"role": "assistant", "content": "from backup"}
        assert primary.chat_completion.call_count == fast_retry.max_attempts
        backup.chat_completion.assert_called_once()


# ---------------------------------------------------------------------------
# TestFallbackManagerExecuteFailure
# ---------------------------------------------------------------------------


class TestFallbackManagerExecuteFailure:
    """测试 execute() 失败场景。"""

    def test_all_models_exhausted(
        self, primary: MagicMock, backup: MagicMock, fast_retry: RetryConfig
    ) -> None:
        """所有模型均失败时，应抛出 AllModelsExhaustedError。"""
        primary.chat_completion.side_effect = TransientError("primary error")
        backup.chat_completion.side_effect = TransientError("backup error")

        mgr = FallbackManager([primary, backup], fast_retry)

        with pytest.raises(AllModelsExhaustedError) as exc_info:
            mgr.execute([{"role": "user", "content": "hello"}])

        assert len(exc_info.value.failed_models) == 2
        assert exc_info.value.failed_models[0]["model"] == "primary-model"
        assert exc_info.value.failed_models[1]["model"] == "backup-model"

    def test_non_retryable_error_stops_fallback(
        self, primary: MagicMock, backup: MagicMock, fast_retry: RetryConfig
    ) -> None:
        """NonRetryableError 应直接抛出，不切换到备用模型。"""
        primary.chat_completion.side_effect = NonRetryableError("invalid api key")
        # backup 不应被调用
        backup.chat_completion.return_value = {"content": "should not reach"}

        mgr = FallbackManager([primary, backup], fast_retry)

        with pytest.raises(NonRetryableError):
            mgr.execute([{"role": "user", "content": "hello"}])

        # 主模型被调用（首次即失败，NonRetryableError 不应重试）
        primary.chat_completion.assert_called()
        # 备用模型不应被调用
        backup.chat_completion.assert_not_called()

    def test_failure_history_after_all_exhausted(
        self, primary: MagicMock, fast_retry: RetryConfig
    ) -> None:
        """所有模型耗尽后，failure_history 应包含失败记录。"""
        primary.chat_completion.side_effect = TransientError("timeout")

        mgr = FallbackManager([primary], fast_retry)

        with pytest.raises(AllModelsExhaustedError):
            mgr.execute([{"role": "user", "content": "hello"}])

        assert len(mgr.failure_history) == 1
        assert mgr.failure_history[0]["model"] == "primary-model"
        assert "timeout" in mgr.failure_history[0]["error"]


# ---------------------------------------------------------------------------
# TestFallbackManagerWithMultipleBackups
# ---------------------------------------------------------------------------


class TestFallbackManagerWithMultipleBackups:
    """测试多个备用模型场景。"""

    def test_second_backup_succeeds(self, fast_retry: RetryConfig) -> None:
        """前两个模型失败，第三个成功。"""
        client_a = _make_mock_client("model-a")
        client_b = _make_mock_client("model-b")
        client_c = _make_mock_client("model-c")

        client_a.chat_completion.side_effect = TransientError("fail a")
        client_b.chat_completion.side_effect = TransientError("fail b")
        client_c.chat_completion.return_value = {"content": "third works"}

        mgr = FallbackManager([client_a, client_b, client_c], fast_retry)
        result = mgr.execute([{"role": "user", "content": "hello"}])

        assert result == {"content": "third works"}
        assert client_c.chat_completion.call_count == 1


# ---------------------------------------------------------------------------
# TestExecuteWithCallbacks
# ---------------------------------------------------------------------------


class TestExecuteWithCallbacks:
    """测试 execute_with_callbacks() 回调钩子。"""

    def test_on_success_fires(self, primary: MagicMock, fast_retry: RetryConfig) -> None:
        """成功时 on_success 回调应被触发。"""
        primary.chat_completion.return_value = {"content": "ok"}

        success_log: list[dict] = []

        def on_success(client, response, attempts):
            success_log.append({
                "model": client.model_name,
                "response": response,
                "attempts": attempts,
            })

        mgr = FallbackManager([primary], fast_retry)
        mgr.execute_with_callbacks(
            [{"role": "user", "content": "hello"}],
            on_success=on_success,
        )

        assert len(success_log) == 1
        assert success_log[0]["model"] == "primary-model"
        assert success_log[0]["response"] == {"content": "ok"}

    def test_on_failure_fires(self, primary: MagicMock, fast_retry: RetryConfig) -> None:
        """TransientError 时 on_failure 回调应被触发。"""
        primary.chat_completion.side_effect = TransientError("timeout")

        failure_log: list[dict] = []

        def on_failure(client, error, attempts):
            failure_log.append({
                "model": client.model_name,
                "error": str(error),
                "attempts": attempts,
            })

        mgr = FallbackManager([primary], fast_retry)
        with pytest.raises(AllModelsExhaustedError):
            mgr.execute_with_callbacks(
                [{"role": "user", "content": "hello"}],
                on_failure=on_failure,
            )

        # 每个尝试都应触发一次 on_failure
        assert len(failure_log) == fast_retry.max_attempts
        assert all(item["model"] == "primary-model" for item in failure_log)

    def test_on_switch_fires(self, primary: MagicMock, backup: MagicMock, fast_retry: RetryConfig) -> None:
        """切换到备用模型时 on_switch 回调应被触发。"""
        primary.chat_completion.side_effect = TransientError("fail")
        backup.chat_completion.return_value = {"content": "backup ok"}

        switch_log: list[dict] = []

        def on_switch(from_client, to_client, reason):
            switch_log.append({
                "from": from_client.model_name,
                "to": to_client.model_name,
                "reason": reason,
            })

        mgr = FallbackManager([primary, backup], fast_retry)
        mgr.execute_with_callbacks(
            [{"role": "user", "content": "hello"}],
            on_switch=on_switch,
        )

        assert len(switch_log) == 1
        assert switch_log[0]["from"] == "primary-model"
        assert switch_log[0]["to"] == "backup-model"

    def test_on_failure_on_non_retryable(
        self, primary: MagicMock, fast_retry: RetryConfig
    ) -> None:
        """NonRetryableError 也应触发 on_failure 回调后才抛出。"""
        primary.chat_completion.side_effect = NonRetryableError("bad key")

        failure_log: list[dict] = []

        def on_failure(client, error, attempts):
            failure_log.append({"model": client.model_name, "error": str(error)})

        mgr = FallbackManager([primary], fast_retry)
        with pytest.raises(NonRetryableError):
            mgr.execute_with_callbacks(
                [{"role": "user", "content": "hello"}],
                on_failure=on_failure,
            )

        assert len(failure_log) == 1
        assert "bad key" in failure_log[0]["error"]
