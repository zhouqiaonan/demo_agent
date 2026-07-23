"""Tests for llm_engine/engine.py — LLMEngine and LLMEngineBuilder."""

from unittest.mock import MagicMock

import pytest

from llm_engine.cost import CostTracker
from llm_engine.engine import LLMEngine, LLMEngineBuilder
from llm_engine.exceptions import TransientError
from llm_engine.metrics import LLMMetrics
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


def _make_success_response(
    content: str = "Hello!",
    model: str = "gpt-4o",
    usage: dict | None = None,
) -> dict:
    """构造一个标准的成功响应字典。

    Args:
        content: 回复内容。
        model: 模型名称。
        usage: Token 用量字典。

    Returns:
        符合 chat_completion 返回格式的字典。
    """
    if usage is None:
        usage = {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        }
    return {
        "role": "assistant",
        "content": content,
        "model": model,
        "usage": usage,
        "finish_reason": "stop",
    }


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fast_retry() -> RetryConfig:
    """返回用于测试的快速重试配置。"""
    return RetryConfig(max_attempts=2, backoff="fixed", min_wait=0.001, max_wait=0.01)


@pytest.fixture
def primary_client() -> MagicMock:
    """主模型 mock 客户端。"""
    client = _make_mock_client("primary-model")
    client.chat_completion.return_value = _make_success_response()
    return client


@pytest.fixture
def backup_client() -> MagicMock:
    """备用模型 mock 客户端。"""
    client = _make_mock_client("backup-model")
    client.chat_completion.return_value = _make_success_response(
        content="from backup",
        model="backup-model",
    )
    return client


# ---------------------------------------------------------------------------
# TestLLMEngineDirectConstruction
# ---------------------------------------------------------------------------


class TestLLMEngineDirectConstruction:
    """测试 LLMEngine 直接构造方式。"""

    def test_construct_with_only_primary(self, primary_client: MagicMock) -> None:
        """仅传入 primary 客户端应正常构造。"""
        engine = LLMEngine(primary=primary_client)
        assert engine.primary is primary_client
        assert engine.fallbacks == []
        assert engine.metrics is None
        assert engine.cost_tracker is None

    def test_construct_with_fallbacks(
        self, primary_client: MagicMock, backup_client: MagicMock
    ) -> None:
        """传入 fallbacks 应在属性中反映。"""
        engine = LLMEngine(primary=primary_client, fallbacks=[backup_client])
        assert engine.primary is primary_client
        assert len(engine.fallbacks) == 1
        assert engine.fallbacks[0] is backup_client

    def test_construct_with_metrics(self, primary_client: MagicMock) -> None:
        """传入 metrics 应在属性中反映。"""
        metrics = LLMMetrics()
        engine = LLMEngine(primary=primary_client, metrics=metrics)
        assert engine.metrics is metrics

    def test_construct_with_cost_tracker(self, primary_client: MagicMock) -> None:
        """传入 cost_tracker 应在属性中反映。"""
        ct = CostTracker()
        engine = LLMEngine(primary=primary_client, cost_tracker=ct)
        assert engine.cost_tracker is ct

    def test_construct_with_retry_config(self, primary_client: MagicMock) -> None:
        """传入 retry_config 应生效。"""
        config = RetryConfig(max_attempts=5, backoff="linear")
        engine = LLMEngine(primary=primary_client, retry_config=config)
        # retry_config 不直接暴露属性，但可通过 fallback 间接验证
        assert engine.primary is primary_client


# ---------------------------------------------------------------------------
# TestLLMEngineBuilder
# ---------------------------------------------------------------------------


class TestLLMEngineBuilder:
    """测试 LLMEngine Builder 模式。"""

    def test_build_simple(self, primary_client: MagicMock) -> None:
        """最简单的 Builder 构造应成功。"""
        engine = (
            LLMEngine.builder()
            .primary(primary_client)
            .build()
        )
        assert engine.primary is primary_client

    def test_builder_chain_with_all_options(
        self, primary_client: MagicMock, backup_client: MagicMock
    ) -> None:
        """完整的 Builder 链式调用应正常工作。"""
        engine = (
            LLMEngine.builder()
            .primary(primary_client)
            .add_fallback(backup_client)
            .with_retry(max_attempts=5, backoff="linear")
            .with_metrics()
            .with_cost_tracking()
            .build()
        )
        assert engine.primary is primary_client
        assert len(engine.fallbacks) == 1
        assert engine.fallbacks[0] is backup_client
        assert engine.metrics is not None
        assert engine.cost_tracker is not None

    def test_builder_add_multiple_fallbacks(
        self, primary_client: MagicMock, backup_client: MagicMock
    ) -> None:
        """Builder 支持多次 add_fallback。"""
        client_c = _make_mock_client("model-c")
        engine = (
            LLMEngine.builder()
            .primary(primary_client)
            .add_fallback(backup_client)
            .add_fallback(client_c)
            .build()
        )
        assert len(engine.fallbacks) == 2

    def test_build_without_primary_raises(self) -> None:
        """未设置 primary 直接 build() 应抛出 ValueError。"""
        with pytest.raises(ValueError, match="primary"):
            LLMEngine.builder().build()

    def test_builder_returns_different_engine(self) -> None:
        """每次 build() 应返回新的 LLMEngine 实例。"""
        client = _make_mock_client("model-a")
        builder = LLMEngine.builder().primary(client)
        engine1 = builder.build()
        engine2 = builder.build()
        assert engine1 is not engine2


# ---------------------------------------------------------------------------
# TestLLMEngineChat
# ---------------------------------------------------------------------------


class TestLLMEngineChat:
    """测试 chat() 方法返回结构。"""

    def test_chat_returns_standardized_dict(
        self, primary_client: MagicMock, fast_retry: RetryConfig
    ) -> None:
        """chat() 应返回包含标准字段的字典。"""
        engine = LLMEngine(primary=primary_client, retry_config=fast_retry)
        result = engine.chat([{"role": "user", "content": "hello"}])

        assert isinstance(result, dict)
        assert "content" in result
        assert "model" in result
        assert "usage" in result
        assert "attempts" in result
        assert "cost" in result
        assert "finish_reason" in result
        assert "raw_response" in result

    def test_chat_content_field(self, primary_client: MagicMock, fast_retry: RetryConfig) -> None:
        """chat() 响应中的 content 应为模型回复文本。"""
        primary_client.chat_completion.return_value = _make_success_response(content="你好！")
        engine = LLMEngine(primary=primary_client, retry_config=fast_retry)
        result = engine.chat([{"role": "user", "content": "hello"}])
        assert result["content"] == "你好！"

    def test_chat_model_field(self, primary_client: MagicMock, fast_retry: RetryConfig) -> None:
        """chat() 响应中的 model 应为实际使用的模型名称。"""
        primary_client.chat_completion.return_value = _make_success_response(
            model="gpt-4o"
        )
        engine = LLMEngine(primary=primary_client, retry_config=fast_retry)
        result = engine.chat([{"role": "user", "content": "hello"}])
        assert result["model"] == "gpt-4o"

    def test_chat_usage_field(self, primary_client: MagicMock, fast_retry: RetryConfig) -> None:
        """chat() 响应中的 usage 应为 Token 用量信息。"""
        usage = {"prompt_tokens": 50, "completion_tokens": 30, "total_tokens": 80}
        primary_client.chat_completion.return_value = _make_success_response(usage=usage)
        engine = LLMEngine(primary=primary_client, retry_config=fast_retry)
        result = engine.chat([{"role": "user", "content": "hello"}])
        assert result["usage"] == usage

    def test_chat_attempts_field(self, primary_client: MagicMock, fast_retry: RetryConfig) -> None:
        """chat() 响应中的 attempts 应记录尝试次数。"""
        engine = LLMEngine(primary=primary_client, retry_config=fast_retry)
        result = engine.chat([{"role": "user", "content": "hello"}])
        assert result["attempts"] >= 1

    def test_chat_finish_reason_field(
        self, primary_client: MagicMock, fast_retry: RetryConfig
    ) -> None:
        """chat() 响应中的 finish_reason 应返回正确值。"""
        engine = LLMEngine(primary=primary_client, retry_config=fast_retry)
        result = engine.chat([{"role": "user", "content": "hello"}])
        assert result["finish_reason"] == "stop"

    def test_chat_raw_response_field(
        self, primary_client: MagicMock, fast_retry: RetryConfig
    ) -> None:
        """chat() 响应中的 raw_response 应为原始返回字典。"""
        raw = _make_success_response(content="raw", model="raw-model")
        primary_client.chat_completion.return_value = raw
        engine = LLMEngine(primary=primary_client, retry_config=fast_retry)
        result = engine.chat([{"role": "user", "content": "hello"}])
        assert result["raw_response"] is raw


# ---------------------------------------------------------------------------
# TestLLMEngineChatWithConfig
# ---------------------------------------------------------------------------


class TestLLMEngineChatWithConfig:
    """测试 chat() 方法使用 GenerationConfig。"""

    def test_chat_with_config(self, primary_client: MagicMock, fast_retry: RetryConfig) -> None:
        """传入 GenerationConfig 应合并到 kwargs 透传给客户端。"""
        from function_caller.config import GenerationConfig

        engine = LLMEngine(primary=primary_client, retry_config=fast_retry)
        config = GenerationConfig(temperature=0.3, max_tokens=512)

        engine.chat(
            [{"role": "user", "content": "hello"}],
            config=config,
        )

        # 验证 temperature 和 max_tokens 被传递给 chat_completion
        call_kwargs = primary_client.chat_completion.call_args[1]
        assert call_kwargs.get("temperature") == 0.3
        assert call_kwargs.get("max_tokens") == 512


# ---------------------------------------------------------------------------
# TestLLMEngineStream
# ---------------------------------------------------------------------------


class TestLLMEngineStream:
    """测试 stream() 方法。"""

    def test_stream_raises_not_implemented(self, primary_client: MagicMock) -> None:
        """stream() 应抛出 NotImplementedError。"""
        engine = LLMEngine(primary=primary_client)
        with pytest.raises(NotImplementedError, match="流式调用暂未实现"):
            engine.stream([{"role": "user", "content": "hello"}])


# ---------------------------------------------------------------------------
# TestLLMEngineProperties
# ---------------------------------------------------------------------------


class TestLLMEngineProperties:
    """测试 LLMEngine 的只读属性。"""

    def test_primary_property(self, primary_client: MagicMock) -> None:
        """primary 属性应返回构造函数中传入的客户端。"""
        engine = LLMEngine(primary=primary_client)
        assert engine.primary is primary_client

    def test_fallbacks_property_empty_by_default(self, primary_client: MagicMock) -> None:
        """未传入 fallbacks 时，属性应返回空列表。"""
        engine = LLMEngine(primary=primary_client)
        assert engine.fallbacks == []

    def test_fallbacks_property_with_backups(
        self, primary_client: MagicMock, backup_client: MagicMock
    ) -> None:
        """传入 fallbacks 时，属性应返回包含它们的列表。"""
        engine = LLMEngine(primary=primary_client, fallbacks=[backup_client])
        assert engine.fallbacks == [backup_client]

    def test_metrics_property(self, primary_client: MagicMock) -> None:
        """metrics 属性应返回传入的或 None。"""
        engine = LLMEngine(primary=primary_client)
        assert engine.metrics is None

        metrics = LLMMetrics()
        engine2 = LLMEngine(primary=primary_client, metrics=metrics)
        assert engine2.metrics is metrics

    def test_cost_tracker_property(self, primary_client: MagicMock) -> None:
        """cost_tracker 属性应返回传入的或 None。"""
        engine = LLMEngine(primary=primary_client)
        assert engine.cost_tracker is None

        ct = CostTracker()
        engine2 = LLMEngine(primary=primary_client, cost_tracker=ct)
        assert engine2.cost_tracker is ct

    def test_total_cost_without_tracker(self, primary_client: MagicMock) -> None:
        """未配置 cost_tracker 时 total_cost 应返回 0.0。"""
        engine = LLMEngine(primary=primary_client)
        assert engine.total_cost == 0.0

    def test_total_cost_with_tracker(
        self, primary_client: MagicMock, fast_retry: RetryConfig
    ) -> None:
        """配置 cost_tracker 后，chat() 调用应累加费用。"""
        # 使用定价表内的模型名才能累计费用
        primary = _make_mock_client("gpt-4o")
        primary.chat_completion.return_value = _make_success_response(
            model="gpt-4o",
            usage={"prompt_tokens": 1000, "completion_tokens": 500, "total_tokens": 1500},
        )

        ct = CostTracker()
        engine = LLMEngine(
            primary=primary,
            retry_config=fast_retry,
            cost_tracker=ct,
        )
        engine.chat([{"role": "user", "content": "hello"}])
        # 使用了 token 用量，费用应大于 0
        assert engine.total_cost > 0

    def test_total_calls_without_tracker(self, primary_client: MagicMock) -> None:
        """未配置 cost_tracker 时 total_calls 应返回 0。"""
        engine = LLMEngine(primary=primary_client)
        assert engine.total_calls == 0

    def test_total_calls_with_tracker(
        self, primary_client: MagicMock, fast_retry: RetryConfig
    ) -> None:
        """配置 cost_tracker 后，chat() 调用应递增 total_calls。"""
        ct = CostTracker()
        engine = LLMEngine(
            primary=primary_client,
            retry_config=fast_retry,
            cost_tracker=ct,
        )
        engine.chat([{"role": "user", "content": "hello"}])
        assert engine.total_calls == 1


# ---------------------------------------------------------------------------
# TestLLMEngineFallback
# ---------------------------------------------------------------------------


class TestLLMEngineFallback:
    """测试 LLMEngine Fallback 场景。"""

    def test_primary_fails_backup_succeeds(
        self,
        primary_client: MagicMock,
        backup_client: MagicMock,
        fast_retry: RetryConfig,
    ) -> None:
        """主模型失败后切换到备用模型，返回的 model 应为备用模型名。"""
        primary_client.chat_completion.side_effect = TransientError("primary timeout")
        backup_client.chat_completion.return_value = _make_success_response(
            content="from backup",
            model="backup-model",
        )

        engine = LLMEngine(
            primary=primary_client,
            fallbacks=[backup_client],
            retry_config=fast_retry,
        )

        result = engine.chat([{"role": "user", "content": "hello"}])

        assert result["content"] == "from backup"
        assert result["model"] == "backup-model"
        # 备用模型应被调用
        backup_client.chat_completion.assert_called()

    def test_metrics_recorded_on_fallback_success(
        self,
        primary_client: MagicMock,
        backup_client: MagicMock,
        fast_retry: RetryConfig,
    ) -> None:
        """Fallback 成功后，metrics 应记录一次成功的调用。"""
        primary_client.chat_completion.side_effect = TransientError("primary timeout")
        backup_client.chat_completion.return_value = _make_success_response(
            content="from backup",
            model="backup-model",
        )

        metrics = LLMMetrics()
        engine = LLMEngine(
            primary=primary_client,
            fallbacks=[backup_client],
            retry_config=fast_retry,
            metrics=metrics,
        )

        engine.chat([{"role": "user", "content": "hello"}])

        summary = metrics.get_summary()
        # 应记录了一次成功的调用（备份模型不在白名单中，被归入 "unknown"）
        assert 'model="unknown",status="success"' in summary

    def test_cost_tracked_on_fallback_success(
        self,
        primary_client: MagicMock,
        backup_client: MagicMock,
        fast_retry: RetryConfig,
    ) -> None:
        """Fallback 成功后，cost_tracker 应正确追踪费用。"""
        # 使用定价表内的模型名确保费用 > 0
        primary_client.model_name = "gpt-4o"
        primary_client.chat_completion.side_effect = TransientError("primary timeout")
        backup_client.model_name = "gpt-4o"
        backup_client.chat_completion.return_value = _make_success_response(
            content="from backup",
            model="gpt-4o",
            usage={"prompt_tokens": 1000, "completion_tokens": 500, "total_tokens": 1500},
        )

        ct = CostTracker()
        engine = LLMEngine(
            primary=primary_client,
            fallbacks=[backup_client],
            retry_config=fast_retry,
            cost_tracker=ct,
        )

        result = engine.chat([{"role": "user", "content": "hello"}])

        assert result["cost"] > 0
        assert engine.total_calls == 1
