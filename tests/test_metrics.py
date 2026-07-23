"""Tests for llm_engine/metrics.py — LLMMetrics."""

import pytest

from llm_engine.metrics import LLMMetrics


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def metrics() -> LLMMetrics:
    """返回一个新的 LLMMetrics 实例，并在 each 测试后重置。"""
    m = LLMMetrics()
    yield m
    m.reset()


# ---------------------------------------------------------------------------
# TestLLMMetricsRecordCallSuccess
# ---------------------------------------------------------------------------


class TestLLMMetricsRecordCallSuccess:
    """测试 record_call(success=True) 的行为。"""

    def test_success_increments_call_total(self, metrics: LLMMetrics) -> None:
        """success=True 应增加 call_total 指标。"""
        metrics.record_call(
            model="gpt-4o",
            tokens={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
            latency_ms=1234.5,
            success=True,
        )
        summary = metrics.get_summary()
        assert 'llm_call_total{model="gpt-4o",status="success"}' in summary

    def test_success_records_token_usage(self, metrics: LLMMetrics) -> None:
        """success=True 应记录 prompt/completion/total 三类 token 用量。"""
        metrics.record_call(
            model="gpt-4o",
            tokens={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
            latency_ms=500.0,
            success=True,
        )
        summary = metrics.get_summary()
        assert 'llm_token_usage_total{model="gpt-4o",type="prompt"}' in summary
        assert 'llm_token_usage_total{model="gpt-4o",type="completion"}' in summary
        assert 'llm_token_usage_total{model="gpt-4o",type="total"}' in summary

    def test_success_records_latency(self, metrics: LLMMetrics) -> None:
        """success=True 应记录延迟指标。"""
        metrics.record_call(
            model="gpt-4o",
            tokens={"prompt_tokens": 10},
            latency_ms=2000.0,
            success=True,
        )
        summary = metrics.get_summary()
        assert 'llm_latency_seconds' in summary

    def test_token_counts_in_summary(self, metrics: LLMMetrics) -> None:
        """Summary 中应包含 token 数量信息。"""
        metrics.record_call(
            model="deepseek-chat",
            tokens={"prompt_tokens": 500, "completion_tokens": 200, "total_tokens": 700},
            latency_ms=800.0,
            success=True,
        )
        summary = metrics.get_summary()
        # 验证 prompt token 计数
        assert 'model="deepseek-chat",type="prompt"} 500.0' in summary
        assert 'model="deepseek-chat",type="completion"} 200.0' in summary
        assert 'model="deepseek-chat",type="total"} 700.0' in summary

    def test_multiple_calls_accumulate(self, metrics: LLMMetrics) -> None:
        """多次 record_call 应正确累加指标。"""
        for _ in range(3):
            metrics.record_call(
                model="gpt-4o",
                tokens={"prompt_tokens": 100, "total_tokens": 100},
                latency_ms=1000.0,
                success=True,
            )
        summary = metrics.get_summary()
        # 3 次调用，每次 100 prompt tokens
        assert 'model="gpt-4o",type="prompt"} 300.0' in summary


# ---------------------------------------------------------------------------
# TestLLMMetricsRecordCallFailure
# ---------------------------------------------------------------------------


class TestLLMMetricsRecordCallFailure:
    """测试 record_call(success=False) 的行为。"""

    def test_failure_increments_call_total_only(self, metrics: LLMMetrics) -> None:
        """success=False 应仅增加 call_total，不记录 token 和延迟。"""
        metrics.record_call(
            model="gpt-4o",
            tokens={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
            latency_ms=500.0,
            success=False,
        )
        summary = metrics.get_summary()

        # 应包含失败的 call_total
        assert 'llm_call_total{model="gpt-4o",status="error"}' in summary
        # 不应记录 token 用量（成功时才记录）
        assert 'model="gpt-4o",type="prompt"}' not in summary
        # 不应有延迟数据行（HELP/TYPE 线由注册产生，但无实际数据点）
        assert 'llm_latency_seconds_bucket{' not in summary
        assert 'llm_latency_seconds_count{' not in summary

    def test_failure_bumps_call_total_error_count(self, metrics: LLMMetrics) -> None:
        """多次失败应正确累加 error 计数。"""
        for _ in range(2):
            metrics.record_call(
                model="gpt-4o",
                tokens={"prompt_tokens": 50},
                latency_ms=100.0,
                success=False,
            )
        summary = metrics.get_summary()
        # 验证 error 计数累加
        assert 'llm_call_total{model="gpt-4o",status="error"} 2.0' in summary


# ---------------------------------------------------------------------------
# TestLLMMetricsTokens
# ---------------------------------------------------------------------------


class TestLLMMetricsTokens:
    """测试 tokens 字典的处理。"""

    def test_missing_tokens_default_to_zero(self, metrics: LLMMetrics) -> None:
        """tokens 字典缺失的键应默认为 0，不引发异常。"""
        metrics.record_call(
            model="gpt-4o",
            tokens={},  # 空字典
            latency_ms=100.0,
            success=True,
        )
        summary = metrics.get_summary()
        # 所有 token 类型应记录为 0
        assert 'model="gpt-4o",type="prompt"} 0.0' in summary
        assert 'model="gpt-4o",type="completion"} 0.0' in summary
        assert 'model="gpt-4o",type="total"} 0.0' in summary

    def test_partial_tokens_dict(self, metrics: LLMMetrics) -> None:
        """仅提供部分 token 键时，缺失键应默认为 0。"""
        metrics.record_call(
            model="gpt-4o",
            tokens={"prompt_tokens": 300},  # 仅 prompt_tokens
            latency_ms=500.0,
            success=True,
        )
        summary = metrics.get_summary()
        assert 'model="gpt-4o",type="prompt"} 300.0' in summary
        assert 'model="gpt-4o",type="completion"} 0.0' in summary
        assert 'model="gpt-4o",type="total"} 0.0' in summary


# ---------------------------------------------------------------------------
# TestLLMMetricsSummary
# ---------------------------------------------------------------------------


class TestLLMMetricsSummary:
    """测试 get_summary() 方法。"""

    def test_get_summary_returns_string(self, metrics: LLMMetrics) -> None:
        """get_summary() 应返回字符串。"""
        summary = metrics.get_summary()
        assert isinstance(summary, str)

    def test_get_summary_prometheus_format(self, metrics: LLMMetrics) -> None:
        """get_summary() 返回 Prometheus 文本格式。"""
        metrics.record_call(
            model="gpt-4o",
            tokens={"prompt_tokens": 10},
            latency_ms=100.0,
            success=True,
        )
        summary = metrics.get_summary()
        # Prometheus 格式特征：包含 HELP 和 TYPE 行
        assert "# HELP" in summary
        assert "# TYPE" in summary

    def test_empty_summary_contains_all_metrics(self, metrics: LLMMetrics) -> None:
        """未记录任何调用时，summary 仍应列出所有已注册的指标名称。"""
        summary = metrics.get_summary()
        assert "llm_call_total" in summary
        assert "llm_token_usage_total" in summary
        assert "llm_latency_seconds" in summary


# ---------------------------------------------------------------------------
# TestLLMMetricsReset
# ---------------------------------------------------------------------------


class TestLLMMetricsReset:
    """测试 reset() 方法。"""

    def test_reset_clears_all_metrics(self, metrics: LLMMetrics) -> None:
        """reset() 应将所有指标清零。"""
        metrics.record_call(
            model="gpt-4o",
            tokens={"prompt_tokens": 100, "total_tokens": 100},
            latency_ms=1000.0,
            success=True,
        )
        metrics.record_call(
            model="gpt-4o",
            tokens={},
            latency_ms=0.0,
            success=False,
        )

        # 重置前应包含数据
        summary_before = metrics.get_summary()
        assert "100.0" in summary_before

        metrics.reset()

        # 重置后不应包含之前的数据
        summary_after = metrics.get_summary()
        assert "100.0" not in summary_after

    def test_reset_then_record(self, metrics: LLMMetrics) -> None:
        """reset() 后再次 record_call 应正常工作。"""
        metrics.record_call(
            model="gpt-4o",
            tokens={"prompt_tokens": 100, "total_tokens": 100},
            latency_ms=100.0,
            success=True,
        )
        metrics.reset()
        metrics.record_call(
            model="gpt-4o",
            tokens={"prompt_tokens": 50, "total_tokens": 50},
            latency_ms=200.0,
            success=True,
        )
        summary = metrics.get_summary()
        assert 'model="gpt-4o",type="prompt"} 50.0' in summary
        # 应不包含重置前的 100
        assert 'model="gpt-4o",type="prompt"} 100.0' not in summary
