"""Tests for llm_engine/cost.py — CostTracker."""

import pytest

from llm_engine.cost import CostTracker


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tracker() -> CostTracker:
    """返回一个新的 CostTracker 实例。"""
    return CostTracker()


# ---------------------------------------------------------------------------
# TestCostTrackerCalculateCost
# ---------------------------------------------------------------------------


class TestCostTrackerCalculateCost:
    """测试 calculate_cost() 费用计算。"""

    def test_gpt4o_cost_calculation(self, tracker: CostTracker) -> None:
        """验证 gpt-4o 模型的费用计算：
        prompt: $2.50/1M tokens, completion: $10.00/1M tokens。
        """
        cost = tracker.calculate_cost(
            "gpt-4o",
            {"prompt_tokens": 1000, "completion_tokens": 500},
        )
        expected_prompt = 1000 / 1_000_000 * 2.50  # $0.0025
        expected_completion = 500 / 1_000_000 * 10.00  # $0.005
        expected = expected_prompt + expected_completion
        assert cost == pytest.approx(expected)
        assert cost > 0

    def test_deepseek_chat_cost_calculation(self, tracker: CostTracker) -> None:
        """验证 deepseek-chat 模型的费用计算：
        prompt: $0.14/1M tokens, completion: $0.28/1M tokens。
        """
        cost = tracker.calculate_cost(
            "deepseek-chat",
            {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000},
        )
        expected = 1.0 * 0.14 + 1.0 * 0.28  # $0.42
        assert cost == pytest.approx(expected)
        assert cost > 0

    def test_gpt4o_mini_cost_calculation(self, tracker: CostTracker) -> None:
        """验证 gpt-4o-mini 模型的费用计算：
        prompt: $0.15/1M tokens, completion: $0.60/1M tokens。
        """
        cost = tracker.calculate_cost(
            "gpt-4o-mini",
            {"prompt_tokens": 500, "completion_tokens": 200},
        )
        expected = 500 / 1_000_000 * 0.15 + 200 / 1_000_000 * 0.60
        assert cost == pytest.approx(expected)

    def test_deepseek_reasoner_cost_calculation(self, tracker: CostTracker) -> None:
        """验证 deepseek-reasoner 模型的费用计算：
        prompt: $0.55/1M tokens, completion: $2.19/1M tokens。
        """
        cost = tracker.calculate_cost(
            "deepseek-reasoner",
            {"prompt_tokens": 1000, "completion_tokens": 1000},
        )
        expected = 1000 / 1_000_000 * 0.55 + 1000 / 1_000_000 * 2.19
        assert cost == pytest.approx(expected)

    def test_zero_tokens_returns_zero(self, tracker: CostTracker) -> None:
        """Token 用量为 0 时应返回 0.0。"""
        cost = tracker.calculate_cost(
            "gpt-4o",
            {"prompt_tokens": 0, "completion_tokens": 0},
        )
        assert cost == 0.0

    def test_unknown_model_returns_zero(self, tracker: CostTracker) -> None:
        """未知模型应返回 0.0（不抛出异常）。"""
        cost = tracker.calculate_cost(
            "unknown-model",
            {"prompt_tokens": 1000, "completion_tokens": 500},
        )
        assert cost == 0.0

    def test_missing_tokens_default_to_zero(self, tracker: CostTracker) -> None:
        """usage 字典缺失键时应默认计为 0，不抛出异常。"""
        cost = tracker.calculate_cost("gpt-4o", {})
        assert cost == 0.0


# ---------------------------------------------------------------------------
# TestCostTrackerRecordCall
# ---------------------------------------------------------------------------


class TestCostTrackerRecordCall:
    """测试 record_call() 累计行为。"""

    def test_record_call_returns_cost(self, tracker: CostTracker) -> None:
        """record_call 应返回计算出的费用。"""
        cost = tracker.record_call(
            "gpt-4o",
            {"prompt_tokens": 1000, "completion_tokens": 500},
        )
        expected = 1000 / 1_000_000 * 2.50 + 500 / 1_000_000 * 10.00
        assert cost == pytest.approx(expected)

    def test_multiple_calls_accumulate_total(self, tracker: CostTracker) -> None:
        """多次 record_call 应累加总费用。"""
        cost1 = tracker.record_call("gpt-4o", {"prompt_tokens": 1000, "completion_tokens": 0})
        cost2 = tracker.record_call("gpt-4o", {"prompt_tokens": 500, "completion_tokens": 500})

        assert tracker.total_cost == pytest.approx(cost1 + cost2)
        assert tracker.call_count == 2

    def test_call_count_increments(self, tracker: CostTracker) -> None:
        """每次 record_call 应递增 call_count。"""
        assert tracker.call_count == 0

        tracker.record_call("gpt-4o", {"prompt_tokens": 10, "completion_tokens": 5})
        assert tracker.call_count == 1

        tracker.record_call("gpt-4o", {"prompt_tokens": 20, "completion_tokens": 10})
        assert tracker.call_count == 2


# ---------------------------------------------------------------------------
# TestCostTrackerCostByModel
# ---------------------------------------------------------------------------


class TestCostTrackerCostByModel:
    """测试 cost_by_model 属性。"""

    def test_cost_by_model_single(self, tracker: CostTracker) -> None:
        """单一模型的费用应正确分组。"""
        tracker.record_call("gpt-4o", {"prompt_tokens": 1000, "completion_tokens": 500})
        tracker.record_call("gpt-4o", {"prompt_tokens": 500, "completion_tokens": 200})

        by_model = tracker.cost_by_model
        assert "gpt-4o" in by_model
        assert isinstance(by_model, dict)
        assert len(by_model) == 1

    def test_cost_by_model_multiple(self, tracker: CostTracker) -> None:
        """多个模型的费用应分别分组。"""
        tracker.record_call("gpt-4o", {"prompt_tokens": 1000, "completion_tokens": 500})
        tracker.record_call("deepseek-chat", {"prompt_tokens": 2000, "completion_tokens": 1000})

        by_model = tracker.cost_by_model
        assert "gpt-4o" in by_model
        assert "deepseek-chat" in by_model
        assert len(by_model) == 2

    def test_cost_by_model_returns_copy(self, tracker: CostTracker) -> None:
        """cost_by_model 应返回防御性副本，修改不影响内部状态。"""
        tracker.record_call("gpt-4o", {"prompt_tokens": 1000, "completion_tokens": 500})

        by_model = tracker.cost_by_model
        by_model["new-model"] = 999.0

        assert "new-model" not in tracker.cost_by_model

    def test_unknown_model_accumulates_zero(self, tracker: CostTracker) -> None:
        """未知模型的 record_call 累加 0.0，但 call_count 仍递增。"""
        tracker.record_call("unknown-model", {"prompt_tokens": 1000, "completion_tokens": 500})
        assert tracker.total_cost == 0.0
        assert tracker.call_count == 1
        # 未知模型会以 0.0 出现在 cost_by_model 中
        assert tracker.cost_by_model["unknown-model"] == 0.0


# ---------------------------------------------------------------------------
# TestCostTrackerGetSummary
# ---------------------------------------------------------------------------


class TestCostTrackerGetSummary:
    """测试 get_summary() 方法。"""

    def test_get_summary_structure(self, tracker: CostTracker) -> None:
        """get_summary() 应返回包含 total_cost、call_count、by_model 的字典。"""
        summary = tracker.get_summary()
        assert isinstance(summary, dict)
        assert "total_cost" in summary
        assert "call_count" in summary
        assert "by_model" in summary

    def test_get_summary_values(self, tracker: CostTracker) -> None:
        """get_summary() 各字段值应与属性一致。"""
        cost1 = tracker.record_call("gpt-4o", {"prompt_tokens": 1000, "completion_tokens": 500})
        cost2 = tracker.record_call("deepseek-chat", {"prompt_tokens": 2000, "completion_tokens": 1000})

        summary = tracker.get_summary()
        assert summary["total_cost"] == pytest.approx(cost1 + cost2)
        assert summary["call_count"] == 2
        assert summary["by_model"]["gpt-4o"] == pytest.approx(cost1)
        assert summary["by_model"]["deepseek-chat"] == pytest.approx(cost2)


# ---------------------------------------------------------------------------
# TestCostTrackerReset
# ---------------------------------------------------------------------------


class TestCostTrackerReset:
    """测试 reset() 方法。"""

    def test_reset_clears_all(self, tracker: CostTracker) -> None:
        """reset() 应将所有累计值归零。"""
        tracker.record_call("gpt-4o", {"prompt_tokens": 1000, "completion_tokens": 500})
        tracker.record_call("deepseek-chat", {"prompt_tokens": 500, "completion_tokens": 200})

        assert tracker.total_cost > 0
        assert tracker.call_count == 2
        assert len(tracker.cost_by_model) == 2

        tracker.reset()

        assert tracker.total_cost == 0.0
        assert tracker.call_count == 0
        assert tracker.cost_by_model == {}

    def test_reset_then_record(self, tracker: CostTracker) -> None:
        """reset() 后再次 record_call 应正常工作。"""
        tracker.record_call("gpt-4o", {"prompt_tokens": 1000, "completion_tokens": 500})
        tracker.reset()
        cost = tracker.record_call("gpt-4o", {"prompt_tokens": 200, "completion_tokens": 100})

        assert tracker.total_cost == pytest.approx(cost)
        assert tracker.call_count == 1
        assert len(tracker.cost_by_model) == 1


# ---------------------------------------------------------------------------
# TestCostTrackerProperties
# ---------------------------------------------------------------------------


class TestCostTrackerProperties:
    """测试只读属性。"""

    def test_total_cost_initially_zero(self, tracker: CostTracker) -> None:
        """初始 total_cost 应为 0.0。"""
        assert tracker.total_cost == 0.0

    def test_call_count_initially_zero(self, tracker: CostTracker) -> None:
        """初始 call_count 应为 0。"""
        assert tracker.call_count == 0

    def test_cost_by_model_initially_empty(self, tracker: CostTracker) -> None:
        """初始 cost_by_model 应为空字典。"""
        assert tracker.cost_by_model == {}
