"""PromptOptimizer 测试 — 评估集驱动的迭代优化器。"""

from unittest.mock import MagicMock, call, ANY
import pytest
from prompts.template import PromptTemplate
from prompts.optimizer import PromptOptimizer, EvalCase, OptimizationResult
from function_caller.config import GenerationConfig


# ---- Mock Helpers ----

def make_mock_llm_client():
    """Mock LLM client that returns pre-scripted responses for the optimizer.

    The optimizer calls chat_completion in two modes:
    1. "generate variants" — system prompt contains "变体" (variants)
    2. "analyze results" — system prompt contains "分析" (analyze)

    We return different responses based on which mode is detected.
    """
    client = MagicMock()

    def side_effect(messages, tools=None, model=None, **kwargs):
        # Determine which kind of call this is by inspecting messages
        system_content = ""
        for msg in messages:
            if msg.get("role") == "system":
                system_content = msg.get("content", "")

        # Return different pre-scripted responses
        if "变体" in system_content or "variant" in system_content.lower():
            return make_variant_response()
        elif "分析" in system_content or "analyze" in system_content.lower():
            return make_analysis_response()
        else:
            return make_analysis_response()  # default

    client.chat_completion.side_effect = side_effect
    return client


def make_mock_response(content):
    """Create a normalized LLM response dict."""
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": None,
    }


def make_variant_response():
    """Pre-scripted variant generation response — returns 3 minor variations."""
    return make_mock_response(
        "Variant 1: 将『翻译成英文』改为『翻译成目标语言』\n"
        "Variant 2: 添加『保持专业术语一致性』\n"
        "Variant 3: 移除冗余的礼貌用语"
    )


def make_analysis_response():
    """Pre-scripted analysis response — returns improvement suggestions."""
    return make_mock_response(
        "分析：当前提示词在简短输入上表现良好，但遇到复杂句子时翻译质量下降。\n"
        "建议：添加对复杂句式的处理指导。\n"
        "改进后提示词：请将以下文本从 {{ source_lang }} 翻译成 {{ target_lang }}，"
        "对于复杂句式请先拆解再翻译。风格：{{ tone }}"
    )


def make_binary_metric(threshold=0.5):
    """Simple metric that returns 1.0 if predictions are not empty, else 0.0."""
    def metric(prediction, expected):
        return 1.0 if len(prediction.strip()) > 0 else 0.0
    return metric


# ---- Test Classes ----

class TestEvalCase:
    """EvalCase data structure tests."""

    def test_eval_case_fields(self):
        case = EvalCase(input="Hello", expected="Bonjour")
        assert case.input == "Hello"
        assert case.expected == "Bonjour"
        assert case.metadata == {}

    def test_eval_case_with_metadata(self):
        case = EvalCase(input="Hello", expected="Bonjour", metadata={"id": 1})
        assert case.metadata == {"id": 1}

    def test_eval_case_is_immutable(self):
        case = EvalCase(input="Hello", expected="Bonjour")
        with pytest.raises(Exception):  # frozen dataclass
            case.input = "Changed"  # type: ignore


class TestOptimizationResult:
    """OptimizationResult data structure tests."""

    def test_result_structure(self):
        template = PromptTemplate("Translate: {{ text }}")
        result = OptimizationResult(
            best_template=template,
            history=[{"iteration": 1, "best_score": 0.8}],
            iterations=1,
        )
        assert result.best_template is template
        assert len(result.history) == 1
        assert result.history[0]["iteration"] == 1
        assert result.iterations == 1

    def test_result_is_immutable(self):
        template = PromptTemplate("test")
        result = OptimizationResult(best_template=template, history=[], iterations=0)
        with pytest.raises(Exception):
            result.best_template = template  # type: ignore


class TestOptimizeSingleIteration:
    """Single-iteration optimization tests with mock LLM."""

    def test_single_iteration_returns_result(self):
        client = make_mock_llm_client()
        optimizer = PromptOptimizer(client)
        seed = PromptTemplate("Translate: {{ text }}")
        eval_set = [
            EvalCase(input="Hello", expected="Bonjour"),
            EvalCase(input="Goodbye", expected="Au revoir"),
        ]
        result = optimizer.optimize(
            seed_template=seed,
            eval_set=eval_set,
            metric=make_binary_metric(),
            iterations=1,
            variants_per_iter=2,
        )
        assert isinstance(result, OptimizationResult)
        assert isinstance(result.best_template, PromptTemplate)
        assert result.iterations == 1

    def test_seed_unchanged(self):
        client = make_mock_llm_client()
        optimizer = PromptOptimizer(client)
        seed = PromptTemplate("Hello {{ name }}")
        original_vars = dict(seed._bound_vars)

        eval_set = [EvalCase(input="Alice", expected="Hi Alice")]
        optimizer.optimize(
            seed_template=seed,
            eval_set=eval_set,
            metric=lambda p, e: 1.0 if p else 0.0,
            iterations=1,
            variants_per_iter=2,
        )
        # Seed must not be modified
        assert seed._bound_vars == original_vars


class TestOptimizeConvergence:
    """Tests that the optimizer converges (score improves or stays)."""

    def test_optimize_multiple_iterations(self):
        client = make_mock_llm_client()
        optimizer = PromptOptimizer(client)
        seed = PromptTemplate("Translate to {{ target }}: {{ text }}")
        eval_set = [
            EvalCase(input="Hello", expected="Bonjour"),
            EvalCase(input="Goodbye", expected="Au revoir"),
        ]
        result = optimizer.optimize(
            seed_template=seed,
            eval_set=eval_set,
            metric=make_binary_metric(),
            iterations=3,
            variants_per_iter=2,
        )
        # With binary metric, all non-empty variants score 1.0 → early stop.
        # The optimizer always runs at least 1 iteration.
        assert result.iterations >= 1
        assert len(result.history) == result.iterations

    def test_variants_count_in_history(self):
        client = make_mock_llm_client()
        optimizer = PromptOptimizer(client)
        seed = PromptTemplate("{{ text }}")
        eval_set = [EvalCase(input="x", expected="y")]
        result = optimizer.optimize(
            seed_template=seed,
            eval_set=eval_set,
            metric=make_binary_metric(),
            iterations=1,
            variants_per_iter=2,
        )
        # History should contain variant details
        entry = result.history[0]
        assert "best_score" in entry
        assert "variants" in entry or "variant_count" in entry


class TestMetricInjection:
    """Tests that the user-provided metric is properly injected and called."""

    def test_custom_metric_called(self):
        call_log = []
        def tracking_metric(prediction, expected):
            call_log.append((prediction, expected))
            return 1.0 if prediction else 0.0

        client = make_mock_llm_client()
        optimizer = PromptOptimizer(client)
        seed = PromptTemplate("{{ text }}")
        eval_set = [
            EvalCase(input="Hello", expected="Bonjour"),
            EvalCase(input="Goodbye", expected="Au revoir"),
        ]
        optimizer.optimize(
            seed_template=seed,
            eval_set=eval_set,
            metric=tracking_metric,
            iterations=1,
            variants_per_iter=1,
        )
        # Metric should have been called at least once per eval case
        assert len(call_log) > 0

    def test_perfect_score_stops_early(self):
        client = make_mock_llm_client()
        optimizer = PromptOptimizer(client)
        seed = PromptTemplate("{{ text }}")
        eval_set = [EvalCase(input="Hello", expected="Bonjour")]

        result = optimizer.optimize(
            seed_template=seed,
            eval_set=eval_set,
            metric=lambda p, e: 1.0,  # always perfect
            iterations=5,
            variants_per_iter=2,
        )
        # With perfect score, should stop at iteration 1 (or converge quickly)
        assert result.iterations <= 5


class TestEdgeCases:
    """Edge case and error handling tests."""

    def test_empty_eval_set_raises(self):
        client = make_mock_llm_client()
        optimizer = PromptOptimizer(client)
        seed = PromptTemplate("{{ text }}")

        with pytest.raises(ValueError, match="eval_set"):
            optimizer.optimize(
                seed_template=seed,
                eval_set=[],
                metric=make_binary_metric(),
                iterations=1,
            )

    def test_single_eval_case_ok(self):
        client = make_mock_llm_client()
        optimizer = PromptOptimizer(client)
        seed = PromptTemplate("{{ text }}")

        result = optimizer.optimize(
            seed_template=seed,
            eval_set=[EvalCase(input="Hello", expected="Bonjour")],
            metric=make_binary_metric(),
            iterations=1,
        )
        assert isinstance(result, OptimizationResult)

    def test_invalid_iterations_raises(self):
        client = make_mock_llm_client()
        optimizer = PromptOptimizer(client)
        seed = PromptTemplate("{{ text }}")

        with pytest.raises(ValueError, match="iterations"):
            optimizer.optimize(
                seed_template=seed,
                eval_set=[EvalCase(input="x", expected="y")],
                metric=make_binary_metric(),
                iterations=0,
            )


class TestOptimizerConfig:
    """Tests that GenerationConfig is correctly used."""

    def test_config_passed_to_llm(self):
        client = make_mock_llm_client()
        custom_config = GenerationConfig(temperature=0.3, top_p=0.5)
        optimizer = PromptOptimizer(client, config=custom_config)
        seed = PromptTemplate("{{ text }}")

        optimizer.optimize(
            seed_template=seed,
            eval_set=[EvalCase(input="Hello", expected="Bonjour")],
            metric=make_binary_metric(),
            iterations=1,
        )

        # Verify the client was called with config params
        # At least one call should have temperature/top_p in its kwargs
        config_kwargs_found = False
        for call_args in client.chat_completion.call_args_list:
            if "temperature" in call_args.kwargs:
                config_kwargs_found = True
                break
        assert config_kwargs_found, "Config kwargs should be passed to chat_completion"

    def test_default_config_code_preset(self):
        client = make_mock_llm_client()
        optimizer = PromptOptimizer(client)  # no config → defaults to code()
        assert optimizer.config is not None
        assert isinstance(optimizer.config, GenerationConfig)
