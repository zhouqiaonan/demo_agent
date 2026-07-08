"""Unit tests for FunctionCaller."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from function_caller.caller import FunctionCaller
from function_caller.config import GenerationConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_mock_response(content: str = "", tool_calls=None, finish_reason: str = "stop") -> dict:
    """Create a normalized response dict matching BaseLLMClient.chat_completion return type."""
    return {
        "content": content,
        "role": "assistant",
        "model": "mock-model",
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        "finish_reason": finish_reason,
        "tool_calls": tool_calls or [],
    }


def make_tool_call(name: str, arguments_dict: dict) -> dict:
    """Create a tool_call entry with JSON-encoded arguments."""
    return {
        "id": f"call_{name}_1",
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments_dict),
        },
    }


def make_mock_client() -> MagicMock:
    """Create a MagicMock that implements the BaseLLMClient interface."""
    client = MagicMock()
    client.chat_completion.return_value = make_mock_response(content="default")
    return client


def make_mock_registry() -> MagicMock:
    """Create a MagicMock that implements the ToolRegistry interface."""
    registry = MagicMock()
    registry.get_tool_defs.return_value = []
    registry.execute.return_value = {"status": "ok"}
    return registry


# ---------------------------------------------------------------------------
# call() — Simple text response (no tool calls)
# ---------------------------------------------------------------------------


class TestCallSimpleResponse:
    """Tests for call() when model returns plain text without tool calls."""

    def test_returns_content_in_result_dict(self):
        client = make_mock_client()
        client.chat_completion.return_value = make_mock_response(content="Hello, World!")
        registry = make_mock_registry()

        caller = FunctionCaller(client, registry)
        result = caller.call([{"role": "user", "content": "Hi"}])

        assert result["content"] == "Hello, World!"

    def test_iterations_is_one(self):
        client = make_mock_client()
        client.chat_completion.return_value = make_mock_response(content="Hello")
        registry = make_mock_registry()

        caller = FunctionCaller(client, registry)
        result = caller.call([{"role": "user", "content": "Hi"}])

        assert result["iterations"] == 1

    def test_tool_calls_made_is_empty(self):
        client = make_mock_client()
        client.chat_completion.return_value = make_mock_response(content="Hello")
        registry = make_mock_registry()

        caller = FunctionCaller(client, registry)
        result = caller.call([{"role": "user", "content": "Hi"}])

        assert result["tool_calls_made"] == []

    def test_messages_contain_original_and_response(self):
        client = make_mock_client()
        client.chat_completion.return_value = make_mock_response(content="Hello")
        registry = make_mock_registry()

        caller = FunctionCaller(client, registry)
        result = caller.call([{"role": "user", "content": "Hi"}])

        # messages list should still contain the original user message
        assert len(result["messages"]) >= 1
        assert result["messages"][0] == {"role": "user", "content": "Hi"}


# ---------------------------------------------------------------------------
# call() — Tool calls
# ---------------------------------------------------------------------------


class TestCallSingleToolCall:
    """Tests for call() with a single tool invocation followed by text response."""

    def test_tool_executed_with_correct_arguments(self):
        client = make_mock_client()
        registry = make_mock_registry()
        registry.execute.return_value = '{"weather": "sunny", "temp": 25}'

        tool_call = make_tool_call("get_weather", {"city": "Beijing"})
        client.chat_completion.side_effect = [
            make_mock_response(content="", tool_calls=[tool_call]),
            make_mock_response(content="Beijing is sunny, 25°C"),
        ]

        caller = FunctionCaller(client, registry)
        caller.call([{"role": "user", "content": "What's the weather?"}])

        registry.execute.assert_called_once_with("get_weather", {"city": "Beijing"})

    def test_tool_result_fed_back_to_model(self):
        client = make_mock_client()
        registry = make_mock_registry()
        registry.execute.return_value = {"weather": "sunny"}

        tool_call = make_tool_call("get_weather", {"city": "Beijing"})
        client.chat_completion.side_effect = [
            make_mock_response(content="", tool_calls=[tool_call]),
            make_mock_response(content="Beijing is sunny"),
        ]

        caller = FunctionCaller(client, registry)
        caller.call([{"role": "user", "content": "Weather?"}])

        # Second call's messages should contain the tool result
        second_call_messages = client.chat_completion.call_args_list[1].kwargs["messages"]
        tool_messages = [msg for msg in second_call_messages if msg["role"] == "tool"]
        assert len(tool_messages) == 1
        assert json.loads(tool_messages[0]["content"]) == {"weather": "sunny"}

    def test_final_content_from_model(self):
        client = make_mock_client()
        registry = make_mock_registry()

        tool_call = make_tool_call("get_weather", {"city": "Beijing"})
        client.chat_completion.side_effect = [
            make_mock_response(content="", tool_calls=[tool_call]),
            make_mock_response(content="Beijing is sunny, 25°C"),
        ]

        caller = FunctionCaller(client, registry)
        result = caller.call([{"role": "user", "content": "Weather?"}])

        assert result["content"] == "Beijing is sunny, 25°C"

    def test_iterations_is_two(self):
        client = make_mock_client()
        registry = make_mock_registry()

        tool_call = make_tool_call("get_weather", {"city": "Beijing"})
        client.chat_completion.side_effect = [
            make_mock_response(content="", tool_calls=[tool_call]),
            make_mock_response(content="Sunny"),
        ]

        caller = FunctionCaller(client, registry)
        result = caller.call([{"role": "user", "content": "Weather?"}])

        assert result["iterations"] == 2

    def test_tool_calls_made_recorded(self):
        client = make_mock_client()
        registry = make_mock_registry()
        registry.execute.return_value = {"weather": "sunny"}

        tool_call = make_tool_call("get_weather", {"city": "Beijing"})
        client.chat_completion.side_effect = [
            make_mock_response(content="", tool_calls=[tool_call]),
            make_mock_response(content="Sunny"),
        ]

        caller = FunctionCaller(client, registry)
        result = caller.call([{"role": "user", "content": "Weather?"}])

        assert result["tool_calls_made"] == [
            {
                "name": "get_weather",
                "arguments": {"city": "Beijing"},
                "result": {"weather": "sunny"},
            }
        ]


class TestCallMultipleIterations:
    """Tests for call() requiring multiple tool-calling rounds."""

    def test_three_iterations(self):
        client = make_mock_client()
        registry = make_mock_registry()

        tool_1 = make_tool_call("step_one", {"x": 1})
        tool_2 = make_tool_call("step_two", {"y": 2})
        client.chat_completion.side_effect = [
            make_mock_response(content="", tool_calls=[tool_1]),
            make_mock_response(content="", tool_calls=[tool_2]),
            make_mock_response(content="All done"),
        ]

        caller = FunctionCaller(client, registry)
        result = caller.call([{"role": "user", "content": "Go"}])

        assert result["iterations"] == 3
        assert result["content"] == "All done"

    def test_both_tools_executed_in_order(self):
        client = make_mock_client()
        registry = make_mock_registry()

        tool_1 = make_tool_call("step_one", {"x": 1})
        tool_2 = make_tool_call("step_two", {"y": 2})
        client.chat_completion.side_effect = [
            make_mock_response(content="", tool_calls=[tool_1]),
            make_mock_response(content="", tool_calls=[tool_2]),
            make_mock_response(content="All done"),
        ]

        caller = FunctionCaller(client, registry)
        caller.call([{"role": "user", "content": "Go"}])

        assert registry.execute.call_args_list[0] == (("step_one", {"x": 1}),)
        assert registry.execute.call_args_list[1] == (("step_two", {"y": 2}),)


class TestCallParallelToolCalls:
    """Tests for call() with multiple tool_calls in a single response."""

    def test_all_tools_executed(self):
        client = make_mock_client()
        registry = make_mock_registry()

        tool_1 = make_tool_call("add", {"a": 1, "b": 2})
        tool_2 = make_tool_call("multiply", {"a": 3, "b": 4})
        client.chat_completion.side_effect = [
            make_mock_response(content="", tool_calls=[tool_1, tool_2]),
            make_mock_response(content="Results: 3 and 12"),
        ]

        caller = FunctionCaller(client, registry)
        caller.call([{"role": "user", "content": "Calc"}])

        assert registry.execute.call_count == 2
        registry.execute.assert_any_call("add", {"a": 1, "b": 2})
        registry.execute.assert_any_call("multiply", {"a": 3, "b": 4})

    def test_both_results_in_tool_calls_made(self):
        client = make_mock_client()
        registry = make_mock_registry()
        registry.execute.side_effect = ["result_add", "result_mult"]

        tool_1 = make_tool_call("add", {"a": 1, "b": 2})
        tool_2 = make_tool_call("multiply", {"a": 3, "b": 4})
        client.chat_completion.side_effect = [
            make_mock_response(content="", tool_calls=[tool_1, tool_2]),
            make_mock_response(content="Done"),
        ]

        caller = FunctionCaller(client, registry)
        result = caller.call([{"role": "user", "content": "Calc"}])

        assert len(result["tool_calls_made"]) == 2
        assert result["tool_calls_made"][0]["name"] == "add"
        assert result["tool_calls_made"][1]["name"] == "multiply"

    def test_both_tool_results_fed_back(self):
        client = make_mock_client()
        registry = make_mock_registry()
        registry.execute.side_effect = [{"sum": 3}, {"product": 12}]

        tool_1 = make_tool_call("add", {"a": 1, "b": 2})
        tool_2 = make_tool_call("multiply", {"a": 3, "b": 4})
        client.chat_completion.side_effect = [
            make_mock_response(content="", tool_calls=[tool_1, tool_2]),
            make_mock_response(content="Done"),
        ]

        caller = FunctionCaller(client, registry)
        caller.call([{"role": "user", "content": "Calc"}])

        second_call_messages = client.chat_completion.call_args_list[1].kwargs["messages"]
        tool_messages = [msg for msg in second_call_messages if msg["role"] == "tool"]
        assert len(tool_messages) == 2


# ---------------------------------------------------------------------------
# call() — System prompt
# ---------------------------------------------------------------------------


class TestCallSystemPrompt:
    """Tests for call() with system_prompt parameter."""

    def test_system_message_prepended(self):
        client = make_mock_client()
        client.chat_completion.return_value = make_mock_response(content="OK")
        registry = make_mock_registry()

        caller = FunctionCaller(client, registry)
        caller.call(
            [{"role": "user", "content": "Hi"}],
            system_prompt="You are helpful.",
        )

        # First message passed to model should be the system message
        sent_messages = client.chat_completion.call_args.kwargs["messages"]
        assert sent_messages[0] == {"role": "system", "content": "You are helpful."}

    def test_user_message_after_system(self):
        client = make_mock_client()
        client.chat_completion.return_value = make_mock_response(content="OK")
        registry = make_mock_registry()

        caller = FunctionCaller(client, registry)
        caller.call(
            [{"role": "user", "content": "Hi"}],
            system_prompt="You are helpful.",
        )

        sent_messages = client.chat_completion.call_args.kwargs["messages"]
        assert sent_messages[1] == {"role": "user", "content": "Hi"}


class TestCallNoSystemPrompt:
    """Tests for call() without system_prompt."""

    def test_no_system_message_when_none(self):
        client = make_mock_client()
        client.chat_completion.return_value = make_mock_response(content="OK")
        registry = make_mock_registry()

        caller = FunctionCaller(client, registry)
        caller.call([{"role": "user", "content": "Hi"}])

        sent_messages = client.chat_completion.call_args.kwargs["messages"]
        roles = [msg["role"] for msg in sent_messages]
        assert "system" not in roles

    def test_user_message_is_first_when_no_system(self):
        client = make_mock_client()
        client.chat_completion.return_value = make_mock_response(content="OK")
        registry = make_mock_registry()

        caller = FunctionCaller(client, registry)
        caller.call([{"role": "user", "content": "Hi"}])

        sent_messages = client.chat_completion.call_args.kwargs["messages"]
        assert sent_messages[0] == {"role": "user", "content": "Hi"}


# ---------------------------------------------------------------------------
# call() — Error / edge cases
# ---------------------------------------------------------------------------


class TestCallMaxIterations:
    """Tests for call() when max_iterations is exceeded."""

    def test_runtime_error_raised_at_limit(self):
        client = make_mock_client()
        registry = make_mock_registry()

        tool_call = make_tool_call("stubborn_tool", {})
        # Always return a tool call — loop never terminates
        client.chat_completion.return_value = make_mock_response(
            content="", tool_calls=[tool_call]
        )

        caller = FunctionCaller(client, registry, max_iterations=3)

        with pytest.raises(RuntimeError, match="达到最大迭代次数"):
            caller.call([{"role": "user", "content": "Go"}])

    def test_exact_max_calls_made(self):
        client = make_mock_client()
        registry = make_mock_registry()

        tool_call = make_tool_call("stubborn_tool", {})
        client.chat_completion.return_value = make_mock_response(
            content="", tool_calls=[tool_call]
        )

        caller = FunctionCaller(client, registry, max_iterations=3)

        with pytest.raises(RuntimeError):
            caller.call([{"role": "user", "content": "Go"}])

        assert client.chat_completion.call_count == 3


class TestCallEmptyResponse:
    """Tests for call() when model returns empty response."""

    def test_runtime_error_raised(self):
        client = make_mock_client()
        client.chat_completion.return_value = make_mock_response(
            content="", tool_calls=[], finish_reason="length"
        )
        registry = make_mock_registry()

        caller = FunctionCaller(client, registry)

        with pytest.raises(RuntimeError, match="空响应"):
            caller.call([{"role": "user", "content": "Hi"}])


class TestCallToolExecutionError:
    """Tests for call() when tool execution fails."""

    def test_error_propagated_to_tool_calls_made(self):
        client = make_mock_client()
        registry = make_mock_registry()
        registry.execute.return_value = {"error": "tool failed"}

        tool_call = make_tool_call("bad_tool", {"x": 1})
        client.chat_completion.side_effect = [
            make_mock_response(content="", tool_calls=[tool_call]),
            make_mock_response(content="Handled error gracefully"),
        ]

        caller = FunctionCaller(client, registry)
        result = caller.call([{"role": "user", "content": "Try"}])

        assert result["tool_calls_made"][0]["result"] == {"error": "tool failed"}

    def test_loop_continues_after_error(self):
        client = make_mock_client()
        registry = make_mock_registry()
        registry.execute.return_value = {"error": "failed"}

        tool_1 = make_tool_call("bad_tool", {"x": 1})
        tool_2 = make_tool_call("good_tool", {"y": 2})
        client.chat_completion.side_effect = [
            make_mock_response(content="", tool_calls=[tool_1]),
            make_mock_response(content="", tool_calls=[tool_2]),
            make_mock_response(content="Recovered"),
        ]

        caller = FunctionCaller(client, registry)
        result = caller.call([{"role": "user", "content": "Try"}])

        assert result["content"] == "Recovered"
        assert result["iterations"] == 3


class TestCallJsonDecodeError:
    """Tests for call() when tool_call arguments are invalid JSON."""

    def test_defaults_to_empty_dict(self):
        client = make_mock_client()
        registry = make_mock_registry()

        # Tool call with malformed JSON arguments
        bad_tool_call = {
            "id": "call_bad_1",
            "type": "function",
            "function": {
                "name": "bad_tool",
                "arguments": "not valid json {{{",
            },
        }
        client.chat_completion.side_effect = [
            make_mock_response(content="", tool_calls=[bad_tool_call]),
            make_mock_response(content="OK"),
        ]

        caller = FunctionCaller(client, registry)
        result = caller.call([{"role": "user", "content": "Hi"}])

        registry.execute.assert_called_once_with("bad_tool", {})
        assert result["tool_calls_made"][0]["arguments"] == {}

    def test_loop_continues_after_json_error(self):
        client = make_mock_client()
        registry = make_mock_registry()

        bad_tool_call = {
            "id": "call_bad_1",
            "type": "function",
            "function": {
                "name": "bad_tool",
                "arguments": "{{{bad json",
            },
        }
        client.chat_completion.side_effect = [
            make_mock_response(content="", tool_calls=[bad_tool_call]),
            make_mock_response(content="Recovered after JSON error"),
        ]

        caller = FunctionCaller(client, registry)
        result = caller.call([{"role": "user", "content": "Hi"}])

        assert result["content"] == "Recovered after JSON error"


# ---------------------------------------------------------------------------
# call() — Result structure
# ---------------------------------------------------------------------------


class TestCallResultStructure:
    """Tests for the structure of the dict returned by call()."""

    def test_has_all_required_keys(self):
        client = make_mock_client()
        client.chat_completion.return_value = make_mock_response(content="Hello")
        registry = make_mock_registry()

        caller = FunctionCaller(client, registry)
        result = caller.call([{"role": "user", "content": "Hi"}])

        assert isinstance(result, dict)
        assert set(result.keys()) == {"content", "messages", "iterations", "tool_calls_made"}

    def test_content_is_string(self):
        client = make_mock_client()
        client.chat_completion.return_value = make_mock_response(content="Hello")
        registry = make_mock_registry()

        caller = FunctionCaller(client, registry)
        result = caller.call([{"role": "user", "content": "Hi"}])

        assert isinstance(result["content"], str)

    def test_messages_is_list_of_dicts(self):
        client = make_mock_client()
        client.chat_completion.return_value = make_mock_response(content="Hello")
        registry = make_mock_registry()

        caller = FunctionCaller(client, registry)
        result = caller.call([{"role": "user", "content": "Hi"}])

        assert isinstance(result["messages"], list)
        assert all(isinstance(msg, dict) for msg in result["messages"])

    def test_iterations_is_int(self):
        client = make_mock_client()
        client.chat_completion.return_value = make_mock_response(content="Hello")
        registry = make_mock_registry()

        caller = FunctionCaller(client, registry)
        result = caller.call([{"role": "user", "content": "Hi"}])

        assert isinstance(result["iterations"], int)

    def test_tool_calls_made_is_list(self):
        client = make_mock_client()
        client.chat_completion.return_value = make_mock_response(content="Hello")
        registry = make_mock_registry()

        caller = FunctionCaller(client, registry)
        result = caller.call([{"role": "user", "content": "Hi"}])

        assert isinstance(result["tool_calls_made"], list)


class TestCallToolCallsRecorded:
    """Tests that tool_calls_made entries have the correct structure."""

    def test_entries_have_name_arguments_result(self):
        client = make_mock_client()
        registry = make_mock_registry()
        registry.execute.return_value = {"status": "ok"}

        tool_call = make_tool_call("my_tool", {"key": "value"})
        client.chat_completion.side_effect = [
            make_mock_response(content="", tool_calls=[tool_call]),
            make_mock_response(content="Done"),
        ]

        caller = FunctionCaller(client, registry)
        result = caller.call([{"role": "user", "content": "Go"}])

        entry = result["tool_calls_made"][0]
        assert set(entry.keys()) == {"name", "arguments", "result"}
        assert entry["name"] == "my_tool"
        assert entry["arguments"] == {"key": "value"}
        assert entry["result"] == {"status": "ok"}

    def test_multiple_calls_recorded_in_order(self):
        client = make_mock_client()
        registry = make_mock_registry()
        registry.execute.side_effect = [{"step": 1}, {"step": 2}]

        tool_1 = make_tool_call("first", {"a": 1})
        tool_2 = make_tool_call("second", {"b": 2})
        client.chat_completion.side_effect = [
            make_mock_response(content="", tool_calls=[tool_1]),
            make_mock_response(content="", tool_calls=[tool_2]),
            make_mock_response(content="Done"),
        ]

        caller = FunctionCaller(client, registry)
        result = caller.call([{"role": "user", "content": "Go"}])

        assert result["tool_calls_made"][0]["name"] == "first"
        assert result["tool_calls_made"][1]["name"] == "second"


# ---------------------------------------------------------------------------
# call() — Custom config
# ---------------------------------------------------------------------------


class TestCallCustomConfig:
    """Tests for call() with a custom GenerationConfig."""

    def test_custom_config_kwargs_passed_to_client(self):
        client = make_mock_client()
        client.chat_completion.return_value = make_mock_response(content="OK")
        registry = make_mock_registry()

        custom_config = GenerationConfig(temperature=0.5, top_p=0.9, max_tokens=100)
        caller = FunctionCaller(client, registry, config=custom_config)
        caller.call([{"role": "user", "content": "Hi"}])

        kwargs = client.chat_completion.call_args.kwargs
        assert kwargs.get("temperature") == 0.5
        assert kwargs.get("top_p") == 0.9
        assert kwargs.get("max_tokens") == 100

    def test_default_config_is_chat_preset(self):
        client = make_mock_client()
        client.chat_completion.return_value = make_mock_response(content="OK")
        registry = make_mock_registry()

        caller = FunctionCaller(client, registry)  # no config → defaults to chat()
        caller.call([{"role": "user", "content": "Hi"}])

        kwargs = client.chat_completion.call_args.kwargs
        assert kwargs.get("temperature") == 0.7
        assert kwargs.get("top_p") == 1.0


# ---------------------------------------------------------------------------
# call() — Custom max_iterations
# ---------------------------------------------------------------------------


class TestCallCustomMaxIterations:
    """Tests for call() with custom max_iterations."""

    def test_respects_custom_max_iterations(self):
        client = make_mock_client()
        registry = make_mock_registry()

        tool_call = make_tool_call("tool", {})
        client.chat_completion.return_value = make_mock_response(
            content="", tool_calls=[tool_call]
        )

        caller = FunctionCaller(client, registry, max_iterations=2)

        with pytest.raises(RuntimeError):
            caller.call([{"role": "user", "content": "Go"}])

        assert client.chat_completion.call_count == 2


# ---------------------------------------------------------------------------
# call_stream()
# ---------------------------------------------------------------------------


class TestCallStream:
    """Tests for call_stream()."""

    def test_raises_not_implemented_error(self):
        client = make_mock_client()
        registry = make_mock_registry()

        caller = FunctionCaller(client, registry)

        with pytest.raises(NotImplementedError, match="Streaming not yet implemented"):
            caller.call_stream([{"role": "user", "content": "Hi"}])


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


class TestConstructor:
    """Tests for FunctionCaller.__init__()."""

    def test_default_config_is_chat_preset(self):
        client = make_mock_client()
        registry = make_mock_registry()

        caller = FunctionCaller(client, registry)

        assert isinstance(caller.config, GenerationConfig)
        expected = GenerationConfig.chat()
        assert caller.config.temperature == expected.temperature
        assert caller.config.top_p == expected.top_p

    def test_custom_config_stored(self):
        client = make_mock_client()
        registry = make_mock_registry()
        custom = GenerationConfig(temperature=0.0, top_p=0.1)

        caller = FunctionCaller(client, registry, config=custom)

        assert caller.config is custom
        assert caller.config.temperature == 0.0

    def test_default_max_iterations_is_ten(self):
        client = make_mock_client()
        registry = make_mock_registry()

        caller = FunctionCaller(client, registry)

        assert caller.max_iterations == 10

    def test_custom_max_iterations_stored(self):
        client = make_mock_client()
        registry = make_mock_registry()

        caller = FunctionCaller(client, registry, max_iterations=5)

        assert caller.max_iterations == 5

    def test_client_and_registry_stored(self):
        client = make_mock_client()
        registry = make_mock_registry()

        caller = FunctionCaller(client, registry)

        assert caller.client is client
        assert caller.registry is registry
