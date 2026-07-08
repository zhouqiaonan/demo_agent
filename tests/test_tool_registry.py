"""Unit tests for ToolRegistry from function_caller.registry."""

from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

import pytest

from function_caller.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Helper functions / classes for tests
# ---------------------------------------------------------------------------


def echo(text: str) -> str:
    """Return the input text unchanged.

    Args:
        text: The input string to echo back.
    """
    return text


def greet(name: str, greeting: str = "Hello") -> str:
    """Greet someone with a customizable greeting.

    Args:
        name: The person to greet.
        greeting: The greeting word.
    """
    return f"{greeting}, {name}!"


def no_docstring(x: int) -> int:
    return x * 2


def compute(a: int, b: float, c: bool) -> bool:
    return c and (a > b)


def process_items(items: list[str]) -> int:
    return len(items)


def maybe_required(value: Optional[str]) -> str:
    return value or "default"


def also_optional(value: str | None = None) -> str:
    return value or "fallback"


def pick_color(color: Color) -> str:
    return color.value


def select_option(option: Literal["a", "b", "c"]) -> str:
    return option


def multi_arg(x: int, y: str, z: float = 1.0) -> str:
    return f"{x}-{y}-{z}"


def raises_error() -> str:
    raise RuntimeError("something went wrong")


class Color(Enum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"


class Calculator:
    """A simple calculator for testing register_from_class."""

    def add(self, a: int, b: int) -> int:
        """Add two integers.

        Args:
            a: First number.
            b: Second number.
        """
        return a + b

    def multiply(self, x: float, y: float) -> float:
        """Multiply two numbers.

        Args:
            x: First factor.
            y: Second factor.
        """
        return x * y

    def _internal(self) -> None:
        """This private method should NOT be registered."""
        pass


# ---------------------------------------------------------------------------
# Tests: register()
# ---------------------------------------------------------------------------


class TestRegister:
    """Tests for ToolRegistry.register()."""

    # -- 1. Basic registration ------------------------------------------------

    def test_basic_registration_structure(self):
        """Register a simple function and verify tool_defs structure."""
        registry = ToolRegistry()
        registry.register(echo)

        defs = registry.get_tool_defs()
        assert len(defs) == 1

        tool = defs[0]
        assert tool["type"] == "function"
        assert tool["function"]["name"] == "echo"
        assert tool["function"]["description"] == "Return the input text unchanged."

        params = tool["function"]["parameters"]
        assert params["type"] == "object"
        assert "text" in params["properties"]
        assert params["properties"]["text"]["type"] == "string"
        assert params["properties"]["text"]["description"] == "The input string to echo back."
        assert params["required"] == ["text"]

    # -- 2. Custom name -------------------------------------------------------

    def test_custom_name_override(self):
        """Register with name= to override the function name."""
        registry = ToolRegistry()
        registry.register(echo, name="my_echo")

        defs = registry.get_tool_defs()
        assert defs[0]["function"]["name"] == "my_echo"

    # -- 3. Custom description ------------------------------------------------

    def test_custom_description_override(self):
        """Register with description= to override docstring extraction."""
        registry = ToolRegistry()
        registry.register(echo, description="Custom echo desc")

        defs = registry.get_tool_defs()
        assert defs[0]["function"]["description"] == "Custom echo desc"

    # -- 4. Schema: required params -------------------------------------------

    def test_required_params_in_schema(self):
        """Params with no default value appear in the 'required' list."""
        registry = ToolRegistry()
        registry.register(greet)

        defs = registry.get_tool_defs()
        required = defs[0]["function"]["parameters"]["required"]
        assert "name" in required
        assert "greeting" not in required

    # -- 5. Schema: optional params -------------------------------------------

    def test_optional_params_not_required(self):
        """Params with default values are NOT in the 'required' list."""
        registry = ToolRegistry()
        registry.register(greet)

        defs = registry.get_tool_defs()
        required = defs[0]["function"]["parameters"]["required"]
        assert "greeting" not in required

    # -- 6. Schema: type mapping ----------------------------------------------

    def test_type_mapping_str_int_float_bool(self):
        """Python types map correctly to JSON Schema types."""
        registry = ToolRegistry()
        registry.register(compute)

        defs = registry.get_tool_defs()
        props = defs[0]["function"]["parameters"]["properties"]

        assert props["a"]["type"] == "integer"
        assert props["b"]["type"] == "number"
        assert props["c"]["type"] == "boolean"

    # -- 7. Schema: list type -------------------------------------------------

    def test_list_type_becomes_array_with_items(self):
        """list[str] becomes array type with string item schema."""
        registry = ToolRegistry()
        registry.register(process_items)

        defs = registry.get_tool_defs()
        items_schema = defs[0]["function"]["parameters"]["properties"]["items"]

        assert items_schema["type"] == "array"
        assert items_schema["items"]["type"] == "string"

    # -- 8. Schema: Optional type ---------------------------------------------

    def test_optional_type_drops_none(self):
        """Optional[str] resolves to string type (null not included)."""
        registry = ToolRegistry()
        registry.register(maybe_required)

        defs = registry.get_tool_defs()
        param_type = defs[0]["function"]["parameters"]["properties"]["value"]["type"]
        assert param_type == "string"

    # -- 9. Schema: Enum type -------------------------------------------------

    def test_enum_type_becomes_string_with_enum_values(self):
        """Python Enum becomes string type with enum constraint."""
        registry = ToolRegistry()
        registry.register(pick_color)

        defs = registry.get_tool_defs()
        color_schema = defs[0]["function"]["parameters"]["properties"]["color"]

        assert color_schema["type"] == "string"
        assert set(color_schema["enum"]) == {"red", "green", "blue"}

    # -- 10. Schema: Literal type ---------------------------------------------

    def test_literal_type_becomes_string_with_enum(self):
        """Literal["a","b","c"] becomes string type with enum values."""
        registry = ToolRegistry()
        registry.register(select_option)

        defs = registry.get_tool_defs()
        option_schema = defs[0]["function"]["parameters"]["properties"]["option"]

        assert option_schema["type"] == "string"
        assert option_schema["enum"] == ["a", "b", "c"]

    # -- 11. Schema: Google-style docstring -----------------------------------

    def test_google_docstring_extracts_param_descriptions(self):
        """Args: section in docstring populates param descriptions."""
        registry = ToolRegistry()
        registry.register(greet)

        defs = registry.get_tool_defs()
        props = defs[0]["function"]["parameters"]["properties"]

        assert props["name"]["description"] == "The person to greet."
        assert props["greeting"]["description"] == "The greeting word."

    # -- 12. Schema: no docstring ---------------------------------------------

    def test_function_without_docstring_has_empty_descriptions(self):
        """Functions without docstrings produce empty descriptions."""
        registry = ToolRegistry()
        registry.register(no_docstring)

        defs = registry.get_tool_defs()
        func_def = defs[0]["function"]

        assert func_def["description"] == ""
        params = func_def["parameters"]["properties"]
        # Description key is absent when there's no docstring to parse
        assert "description" not in params["x"]

    # -- 13. Chain registration -----------------------------------------------

    def test_register_returns_self_for_chaining(self):
        """register() returns self, enabling fluent chaining."""
        registry = ToolRegistry()

        result = registry.register(echo)
        assert result is registry

        # Chained calls
        registry.register(echo).register(greet).register(compute)
        assert len(registry.get_tool_defs()) == 3


# ---------------------------------------------------------------------------
# Tests: register_from_class()
# ---------------------------------------------------------------------------


class TestRegisterFromClass:
    """Tests for ToolRegistry.register_from_class()."""

    # -- 14. All public methods -----------------------------------------------

    def test_registers_all_public_methods_by_default(self):
        """Without method_names, all non-_ prefixed methods are registered."""
        registry = ToolRegistry()
        calc = Calculator()
        registry.register_from_class(calc)

        defs = registry.get_tool_defs()
        names = {d["function"]["name"] for d in defs}

        assert "add" in names
        assert "multiply" in names
        assert "_internal" not in names
        assert len(defs) == 2

    # -- 15. Specific methods ------------------------------------------------

    def test_registers_only_specified_methods(self):
        """With method_names list, only those methods are registered."""
        registry = ToolRegistry()
        calc = Calculator()
        registry.register_from_class(calc, method_names=["add"])

        defs = registry.get_tool_defs()
        names = {d["function"]["name"] for d in defs}

        assert names == {"add"}

    # -- 16. Skips private methods -------------------------------------------

    def test_skips_private_methods(self):
        """Methods starting with _ are excluded even with method_names=None."""
        registry = ToolRegistry()
        calc = Calculator()
        registry.register_from_class(calc)

        defs = registry.get_tool_defs()
        names = {d["function"]["name"] for d in defs}

        assert "_internal" not in names
        # Even if explicitly requested, it would still be registered
        # (that's by design — method_names overrides the filter)
        registry2 = ToolRegistry()
        registry2.register_from_class(calc, method_names=["_internal", "add"])
        defs2 = registry2.get_tool_defs()
        names2 = {d["function"]["name"] for d in defs2}
        assert "_internal" in names2


# ---------------------------------------------------------------------------
# Tests: execute()
# ---------------------------------------------------------------------------


class TestExecute:
    """Tests for ToolRegistry.execute()."""

    # -- 17. Successful execution ---------------------------------------------

    def test_successful_execution_returns_result(self):
        """execute() calls the registered function and returns its result."""
        registry = ToolRegistry()
        registry.register(echo)

        result = registry.execute("echo", {"text": "hello"})
        assert result == "hello"

    # -- 18. Unknown tool -----------------------------------------------------

    def test_unknown_tool_raises_value_error(self):
        """Executing an unregistered tool name raises ValueError."""
        registry = ToolRegistry()
        registry.register(echo)

        with pytest.raises(ValueError, match="工具 'nope' 未注册"):
            registry.execute("nope", {})

    # -- 19. Execution error --------------------------------------------------

    def test_function_raises_returns_error_dict(self):
        """When the function raises, execute() returns an error dict."""
        registry = ToolRegistry()
        registry.register(raises_error)

        result = registry.execute("raises_error", {})
        assert isinstance(result, dict)
        assert "error" in result
        assert result["error"] == "something went wrong"

    # -- 20. Multiple arguments -----------------------------------------------

    def test_execute_with_multiple_arguments(self):
        """execute() passes multiple keyword arguments correctly."""
        registry = ToolRegistry()
        registry.register(multi_arg)

        result = registry.execute("multi_arg", {"x": 42, "y": "abc", "z": 3.14})
        assert result == "42-abc-3.14"

    def test_execute_uses_default_values(self):
        """execute() can omit optional arguments and uses defaults."""
        registry = ToolRegistry()
        registry.register(greet)

        result = registry.execute("greet", {"name": "World"})
        assert result == "Hello, World!"


# ---------------------------------------------------------------------------
# Tests: get_tool_defs()
# ---------------------------------------------------------------------------


class TestGetToolDefs:
    """Tests for ToolRegistry.get_tool_defs()."""

    # -- 21. Empty registry ---------------------------------------------------

    def test_empty_registry_returns_empty_list(self):
        """A fresh registry with no tools returns []."""
        registry = ToolRegistry()
        assert registry.get_tool_defs() == []

    # -- 22. Multiple tools ---------------------------------------------------

    def test_multiple_tools_all_returned(self):
        """All registered tools appear in get_tool_defs()."""
        registry = ToolRegistry()
        registry.register(echo)
        registry.register(greet)
        registry.register(compute)

        defs = registry.get_tool_defs()
        assert len(defs) == 3

        names = {d["function"]["name"] for d in defs}
        assert names == {"echo", "greet", "compute"}

    # -- 23. Correct format ---------------------------------------------------

    def test_each_entry_has_correct_format(self):
        """Every entry has type:"function" and function:{name,description,parameters}."""
        registry = ToolRegistry()
        registry.register(echo)

        defs = registry.get_tool_defs()
        entry = defs[0]

        assert "type" in entry
        assert entry["type"] == "function"

        assert "function" in entry
        func_block = entry["function"]
        assert "name" in func_block
        assert "description" in func_block
        assert "parameters" in func_block
        assert isinstance(func_block["name"], str)
        assert isinstance(func_block["description"], str)
        assert isinstance(func_block["parameters"], dict)


# ---------------------------------------------------------------------------
# Tests: Additional Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Additional edge-case and integration-style tests."""

    def test_register_then_execute_from_class(self):
        """End-to-end: register_from_class then execute one of the methods."""
        registry = ToolRegistry()
        calc = Calculator()
        registry.register_from_class(calc, method_names=["add"])

        result = registry.execute("add", {"a": 3, "b": 4})
        assert result == 7

    def test_register_overwrites_existing_name(self):
        """Registering with the same name replaces the previous function."""
        registry = ToolRegistry()

        def first() -> str:
            return "first"

        def second() -> str:
            return "second"

        registry.register(first, name="func")
        registry.register(second, name="func")

        result = registry.execute("func", {})
        assert result == "second"
        assert len(registry.get_tool_defs()) == 1

    def test_register_from_class_returns_self(self):
        """register_from_class() returns self for chaining."""
        registry = ToolRegistry()
        result = registry.register_from_class(Calculator())
        assert result is registry

    def test_x_or_none_type_resolves_to_underlying_type(self):
        """Python 3.10+ str | None with default resolves correctly."""
        registry = ToolRegistry()
        registry.register(also_optional)

        defs = registry.get_tool_defs()
        props = defs[0]["function"]["parameters"]["properties"]
        required = defs[0]["function"]["parameters"].get("required", [])

        assert props["value"]["type"] == "string"
        assert "value" not in required
