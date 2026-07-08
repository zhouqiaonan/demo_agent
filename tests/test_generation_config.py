"""Unit tests for GenerationConfig class."""

import pytest

from function_caller.config import GenerationConfig


class TestGenerationConfig:
    """Tests for GenerationConfig dataclass — defaults, presets, serialization, repr, and equality."""

    # ------------------------------------------------------------------
    # Default values
    # ------------------------------------------------------------------

    def test_default_constructor(self):
        """Verify default constructor sets expected values."""
        config = GenerationConfig()
        assert config.temperature == 0.7
        assert config.top_p == 1.0
        assert config.frequency_penalty == 0.0
        assert config.presence_penalty == 0.0
        assert config.max_tokens is None
        assert config.stop is None

    def test_custom_constructor_override_all(self):
        """Verify every field can be overridden via the constructor."""
        config = GenerationConfig(
            temperature=0.5,
            top_p=0.9,
            frequency_penalty=0.2,
            presence_penalty=0.3,
            max_tokens=2048,
            stop=["\n", "END"],
        )
        assert config.temperature == 0.5
        assert config.top_p == 0.9
        assert config.frequency_penalty == 0.2
        assert config.presence_penalty == 0.3
        assert config.max_tokens == 2048
        assert config.stop == ["\n", "END"]

    def test_custom_constructor_partial_override(self):
        """Verify individual fields can be overridden; others keep defaults."""
        config = GenerationConfig(temperature=0.0, max_tokens=512)
        assert config.temperature == 0.0
        assert config.top_p == 1.0  # default
        assert config.frequency_penalty == 0.0  # default
        assert config.presence_penalty == 0.0  # default
        assert config.max_tokens == 512
        assert config.stop is None  # default

    # ------------------------------------------------------------------
    # Preset factories
    # ------------------------------------------------------------------

    def test_code_preset(self):
        """code() preset: deterministic low-temperature config."""
        config = GenerationConfig.code()
        assert config.temperature == 0.0
        assert config.top_p == 0.1
        assert config.frequency_penalty == 0.0
        assert config.presence_penalty == 0.0
        assert config.max_tokens is None
        assert config.stop is None

    def test_chat_preset(self):
        """chat() preset: balanced conversational config."""
        config = GenerationConfig.chat()
        assert config.temperature == 0.7
        assert config.top_p == 1.0
        assert config.frequency_penalty == 0.0
        assert config.presence_penalty == 0.0
        assert config.max_tokens is None
        assert config.stop is None

    def test_creative_preset(self):
        """creative() preset: high-temperature config with penalties."""
        config = GenerationConfig.creative()
        assert config.temperature == 1.0
        assert config.top_p == 0.95
        assert config.frequency_penalty == 0.3
        assert config.presence_penalty == 0.1
        assert config.max_tokens is None
        assert config.stop is None

    # ------------------------------------------------------------------
    # to_dict()
    # ------------------------------------------------------------------

    def test_to_dict_basic(self):
        """to_dict() always includes the 4 core float parameters."""
        config = GenerationConfig()
        d = config.to_dict()
        assert d == {
            "temperature": 0.7,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        }
        assert "max_tokens" not in d
        assert "stop" not in d

    def test_to_dict_with_max_tokens(self):
        """to_dict() includes max_tokens when set to a non-None value."""
        config = GenerationConfig(max_tokens=4096)
        d = config.to_dict()
        assert d["max_tokens"] == 4096

    def test_to_dict_with_stop(self):
        """to_dict() includes stop list when set to a non-None value."""
        config = GenerationConfig(stop=["</s>"])
        d = config.to_dict()
        assert d["stop"] == ["</s>"]

    def test_to_dict_omits_none_fields(self):
        """to_dict() omits max_tokens and stop when they are None."""
        config = GenerationConfig(max_tokens=None, stop=None)
        d = config.to_dict()
        assert "max_tokens" not in d
        assert "stop" not in d

    def test_to_dict_only_includes_set_optional_fields(self):
        """to_dict() includes both max_tokens and stop when both are set."""
        config = GenerationConfig(
            temperature=0.8,
            top_p=0.9,
            frequency_penalty=0.1,
            presence_penalty=0.2,
            max_tokens=1024,
            stop=["STOP"],
        )
        d = config.to_dict()
        assert d == {
            "temperature": 0.8,
            "top_p": 0.9,
            "frequency_penalty": 0.1,
            "presence_penalty": 0.2,
            "max_tokens": 1024,
            "stop": ["STOP"],
        }

    def test_to_dict_creative_preset(self):
        """to_dict() on creative preset includes penalties but no optional fields."""
        d = GenerationConfig.creative().to_dict()
        assert d == {
            "temperature": 1.0,
            "top_p": 0.95,
            "frequency_penalty": 0.3,
            "presence_penalty": 0.1,
        }
        assert "max_tokens" not in d
        assert "stop" not in d

    # ------------------------------------------------------------------
    # __repr__
    # ------------------------------------------------------------------

    def test_repr_code_preset(self):
        """__repr__ identifies the code preset label."""
        r = repr(GenerationConfig.code())
        assert r.startswith("GenerationConfig.code(")
        assert "temperature=0.0" in r
        assert "top_p=0.1" in r

    def test_repr_chat_preset(self):
        """__repr__ identifies the chat preset label."""
        r = repr(GenerationConfig.chat())
        assert r.startswith("GenerationConfig.chat(")
        assert "temperature=0.7" in r
        assert "top_p=1.0" in r

    def test_repr_creative_preset(self):
        """__repr__ identifies the creative preset and includes penalty fields."""
        r = repr(GenerationConfig.creative())
        assert r.startswith("GenerationConfig.creative(")
        assert "temperature=1.0" in r
        assert "top_p=0.95" in r
        assert "frequency_penalty=0.3" in r
        assert "presence_penalty=0.1" in r

    def test_repr_custom_default(self):
        """__repr__ shows 'custom' for a non-preset config (default constructor)."""
        r = repr(GenerationConfig())
        # Default config has temperature=0.7, top_p=1.0 which matches chat preset
        assert r.startswith("GenerationConfig.chat(")

    def test_repr_custom_non_preset(self):
        """__repr__ shows 'custom' for a config that doesn't match any preset."""
        config = GenerationConfig(temperature=0.5, top_p=0.8)
        r = repr(config)
        assert r.startswith("GenerationConfig.custom(")
        assert "temperature=0.5" in r
        assert "top_p=0.8" in r

    def test_repr_custom_with_optional_fields(self):
        """__repr__ includes max_tokens and stop when set on a custom config."""
        config = GenerationConfig(temperature=0.3, max_tokens=256, stop=["\n"])
        r = repr(config)
        assert r.startswith("GenerationConfig.custom(")
        assert "max_tokens=256" in r
        assert "stop=['\\n']" in r

    def test_repr_chat_defaults_match_preset(self):
        """chat() defaults match the chat preset, so repr shows chat label."""
        config = GenerationConfig(temperature=0.7, top_p=1.0)
        r = repr(config)
        assert r.startswith("GenerationConfig.chat(")

    # ------------------------------------------------------------------
    # Preset immutability / isolation
    # ------------------------------------------------------------------

    def test_presets_return_new_instances(self):
        """Each preset call returns a distinct object (not a singleton)."""
        a = GenerationConfig.code()
        b = GenerationConfig.code()
        assert a is not b
        assert a == b

    def test_modifying_preset_instance_does_not_affect_future_calls(self):
        """Mutating one preset result does not change the next call."""
        config = GenerationConfig.code()
        config.temperature = 0.5  # mutate
        fresh = GenerationConfig.code()
        assert fresh.temperature == 0.0  # unchanged

    # ------------------------------------------------------------------
    # Equality
    # ------------------------------------------------------------------

    def test_equal_configs(self):
        """Configs with identical field values are equal."""
        a = GenerationConfig(temperature=0.5, top_p=0.9)
        b = GenerationConfig(temperature=0.5, top_p=0.9)
        assert a == b

    def test_not_equal_different_temperature(self):
        """Configs differing in temperature are not equal."""
        a = GenerationConfig(temperature=0.5)
        b = GenerationConfig(temperature=0.6)
        assert a != b

    def test_not_equal_different_optional_fields(self):
        """Configs differing in max_tokens are not equal."""
        a = GenerationConfig(max_tokens=1024)
        b = GenerationConfig(max_tokens=2048)
        assert a != b

    def test_equal_with_none_and_set(self):
        """Config with stop=None is not equal to config with stop=['END']."""
        a = GenerationConfig(stop=None)
        b = GenerationConfig(stop=["END"])
        assert a != b

    def test_presets_equal_manually_built_equivalents(self):
        """A manually built config with preset values equals the preset."""
        code_config = GenerationConfig(temperature=0.0, top_p=0.1)
        assert code_config == GenerationConfig.code()

        creative_config = GenerationConfig(
            temperature=1.0,
            top_p=0.95,
            frequency_penalty=0.3,
            presence_penalty=0.1,
        )
        assert creative_config == GenerationConfig.creative()

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_to_dict_preserves_list_reference_for_stop(self):
        """to_dict() returns the same list object for stop (not a copy)."""
        stop_list = ["END", "STOP"]
        config = GenerationConfig(stop=stop_list)
        d = config.to_dict()
        assert d["stop"] is stop_list
