"""PromptTemplate 单元测试 — 构造、渲染、不可变性、边界场景、转义、相等性。"""

import pytest

from prompts import PromptTemplate


# ============================================================================
# 构造测试
# ============================================================================


class TestConstruction:
    """测试 PromptTemplate 的构造行为。"""

    def test_construct_with_defaults(self):
        """默认构造：version=1, metadata={}。"""
        t = PromptTemplate("Hello")
        assert t.template_str == "Hello"
        assert t.version == 1
        assert t.metadata == {}

    def test_construct_with_custom_version(self):
        """指定 version=5。"""
        t = PromptTemplate("Hello", version=5)
        assert t.version == 5

    def test_construct_with_metadata(self):
        """指定 metadata={"author": "me"}。"""
        t = PromptTemplate("Hello", metadata={"author": "me"})
        assert t.metadata == {"author": "me"}

    def test_empty_template(self):
        """空模板字符串应当合法。"""
        t = PromptTemplate("")
        assert t.template_str == ""
        result = t.render()
        assert result == ""


# ============================================================================
# 渲染测试
# ============================================================================


class TestRender:
    """测试 render() 方法的变量替换行为。"""

    def test_render_single_variable(self):
        """单个变量替换：template.render(name="World") → "Hello, World!"。"""
        t = PromptTemplate("Hello, {{ name }}!")
        result = t.render(name="World")
        assert result == "Hello, World!"

    def test_render_multiple_variables(self):
        """两个变量在一次 render 调用中同时替换。"""
        t = PromptTemplate("{{ greeting }}, {{ name }}!")
        result = t.render(greeting="Hello", name="World")
        assert result == "Hello, World!"

    def test_render_with_repeated_variable(self):
        """同一 {{ var }} 出现多次，全部替换。"""
        t = PromptTemplate("{{ x }} + {{ x }} = {{ y }}")
        result = t.render(x="1", y="2")
        assert result == "1 + 1 = 2"

    def test_render_preserves_non_template_text(self):
        """不含 {{ }} 的文本原样输出。"""
        t = PromptTemplate("Plain text without any variables.")
        result = t.render()
        assert result == "Plain text without any variables."

    def test_render_converts_values_to_str(self):
        """非字符串值应通过 str() 转换。"""
        t = PromptTemplate("Count: {{ n }}")
        result = t.render(n=42)
        assert result == "Count: 42"
        assert isinstance(result, str)


# ============================================================================
# 不可变性测试
# ============================================================================


class TestImmutability:
    """测试 with_var() 的不可变性语义。"""

    def test_with_var_returns_new_instance(self):
        """with_var 返回新实例，不是原实例。"""
        t1 = PromptTemplate("Hello, {{ name }}!")
        t2 = t1.with_var("name", "Alice")
        assert t2 is not t1

    def test_with_var_does_not_mutate_original(self):
        """with_var 不修改原始模板的 _bound_vars。"""
        t1 = PromptTemplate("Hello, {{ name }}!")
        _ = t1.with_var("name", "Alice")
        # 原始模板在缺失变量时应当抛出 ValueError
        with pytest.raises(ValueError):
            t1.render()

    def test_with_var_increments_version(self):
        """with_var 后 version 应自增 1。"""
        t1 = PromptTemplate("Hello, {{ name }}!")
        t2 = t1.with_var("name", "Alice")
        assert t2.version == t1.version + 1

    def test_chain_multiple_with_var(self):
        """链式调用 with_var 绑定多个变量。"""
        t = PromptTemplate("{{ a }} and {{ b }}")
        t = t.with_var("a", "1").with_var("b", "2")
        result = t.render()
        assert result == "1 and 2"


# ============================================================================
# 边界场景测试
# ============================================================================


class TestEdgeCases:
    """测试边界场景和异常路径。"""

    def test_missing_variable_raises_value_error(self):
        """模板中使用但 render() 未提供的变量应抛出 ValueError。"""
        t = PromptTemplate("Hello, {{ name }}!")
        with pytest.raises(ValueError, match="name"):
            t.render()

    def test_unused_kwargs_behavior(self):
        """额外 kwargs 被忽略（不抛异常）。"""
        t = PromptTemplate("Hello, {{ name }}!")
        result = t.render(name="World", extra="unused")
        assert result == "Hello, World!"

    def test_variable_with_special_chars(self):
        """变量值包含 {{, }}, 反斜杠等特殊字符，应当按字面量输出。"""
        t = PromptTemplate("Value: {{ x }}")
        result = t.render(x="a {{ b }} c")
        assert result == "Value: a {{ b }} c"

        result = t.render(x="back\\slash")
        assert result == "Value: back\\slash"

    def test_render_with_bound_vars(self):
        """with_var 绑定的变量在 render 时自动可用。"""
        t = PromptTemplate("Hello, {{ name }}!").with_var("name", "Alice")
        result = t.render()
        assert result == "Hello, Alice!"

    def test_render_kwargs_override_bound_vars(self):
        """render() 的 kwargs 优先级高于 _bound_vars。"""
        t = PromptTemplate("Hello, {{ name }}!").with_var("name", "Alice")
        result = t.render(name="Bob")
        assert result == "Hello, Bob!"


# ============================================================================
# 转义测试
# ============================================================================


class TestEscaping:
    r"""测试 \{ 转义语法。"""

    def test_escaped_braces_literal(self):
        r"""\{ 应被当作字面量 { 输出。"""
        t = PromptTemplate(r"Literal: \{this is not a variable}")
        result = t.render()
        assert result == "Literal: {this is not a variable}"

    def test_mixed_escaped_and_variable(self):
        r"""混合：\{name} 是字面量，{{ name }} 是变量。"""
        t = PromptTemplate(r"Escape: \{name}, Variable: {{ name }}")
        result = t.render(name="Alice")
        assert result == "Escape: {name}, Variable: Alice"

    def test_double_escaped_brace(self):
        r"""\\{ 即反斜杠后跟字面量大括号。"""
        t = PromptTemplate(r"Backslash: \\{brace}")
        result = t.render()
        assert result == r"Backslash: \{brace}"

    def test_escaped_brace_with_template_nearby(self):
        r"""\{ 转义与正常 {{ }} 变量并存。"""
        t = PromptTemplate(r"\{left} {{ middle }} \{right}")
        result = t.render(middle="core")
        assert result == "{left} core {right}"


# ============================================================================
# 相等性测试
# ============================================================================


class TestEquality:
    """测试 PromptTemplate 的 == 语义。"""

    def test_same_template_equal(self):
        """相同 template_str + version → == True。"""
        a = PromptTemplate("Hello", version=1)
        b = PromptTemplate("Hello", version=1)
        assert a == b

    def test_different_version_not_equal(self):
        """相同 template_str 但不同 version → == False。"""
        a = PromptTemplate("Hello", version=1)
        b = PromptTemplate("Hello", version=2)
        assert a != b

    def test_different_template_not_equal(self):
        """不同 template_str → == False。"""
        a = PromptTemplate("Hello")
        b = PromptTemplate("World")
        assert a != b

    def test_different_metadata_not_equal(self):
        """相同 template_str 但不同 metadata → == False。"""
        a = PromptTemplate("Hello", metadata={"a": "1"})
        b = PromptTemplate("Hello", metadata={"a": "2"})
        assert a != b

    def test_different_bound_vars_not_equal(self):
        """相同 template_str 但不同 _bound_vars → == False。"""
        a = PromptTemplate("Hello, {{ name }}!").with_var("name", "Alice")
        b = PromptTemplate("Hello, {{ name }}!").with_var("name", "Bob")
        assert a != b

# ============================================================================
# 预设模板测试
# ============================================================================

from prompts.presets import CodeReviewer, Translator, Summarizer, Classifier, RolePlayer


class TestCodeReviewerPreset:
    """测试 code_review 预设。"""

    def test_renders_all_variables(self):
        """渲染时替换 language, focus, style 三个变量。"""
        result = CodeReviewer.render(
            language="Go",
            focus="并发安全",
            style="简洁",
        )
        assert "Go" in result
        assert "并发安全" in result
        assert "简洁" in result

    def test_output_length_reasonable(self):
        """渲染输出应 > 200 字符（详细的审查提示）。"""
        result = CodeReviewer.render(
            language="Python",
            focus="代码质量和安全性",
            style="详细",
        )
        assert len(result) > 200

    def test_defaults_exist(self):
        """使用 render() 不传参应使用默认值渲染成功。"""
        result = CodeReviewer.render()
        assert "Python" in result
        assert "代码质量和安全性" in result
        assert "详细" in result

    def test_preset_version_is_one(self):
        """预设的 version 应为 1。"""
        assert CodeReviewer.version == 1


class TestTranslatorPreset:
    """测试 translator 预设。"""

    def test_renders_source_and_target(self):
        """渲染时替换 source_lang 和 target_lang。"""
        result = Translator.render(
            source_lang="日本語",
            target_lang="中文",
            tone="口语化",
        )
        assert "日本語" in result
        assert "中文" in result
        assert "口语化" in result

    def test_tone_parameter(self):
        """tone 参数出现在渲染输出中。"""
        result = Translator.render(
            source_lang="中文",
            target_lang="English",
            tone="轻松",
        )
        assert "轻松" in result


class TestSummarizerPreset:
    """测试 summarizer 预设。"""

    def test_format_parameter(self):
        """format 参数在输出中体现。"""
        result = Summarizer.render(
            format="段落",
            max_length="300字",
            audience="普通用户",
        )
        assert "段落" in result
        assert "300字" in result
        assert "普通用户" in result

    def test_max_length_parameter(self):
        """max_length 参数在输出中体现。"""
        result = Summarizer.render(
            format="bullet",
            max_length="50字",
            audience="管理人员",
        )
        assert "50字" in result


class TestClassifierPreset:
    """测试 classifier 预设。"""

    def test_categories_as_list(self):
        """categories 出现在输出中。"""
        result = Classifier.render(
            categories="bug, feature, question",
            input_format="json",
        )
        assert "bug, feature, question" in result

    def test_json_output_instruction(self):
        """输出应包含 JSON 格式化指引。"""
        result = Classifier.render()
        assert "json" in result.lower()
        # JSON 格式说明关键词
        keywords = ["json", "JSON"]
        assert any(kw in result for kw in keywords)


class TestRolePlayerPreset:
    """测试 roleplay 预设。"""

    def test_persona_detail_present(self):
        """persona 详情出现在输出中。"""
        result = RolePlayer.render(
            role_name="Python 专家",
            persona="拥有 10 年 Python 开发经验的资深工程师",
            scenario="代码咨询",
        )
        assert "Python 专家" in result
        assert "10 年" in result
        assert "代码咨询" in result

    def test_scenario_parameter(self):
        """scenario 参数在输出中体现。"""
        result = RolePlayer.render(
            role_name="产品经理",
            persona="善于挖掘用户需求",
            scenario="需求讨论",
        )
        assert "需求讨论" in result


class TestAllPresets:
    """测试所有预设的通用属性。"""

    def test_all_presets_are_prompt_template(self):
        """所有预设都应是 PromptTemplate 实例。"""
        for preset in [CodeReviewer, Translator, Summarizer, Classifier, RolePlayer]:
            assert isinstance(preset, PromptTemplate), (
                f"{preset} is not a PromptTemplate"
            )

    def test_all_presets_render_without_error(self):
        """所有预设使用默认值渲染应不抛异常。"""
        for preset in [CodeReviewer, Translator, Summarizer, Classifier, RolePlayer]:
            try:
                result = preset.render()
                assert isinstance(result, str)
                assert len(result) > 0
            except Exception as e:
                pytest.fail(f"Preset {preset} failed to render with defaults: {e}")
