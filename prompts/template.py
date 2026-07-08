"""提示词模板模块 — 提供 PromptTemplate 数据类，支持变量替换与版本管理。"""

import re
import uuid
from dataclasses import dataclass, field

# 用于转义 \{ → { 的内部占位符
# 包含 UUID 组件使得与任何变量值的冲突在概率上不可行
_SENTINEL_LBRACE = f"\x00PT_LBRACE_{uuid.uuid4().hex[:8]}\x00"

# 匹配 {{ var_name }} 的正则（支持内部可选空白）
_VAR_PATTERN = re.compile(r"\{\{\s*([a-zA-Z_]\w*)\s*\}\}")

# 变量值最大长度，防止 LLM 提示词中的资源耗尽攻击
_MAX_VALUE_LENGTH = 4096


@dataclass(frozen=True)
class PromptTemplate:
    """支持变量替换和版本管理的提示词模板。

    使用 ``{{ variable_name }}`` 语法定义变量占位符，
    通过 ``render(**kwargs)`` 方法进行变量替换。
    采用不可变设计，``with_var()`` 返回新实例。
    """

    template_str: str
    """模板字符串，包含 ``{{ var }}`` 占位符。"""

    version: int = 1
    """模板版本号，每次 ``with_var()`` 自增。"""

    metadata: dict = field(default_factory=dict)
    """模板元数据，自由存储附加信息。"""

    _bound_vars: dict = field(default_factory=dict, repr=False)
    """内部绑定的变量字典，通过 ``with_var()`` 累积。"""

    # ------------------------------------------------------------------
    # 渲染
    # ------------------------------------------------------------------

    @staticmethod
    def _sanitize_value(value: str) -> str:
        """清理变量值以安全插入模板。

        移除空字节（可能干扰基于哨兵的转义机制），
        并截断过长的值以防止 LLM 提示词资源耗尽。

        Args:
            value: 待插入模板的原始变量值。

        Returns:
            清理后的安全字符串。
        """
        # 移除可能干扰基于哨兵的转义机制的空字节
        value = value.replace("\x00", "")
        # 截断以防止 LLM 提示词资源耗尽
        return value[:_MAX_VALUE_LENGTH]

    def render(self, **kwargs) -> str:
        """渲染模板，将变量占位符替换为实际值。

        合并 ``_bound_vars`` 与 ``kwargs``（后者优先级更高），
        替换模板中所有 ``{{ var }}`` 占位符。
        若模板中使用了未提供的变量，抛出 ``ValueError``。

        所有变量值在插入前通过 ``_sanitize_value`` 清理。

        Args:
            **kwargs: 运行时传入的变量键值对。

        Returns:
            渲染后的字符串。

        Raises:
            ValueError: 模板中存在未提供值的变量。
        """
        # 合并变量（kwargs 优先）
        all_vars: dict[str, str] = {**self._bound_vars, **kwargs}

        # 第一步：转义 \{ → 哨兵
        escaped = self.template_str.replace(r"\{", _SENTINEL_LBRACE)

        # 第二步：收集模板中使用的所有变量名
        required_vars = set(_VAR_PATTERN.findall(escaped))

        # 第三步：校验变量完整性
        missing = required_vars - set(all_vars.keys())
        if missing:
            raise ValueError(f"缺少变量: {', '.join(sorted(missing))}")

        # 第四步：替换变量（对每个值进行清理）
        result = _VAR_PATTERN.sub(
            lambda m: self._sanitize_value(str(all_vars[m.group(1)])), escaped
        )

        # 第五步：恢复转义大括号
        result = result.replace(_SENTINEL_LBRACE, "{")

        return result

    # ------------------------------------------------------------------
    # 不可变更新
    # ------------------------------------------------------------------

    def with_var(self, key: str, value: str) -> "PromptTemplate":
        """绑定一个变量，返回新实例。

        原始实例不受影响（不可变语义），version 自增 1。

        Args:
            key: 变量名。
            value: 变量值。

        Returns:
            绑定了新变量的 PromptTemplate 新实例。
        """
        new_vars = {**self._bound_vars, key: value}
        return PromptTemplate(
            template_str=self.template_str,
            version=self.version + 1,
            metadata=dict(self.metadata),  # 浅拷贝，防止共享可变状态
            _bound_vars=new_vars,
        )
