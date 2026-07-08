"""工具注册模块 — 将 Python 函数转为 OpenAI 兼容的 JSON Schema 工具定义。"""

from __future__ import annotations

import inspect
import re
import typing
from collections.abc import Callable
from enum import Enum
from typing import Any, Literal, get_args, get_origin


class ToolRegistry:
    """工具注册表，管理可被大模型调用的 Python 函数。

    使用方式::

        registry = ToolRegistry()
        registry.register(my_func).register(another_func)
        tool_defs = registry.get_tool_defs()
        result = registry.execute("my_func", {"arg1": "value"})
    """

    def __init__(self) -> None:
        """初始化空注册表。"""
        self._tools: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # 注册
    # ------------------------------------------------------------------

    def register(
        self,
        func: Callable[..., Any],
        name: str | None = None,
        description: str | None = None,
    ) -> "ToolRegistry":
        """注册一个 Python 函数为可调用工具。

        Args:
            func: 要注册的 Python 可调用对象。
            name: 工具名称，默认为 ``func.__name__``。
            description: 工具描述，默认为函数的 docstring 首行。

        Returns:
            self，支持链式调用。
        """
        tool_name = name or func.__name__
        tool_desc = description or self._extract_short_description(func)
        schema = self._generate_schema(func)

        self._tools[tool_name] = {
            "func": func,
            "schema": schema,
            "description": tool_desc,
        }
        return self

    def register_from_class(
        self,
        instance: object,
        method_names: list[str] | None = None,
    ) -> "ToolRegistry":
        """从类实例注册其公开方法。

        Args:
            instance: 类实例。
            method_names: 要注册的方法名列表；为 None 时注册所有不以 ``_`` 开头的公开方法。

        Returns:
            self，支持链式调用。
        """
        if method_names is None:
            method_names = [
                name
                for name, member in inspect.getmembers(instance, inspect.ismethod)
                if not name.startswith("_")
            ]

        for name in method_names:
            func = getattr(instance, name)
            self.register(func, name=name)

        return self

    # ------------------------------------------------------------------
    # 查询 & 执行
    # ------------------------------------------------------------------

    def get_tool_defs(self) -> list[dict[str, Any]]:
        """获取 OpenAI 兼容格式的工具定义列表。

        Returns:
            工具定义列表，可直接传入 ``chat_completion`` 的 ``tools`` 参数。
        """
        result: list[dict[str, Any]] = []
        for name, entry in self._tools.items():
            result.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": entry["description"],
                        "parameters": entry["schema"],
                    },
                }
            )
        return result

    def execute(self, name: str, arguments: dict[str, Any]) -> Any:
        """执行指定名称的工具。

        Args:
            name: 工具名称。
            arguments: 关键字参数字典。

        Returns:
            工具函数的返回值。

        Raises:
            ValueError: 工具未注册。
        """
        if name not in self._tools:
            raise ValueError(f"工具 '{name}' 未注册。已注册的工具: {list(self._tools.keys())}")

        func = self._tools[name]["func"]
        try:
            return func(**arguments)
        except Exception as e:
            return {"error": str(e)}

    # ------------------------------------------------------------------
    # Schema 生成
    # ------------------------------------------------------------------

    def _generate_schema(self, func: Callable[..., Any]) -> dict[str, Any]:
        """为函数生成 JSON Schema 参数定义。

        利用 ``inspect.signature`` 和 ``typing.get_type_hints`` 推断参数类型，
        从 Google 风格 docstring 的 ``Args:`` 段提取参数描述。

        Args:
            func: Python 函数。

        Returns:
            JSON Schema 对象 ``{"type": "object", "properties": {...}, "required": [...]}``。
        """
        sig = inspect.signature(func)
        try:
            type_hints = typing.get_type_hints(func)
        except Exception:
            type_hints = {}

        param_descriptions = self._parse_google_args(func)

        properties: dict[str, Any] = {}
        required: list[str] = []

        for param_name, param in sig.parameters.items():
            # 跳过 self / cls
            if param_name in ("self", "cls"):
                continue

            py_type = type_hints.get(param_name, str)
            prop_schema = self._type_to_json_schema(py_type)

            # 附加参数描述
            desc = param_descriptions.get(param_name, "")
            if desc:
                prop_schema["description"] = desc

            properties[param_name] = prop_schema

            # 无默认值且非可变参数 → required
            if param.default is inspect.Parameter.empty and param.kind not in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                required.append(param_name)

        schema: dict[str, Any] = {
            "type": "object",
            "properties": properties,
        }
        if required:
            schema["required"] = required

        return schema

    # ------------------------------------------------------------------
    # 类型映射
    # ------------------------------------------------------------------

    def _type_to_json_schema(self, py_type: Any) -> dict[str, Any]:
        """将 Python 类型注解映射为 JSON Schema 类型。

        支持：``str``、``int``、``float``、``bool``、``list[X]``、
        ``dict[str, X]``、``Optional[X]``、``X | None``、``Literal["a","b"]``、
        ``Enum`` 子类。

        Args:
            py_type: Python 类型或注解。

        Returns:
            JSON Schema 类型定义字典。
        """
        origin = get_origin(py_type)
        args = get_args(py_type)

        # --- Union / Optional[X] / X | None ---
        if origin is not None and self._is_union(origin):
            non_none = [a for a in args if a is not type(None)]
            if len(non_none) == 1:
                return self._type_to_json_schema(non_none[0])
            return {"type": "string"}

        # --- list[X] ---
        if origin is list:
            if args and len(args) == 1:
                return {"type": "array", "items": self._type_to_json_schema(args[0])}
            return {"type": "array", "items": {"type": "string"}}

        # --- dict[str, X] ---
        if origin is dict:
            if args and len(args) >= 2:
                return {"type": "object", "additionalProperties": self._type_to_json_schema(args[1])}
            return {"type": "object"}

        # --- Literal["a", "b"] ---
        if origin is Literal:
            # All Literal values are strings → string enum
            literal_values = [a for a in args]
            if all(isinstance(v, str) for v in literal_values):
                return {"type": "string", "enum": literal_values}
            if all(isinstance(v, int) for v in literal_values):
                return {"type": "integer", "enum": literal_values}
            return {"type": "string", "enum": [str(v) for v in literal_values]}

        # --- Enum 子类 ---
        if isinstance(py_type, type) and issubclass(py_type, Enum):
            return {"type": "string", "enum": [e.value for e in py_type]}

        # --- 基础类型 ---
        type_map: dict[type, dict[str, Any]] = {
            str: {"type": "string"},
            int: {"type": "integer"},
            float: {"type": "number"},
            bool: {"type": "boolean"},
        }
        if py_type in type_map:
            return type_map[py_type]

        # --- 兜底 ---
        return {"type": "string", "description": f"Expected {py_type}"}

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_short_description(func: Callable[..., Any]) -> str:
        """从函数 docstring 提取首行作为简短描述。"""
        if func.__doc__:
            return inspect.cleandoc(func.__doc__).split("\n")[0].strip()
        return ""

    @staticmethod
    def _parse_google_args(func: Callable[..., Any]) -> dict[str, str]:
        """从 Google 风格 docstring 的 ``Args:`` 段解析参数描述。

        支持的格式::

            Args:
                param_name: 描述文本（可以跨行）。
                param2: 另一段描述。

        Returns:
            参数名到描述文本的映射。
        """
        if not func.__doc__:
            return {}

        doc = inspect.cleandoc(func.__doc__)
        # 找到 Args: 段
        match = re.search(r"^Args:\s*$", doc, re.MULTILINE)
        if not match:
            return {}

        args_section = doc[match.end() :]
        # 解析每一行，提取 param: description
        descriptions: dict[str, str] = {}
        current_param: str | None = None
        current_desc: list[str] = []

        for line in args_section.split("\n"):
            # 匹配新的 param: 行
            param_match = re.match(r"^\s+(\w+):\s*(.*)", line)
            if param_match:
                # 保存前一个参数
                if current_param is not None:
                    descriptions[current_param] = " ".join(current_desc).strip()
                current_param = param_match.group(1)
                current_desc = [param_match.group(2)] if param_match.group(2) else []
            elif current_param is not None:
                # 续行（缩进后的文本）
                stripped = line.strip()
                if stripped and not stripped.startswith(("Returns:", "Raises:", "Yields:", "Examples:")):
                    current_desc.append(stripped)
                elif stripped.startswith(("Returns:", "Raises:", "Yields:", "Examples:")):
                    # 遇到下一个 section，停止解析
                    break

        # 保存最后一个参数
        if current_param is not None and current_desc:
            descriptions[current_param] = " ".join(current_desc).strip()

        return descriptions

    @staticmethod
    def _is_union(origin: Any) -> bool:
        """判断 origin 是否为 Union 类型（兼容 Python 3.9/3.10+）。

        Python 3.10+ 引入 ``X | Y`` 语法糖，其 origin 为 ``types.UnionType``。
        本方法同时兼容 ``typing.Union`` 和 ``types.UnionType``。
        """
        # typing.Union (Python 3.9+) — 直接判等
        if origin is typing.Union:
            return True
        # types.UnionType (Python 3.10+) — 无 __origin__，靠字符串匹配
        origin_str = str(origin)
        return "UnionType" in origin_str
