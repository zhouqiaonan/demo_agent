"""函数调用器模块 — 实现大模型的 tool-calling 循环逻辑。"""

from __future__ import annotations

import json
from typing import Any

from llm_client.base import BaseLLMClient
from function_caller.config import GenerationConfig
from function_caller.registry import ToolRegistry


class FunctionCaller:
    """函数调用器，管理大模型的多轮 tool-calling 交互。

    维护对话上下文并循环处理模型的 tool_calls 请求，直到模型返回纯文本响应
    或达到最大迭代次数。

    使用方式::

        from llm_client import OpenAIClient
        from function_caller import FunctionCaller, ToolRegistry, GenerationConfig

        registry = ToolRegistry()
        registry.register(get_weather)

        client = OpenAIClient(api_key="...", base_url="...", model_name="gpt-4")
        caller = FunctionCaller(client, registry)

        result = caller.call(
            messages=[{"role": "user", "content": "北京今天天气怎么样？"}],
            system_prompt="你是一个天气助手。",
        )
        print(result["content"])
    """

    def __init__(
        self,
        client: BaseLLMClient,
        registry: ToolRegistry,
        config: GenerationConfig | None = None,
        max_iterations: int = 10,
    ) -> None:
        """初始化函数调用器。

        Args:
            client: 符合 ``BaseLLMClient`` 接口的大模型客户端。
            registry: 已注册工具的 ``ToolRegistry`` 实例。
            config: 生成配置，默认为 ``GenerationConfig.chat()``。
            max_iterations: 最大 tool-calling 循环次数，防止无限循环。
        """
        self.client = client
        self.registry = registry
        self.config = config if config is not None else GenerationConfig.chat()
        self.max_iterations = max_iterations

    # ------------------------------------------------------------------
    # 核心调用
    # ------------------------------------------------------------------

    def call(
        self,
        messages: list[dict[str, Any]],
        system_prompt: str | None = None,
    ) -> dict[str, Any]:
        """执行 tool-calling 循环，返回最终结果。

        流程::

            1. 构建消息列表（可选的 system prompt）
            2. 调用模型 → 检查 tool_calls
            3. 如有 tool_calls → 执行工具 → 将结果追加到 messages → 回到步骤 2
            4. 如仅有 content → 返回最终结果
            5. 如既无 content 也无 tool_calls → 抛出错误

        Args:
            messages: 初始对话消息列表。
            system_prompt: 可选的 system 角色提示词。

        Returns:
            包含最终结果和完整上下文的字典::

                {
                    "content": str,          # 模型最终文本回复
                    "messages": list[dict],   # 完整对话历史（含工具调用与结果）
                    "iterations": int,        # 循环迭代次数
                    "tool_calls_made": list[  # 已执行的工具调用记录
                        {"name": str, "arguments": dict, "result": Any}
                    ],
                }

        Raises:
            RuntimeError: 在达到最大迭代次数或收到空响应时抛出。
        """
        # 构建初始消息列表
        if system_prompt:
            full_messages: list[dict[str, Any]] = [
                {"role": "system", "content": system_prompt},
                *messages,
            ]
        else:
            full_messages = list(messages)

        all_tool_calls: list[dict[str, Any]] = []
        iteration = 0
        response: dict[str, Any] = {}

        while iteration < self.max_iterations:
            iteration += 1

            # 调用模型
            response = self.client.chat_completion(
                messages=full_messages,
                tools=self.registry.get_tool_defs(),
                **self.config.to_dict(),
            )

            tool_calls = response.get("tool_calls", []) or []
            content = response.get("content", "")

            # 情况 1: 模型请求调用工具
            if tool_calls:
                # 将模型的 tool_calls 消息追加到对话
                assistant_message: dict[str, Any] = {
                    "role": "assistant",
                    "content": content or None,
                    "tool_calls": tool_calls,
                }
                full_messages.append(assistant_message)

                # 执行每个工具调用
                for tc in tool_calls:
                    func_info = tc.get("function")
                    if not func_info:
                        # Malformed tool call — append error and skip
                        full_messages.append({
                            "role": "tool",
                            "tool_call_id": tc.get("id", ""),
                            "content": json.dumps({"error": "Invalid tool call: missing function info"}, ensure_ascii=False),
                        })
                        continue
                    func_name = func_info.get("name", "unknown")
                    try:
                        func_args = json.loads(func_info.get("arguments", "{}"))
                    except json.JSONDecodeError:
                        func_args = {}

                    func_args = self._sanitize_args(func_name, func_args)

                    result = self.registry.execute(func_name, func_args)

                    all_tool_calls.append(
                        {
                            "name": func_name,
                            "arguments": func_args,
                            "result": result,
                        }
                    )

                    # 追加工具结果消息
                    full_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.get("id", ""),
                            "content": json.dumps(result, ensure_ascii=False),
                        }
                    )

                continue  # 回到循环开头，再次调用模型

            # 情况 2: 模型返回纯文本
            if content:
                return {
                    "content": content,
                    "messages": full_messages,
                    "iterations": iteration,
                    "tool_calls_made": all_tool_calls,
                }

            # 情况 3: 既无 content 也无 tool_calls
            raise RuntimeError(
                f"模型返回了空响应（finish_reason={response.get('finish_reason', 'unknown')}）"
            )

        # 达到最大迭代次数
        raise RuntimeError(
            f"达到最大迭代次数 {self.max_iterations}。"
            f"最后一条模型角色: {response.get('role', 'unknown')}，"
            f"finish_reason: {response.get('finish_reason', 'unknown')}，"
            f"content 预览: {str(response.get('content', ''))[:200]}"
        )

    # ------------------------------------------------------------------
    # 流式调用（暂未实现）
    # ------------------------------------------------------------------

    def call_stream(
        self,
        messages: list[dict[str, Any]],
        system_prompt: str | None = None,
    ) -> Any:
        """流式 tool-calling 调用（暂未实现）。

        Args:
            messages: 初始对话消息列表。
            system_prompt: 可选的 system 角色提示词。

        Raises:
            NotImplementedError: 流式调用尚未实现。
        """
        raise NotImplementedError("Streaming not yet implemented")

    # ------------------------------------------------------------------
    # 参数清理
    # ------------------------------------------------------------------

    def _sanitize_args(self, func_name: str, raw_args: dict) -> dict:
        """验证并清理 LLM 生成的工具参数。

        只保留 schema 中定义的参数，丢弃多余参数。
        """
        if func_name not in self.registry._tools:
            return raw_args

        schema = self.registry._tools[func_name]["schema"]
        expected_props = schema.get("properties", {})

        if not expected_props:
            return raw_args

        sanitized = {}
        for key, value in raw_args.items():
            if key not in expected_props:
                continue  # 丢弃 schema 中未定义的参数

            # 基本类型转换
            prop_type = expected_props[key].get("type", "string")
            try:
                if prop_type == "integer" and not isinstance(value, int):
                    value = int(value)
                elif prop_type == "number" and not isinstance(value, (int, float)):
                    value = float(value)
                elif prop_type == "string" and not isinstance(value, str):
                    value = str(value)
                elif prop_type == "boolean" and not isinstance(value, bool):
                    value = bool(value)
            except (ValueError, TypeError):
                continue  # 类型转换失败，跳过该参数

            sanitized[key] = value

        return sanitized
