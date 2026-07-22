"""MessageSummarizer —— 通过 LLM 客户端压缩对话历史。"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from function_caller.config import GenerationConfig


_SUMMARIZE_SYSTEM_PROMPT = (
    "Summarize the following conversation in under {max_tokens} tokens. "
    "Focus on key topics, decisions, and user preferences."
)


class MessageSummarizer:
    """调用 LLM 生成对话历史的简洁摘要。"""

    def __init__(
        self,
        client: object,
        config: GenerationConfig | None = None,
        max_summary_tokens: int = 200,
    ) -> None:
        self.client = client
        self._config = config  # 在 summarize() 中延迟设置默认值
        self.max_summary_tokens = max_summary_tokens

    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------

    def summarize(self, messages: list[dict]) -> str:
        """返回 *messages* 的简洁摘要字符串。"""
        # 1. 过滤掉 system 消息
        non_system = [m for m in messages if m.get("role") != "system"]

        # 2. 空输入 → 空摘要
        if not non_system:
            return ""

        # 3. 构造 prompt
        system_prompt = _SUMMARIZE_SYSTEM_PROMPT.format(max_tokens=self.max_summary_tokens)

        # 将消息序列化为单条 user 消息用于摘要
        serialized = _serialize_messages(non_system)
        prompt_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Please summarize:\n\n{serialized}"},
        ]

        # 4. 调用 LLM 客户端
        config = self._config
        if config is None:
            from function_caller.config import GenerationConfig

            config = GenerationConfig.chat()

        response = self.client.chat_completion(
            messages=prompt_messages,
            **config.to_dict(),
        )

        # 5. 提取并返回内容
        content = response.get("content", "")
        if isinstance(content, list):
            # 部分客户端可能返回结构化内容
            return "".join(str(c) for c in content)
        return str(content)

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """估算纯文本字符串的 token 数。"""
        from session._token_utils import count_text

        return count_text(text)


# ------------------------------------------------------------------
# 内部辅助函数
# ------------------------------------------------------------------


def _serialize_messages(messages: list[dict]) -> str:
    """将消息列表转换为纯文本对话记录。"""
    lines: list[str] = []
    for m in messages:
        role = m.get("role", "unknown")
        content = m.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)
