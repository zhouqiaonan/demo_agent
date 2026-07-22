"""ContextManager —— 基于 token/消息数量的滑动窗口消息截断。"""

from __future__ import annotations

import tiktoken


class ContextManager:
    """将消息列表截断到 token 或消息数量限制以内。

    当 ``preserve_system=True`` 时，system 消息会被保留在顶部且不计入限制。
    当 ``max_messages`` 和 ``max_tokens`` 同时设置时，``max_tokens`` 优先。
    """

    def __init__(
        self,
        max_messages: int | None = None,
        max_tokens: int | None = None,
        preserve_system: bool = True,
        model: str = "gpt-4o",
    ) -> None:
        self.max_messages = max_messages
        self.max_tokens = max_tokens
        self.preserve_system = preserve_system
        self.model = model

    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------

    def apply(self, messages: list[dict]) -> tuple[list[dict], int]:
        """返回一个 ``(截断后的消息, 保留的消息条数)`` 元组。

        *截断后的消息* 是满足配置限制的消息列表。
        *保留的消息条数* 是从尾部开始保留的 **非 system** 消息的数量。
        """
        if not messages:
            return [], 0

        # 1. 将 system 消息与其他消息分离
        if self.preserve_system:
            system_msgs = [m for m in messages if m.get("role") == "system"]
            non_system = [m for m in messages if m.get("role") != "system"]
        else:
            system_msgs = []
            non_system = list(messages)

        # 2. 对非 system 消息执行截断（从尾部开始保留）
        if self.max_tokens is not None:
            truncated = self._truncate_by_tokens(non_system)
        elif self.max_messages is not None:
            truncated = self._truncate_by_count(non_system)
        else:
            truncated = non_system

        kept_count = len(truncated)

        # 3. 将 system 消息放回列表顶部
        return system_msgs + truncated, kept_count

    @staticmethod
    def count_tokens(messages: list[dict], model: str = "gpt-4o") -> int:
        """使用 tiktoken 计算 *messages* 的总 token 数。"""
        from session._token_utils import count_messages

        return count_messages(messages, model)

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------

    def _truncate_by_count(self, messages: list[dict]) -> list[dict]:
        """从尾部开始，最多保留 *max_messages* 条消息。"""
        if self.max_messages is None:
            return messages
        if len(messages) <= self.max_messages:
            return messages
        return messages[-self.max_messages :]

    def _truncate_by_tokens(self, messages: list[dict]) -> list[dict]:
        """从尾部开始，保留尽可能多的消息使其 token 总数不超过 *max_tokens*。"""
        if self.max_tokens is None:
            return messages

        # 在本方法内缓存编码器，避免每条消息都重复创建
        try:
            enc = tiktoken.encoding_for_model(self.model)
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")

        def _msg_tokens(msg: dict) -> int:
            total = 4
            for v in msg.values():
                if isinstance(v, str):
                    total += len(enc.encode(v))
            return total

        # 从尾部向前遍历，累积消息直到超出 token 预算
        kept: list[dict] = []
        running_tokens = 0

        for msg in reversed(messages):
            msg_tokens = _msg_tokens(msg)
            if running_tokens + msg_tokens > self.max_tokens:
                break
            kept.append(msg)
            running_tokens += msg_tokens

        # 恢复原始顺序
        kept.reverse()
        return kept
