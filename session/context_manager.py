"""ContextManager — Slide-window message truncation with token/message limits."""

from __future__ import annotations

import tiktoken


class ContextManager:
    """Truncates a list of messages to fit within token or message-count limits.

    System messages are preserved at the top and not counted toward limits
    when ``preserve_system=True``.  When both ``max_messages`` and ``max_tokens``
    are set, ``max_tokens`` takes priority.
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
    # Public API
    # ------------------------------------------------------------------

    def apply(self, messages: list[dict]) -> tuple[list[dict], int]:
        """Return a tuple of ``(truncated_messages, kept_count)``.

        *truncated_messages* is the list of messages that fit within the
        configured limits.  *kept_count* is the number of **non-system**
        messages that were retained (starting from the end of the input).
        """
        if not messages:
            return [], 0

        # 1. Separate system messages from the rest
        if self.preserve_system:
            system_msgs = [m for m in messages if m.get("role") == "system"]
            non_system = [m for m in messages if m.get("role") != "system"]
        else:
            system_msgs = []
            non_system = list(messages)

        # 2. Apply truncation on non-system messages (keep from the END)
        if self.max_tokens is not None:
            truncated = self._truncate_by_tokens(non_system)
        elif self.max_messages is not None:
            truncated = self._truncate_by_count(non_system)
        else:
            truncated = non_system

        kept_count = len(truncated)

        # 3. Prepend system messages back at top
        return system_msgs + truncated, kept_count

    @staticmethod
    def count_tokens(messages: list[dict], model: str = "gpt-4o") -> int:
        """Count total tokens consumed by *messages* using tiktoken."""
        from session._token_utils import count_messages

        return count_messages(messages, model)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _truncate_by_count(self, messages: list[dict]) -> list[dict]:
        """Keep at most *max_messages* messages from the end."""
        if self.max_messages is None:
            return messages
        if len(messages) <= self.max_messages:
            return messages
        return messages[-self.max_messages :]

    def _truncate_by_tokens(self, messages: list[dict]) -> list[dict]:
        """Keep as many messages from the end as fit within *max_tokens*."""
        if self.max_tokens is None:
            return messages

        # Cache the encoder for this call so we don't create one per message.
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

        # Walk from the end, accumulating messages until we'd exceed the budget
        kept: list[dict] = []
        running_tokens = 0

        for msg in reversed(messages):
            msg_tokens = _msg_tokens(msg)
            if running_tokens + msg_tokens > self.max_tokens:
                break
            kept.append(msg)
            running_tokens += msg_tokens

        # Reverse back to original order
        kept.reverse()
        return kept
