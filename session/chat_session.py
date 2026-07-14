"""ChatSession — Hybrid memory session combining ContextManager + MessageSummarizer."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from session.context_manager import ContextManager

if TYPE_CHECKING:
    from session.summarizer import MessageSummarizer


class ChatSession:
    """A conversation session with sliding-window truncation and optional summarization.

    Hybrid memory: recent messages are kept verbatim (ContextManager window);
    older messages outside the window are compressed via MessageSummarizer.
    """

    def __init__(
        self,
        context_manager: ContextManager,
        summarizer: MessageSummarizer | None = None,
        system_prompt: str | None = None,
        session_id: str | None = None,
    ) -> None:
        self._context_manager = context_manager
        self._summarizer = summarizer
        self._system_prompt = system_prompt
        self._session_id = session_id if session_id is not None else str(uuid.uuid4())
        self._history: list[dict] = []
        self._last_summary: str | None = None
        self._history_version = 0
        self._cached_context: list[dict] | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def history(self) -> list[dict]:
        """Return a **copy** of the full message history."""
        return list(self._history)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_message(self, role: str, content: str) -> None:
        """Append a message to the conversation history."""
        self._history.append({"role": role, "content": content})
        self._history_version += 1
        self._cached_context = None

    # ------------------------------------------------------------------
    # Context assembly (hybrid memory)
    # ------------------------------------------------------------------

    def get_context(self) -> list[dict]:
        """Assemble the current context window with optional summary.

        Result is cached until the history is modified via ``add_message``.
        """
        if self._cached_context is not None:
            return self._cached_context

        # 1. Apply context_manager to full history → window messages + kept count
        window_messages, kept_count = self._context_manager.apply(self._history)

        # 2. Identify older messages (those outside the window) using the kept count.
        #    kept_count is the number of non-system messages retained from the end.
        older_messages: list[dict] = []
        if kept_count > 0 and kept_count < len(self._history):
            older_messages = self._history[: -kept_count]

        # 3. If older messages exist AND summarizer is available → summarize
        summary_text: str | None = None
        if older_messages and self._summarizer is not None:
            try:
                summary_text = self._summarizer.summarize(older_messages)
                self._last_summary = summary_text
            except Exception:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning("Summarization failed, using window-only context", exc_info=True)
                summary_text = None

        # 4. Assemble: system_prompt + summary (if any) + window_messages
        system_content = self._system_prompt or "You are a helpful assistant."

        if summary_text:
            system_content += f"\n\n[Conversation Summary]: {summary_text}"

        context: list[dict] = [{"role": "system", "content": system_content}]

        # Only add non-system window messages (context_manager may have included system)
        for msg in window_messages:
            if msg.get("role") != "system":
                context.append(dict(msg))

        self._cached_context = context

        # 5. Post-hoc budget check: summary is appended AFTER ContextManager
        #    already enforced max_tokens on the window, so total may exceed budget.
        if self._context_manager.max_tokens is not None and summary_text:
            total = ContextManager.count_tokens(context, self._context_manager.model)
            if total > self._context_manager.max_tokens:
                base_prompt = self._system_prompt or "You are a helpful assistant."
                header = "\n\n[Conversation Summary]: "
                model = self._context_manager.model

                # Count tokens for window-only messages (no system, no summary).
                window_only = ContextManager.count_tokens(
                    [dict(m) for m in window_messages if m.get("role") != "system"], model
                )

                # How much budget remains for the system message content?
                available_for_system = self._context_manager.max_tokens - window_only
                if available_for_system <= 0:
                    # Window already fills the budget — use minimal system prompt
                    context = [{"role": "system", "content": base_prompt}] + [
                        dict(m) for m in window_messages if m.get("role") != "system"
                    ]
                else:
                    # Build a mock system message with header to measure exact token cost
                    header_system_tokens = ContextManager.count_tokens(
                        [{"role": "system", "content": base_prompt + header}], model
                    )
                    if header_system_tokens <= available_for_system:
                        available_for_summary = available_for_system - header_system_tokens
                        truncated = self._truncate_summary(summary_text, available_for_summary)
                        context[0]["content"] = base_prompt + header + truncated
                    else:
                        context[0]["content"] = base_prompt

                self._cached_context = context

        return context

    def _truncate_summary(self, text: str, max_tokens: int) -> str:
        """Truncate *text* so it fits within *max_tokens* tokens."""
        from session._token_utils import get_encoder

        enc = get_encoder(self._context_manager.model)
        tokens = enc.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return enc.decode(tokens[:max_tokens])

    @property
    def last_summary(self) -> str | None:
        """Return the most recent summary, or None if no summary has been generated."""
        return self._last_summary

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize session state to a plain dict (in-memory only, no I/O)."""
        return {
            "session_id": self._session_id,
            "system_prompt": self._system_prompt,
            "context_config": {
                "max_messages": self._context_manager.max_messages,
                "max_tokens": self._context_manager.max_tokens,
                "preserve_system": self._context_manager.preserve_system,
                "model": self._context_manager.model,
            },
            "history": [dict(m) for m in self._history],
            "last_summary": self._last_summary,
        }

    @classmethod
    def from_dict(
        cls,
        data: dict,
        context_manager: ContextManager,
        summarizer: MessageSummarizer | None = None,
    ) -> "ChatSession":
        """Restore a ChatSession from a dict produced by ``to_dict()``."""
        session = cls(
            context_manager=context_manager,
            summarizer=summarizer,
            system_prompt=data.get("system_prompt"),
            session_id=data.get("session_id"),
        )
        session._history = list(data.get("history", []))
        session._last_summary = data.get("last_summary")
        return session
