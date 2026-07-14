"""MessageSummarizer — compress conversation history via an LLM client."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from function_caller.config import GenerationConfig


_SUMMARIZE_SYSTEM_PROMPT = (
    "Summarize the following conversation in under {max_tokens} tokens. "
    "Focus on key topics, decisions, and user preferences."
)


class MessageSummarizer:
    """Calls an LLM to produce a concise summary of conversation history."""

    def __init__(
        self,
        client: object,
        config: GenerationConfig | None = None,
        max_summary_tokens: int = 200,
    ) -> None:
        self.client = client
        self._config = config  # lazy default in summarize()
        self.max_summary_tokens = max_summary_tokens

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def summarize(self, messages: list[dict]) -> str:
        """Return a concise summary string for *messages*."""
        # 1. Filter out system messages
        non_system = [m for m in messages if m.get("role") != "system"]

        # 2. Empty input → empty summary
        if not non_system:
            return ""

        # 3. Build the prompt
        system_prompt = _SUMMARIZE_SYSTEM_PROMPT.format(max_tokens=self.max_summary_tokens)

        # Serialize the messages into a single user message for summarization
        serialized = _serialize_messages(non_system)
        prompt_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Please summarize:\n\n{serialized}"},
        ]

        # 4. Call the LLM client
        config = self._config
        if config is None:
            from function_caller.config import GenerationConfig

            config = GenerationConfig.chat()

        response = self.client.chat_completion(
            messages=prompt_messages,
            **config.to_dict(),
        )

        # 5. Extract and return content
        content = response.get("content", "")
        if isinstance(content, list):
            # Some clients return structured content
            return "".join(str(c) for c in content)
        return str(content)

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Estimate token count of a plain-text string."""
        from session._token_utils import count_text

        return count_text(text)


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _serialize_messages(messages: list[dict]) -> str:
    """Convert a list of messages into a plain-text transcript."""
    lines: list[str] = []
    for m in messages:
        role = m.get("role", "unknown")
        content = m.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)
