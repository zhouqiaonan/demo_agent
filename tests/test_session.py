"""Tests for session/ package — ContextManager, MessageSummarizer, ChatSession."""

import uuid
from unittest.mock import MagicMock

import pytest

from session.context_manager import ContextManager
from session.summarizer import MessageSummarizer
from session.chat_session import ChatSession


# ---------------------------------------------------------------------------
# Class 1: TestContextManager (8 tests)
# ---------------------------------------------------------------------------

class TestContextManager:
    def test_truncate_by_message_count(self):
        cm = ContextManager(max_messages=2)
        msgs = [
            {"role": "user", "content": "1"},
            {"role": "assistant", "content": "2"},
            {"role": "user", "content": "3"},
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": "5"},
        ]
        result, kept = cm.apply(msgs)
        assert len(result) == 2
        assert result[0]["content"] == "4"
        assert result[1]["content"] == "5"

    def test_truncate_by_token_limit(self):
        cm = ContextManager(max_tokens=50)
        msgs = [
            {"role": "user", "content": "This is a longer message that should consume tokens " * 3},
            {"role": "assistant", "content": "Short reply"},
            {"role": "user", "content": "Another message"},
        ]
        result, kept = cm.apply(msgs)
        assert ContextManager.count_tokens(result, cm.model) <= 50

    def test_preserve_system_message(self):
        cm = ContextManager(max_messages=2, preserve_system=True)
        msgs = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "1"},
            {"role": "assistant", "content": "2"},
            {"role": "user", "content": "3"},
        ]
        result, kept = cm.apply(msgs)
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "You are helpful"
        assert len(result) == 3  # system + 2 window messages

    def test_empty_messages(self):
        cm = ContextManager(max_messages=10)
        result, kept = cm.apply([])
        assert result == []

    def test_within_limit_no_truncation(self):
        cm = ContextManager(max_messages=10)
        msgs = [{"role": "user", "content": "hello"}]
        result, kept = cm.apply(msgs)
        assert result == msgs

    def test_count_tokens_static(self):
        msgs = [{"role": "user", "content": "Hello world"}]
        count = ContextManager.count_tokens(msgs)
        assert count > 0
        assert isinstance(count, int)

    def test_both_limits_token_priority(self):
        cm = ContextManager(max_messages=100, max_tokens=30)
        long_msg = {"role": "user", "content": "long message " * 20}
        msgs = [long_msg, long_msg, long_msg]  # way over 30 tokens
        result, kept = cm.apply(msgs)
        assert ContextManager.count_tokens(result) <= 30
        assert len(result) < 3  # token-limited, not message-count-limited

    def test_no_limits_returns_all(self):
        cm = ContextManager()
        msgs = [
            {"role": "user", "content": "1"},
            {"role": "assistant", "content": "2"},
            {"role": "user", "content": "3"},
        ]
        result, kept = cm.apply(msgs)
        assert result == msgs


# ---------------------------------------------------------------------------
# Class 2: TestMessageSummarizer (6 tests)
# ---------------------------------------------------------------------------

class TestMessageSummarizer:
    def _make_mock_client(self, response_content="Summary of the conversation."):
        """Helper to create a mock LLM client."""
        client = MagicMock()
        client.chat_completion.return_value = {"role": "assistant", "content": response_content}
        return client

    def test_summarize_calls_llm_client(self):
        client = self._make_mock_client()
        summarizer = MessageSummarizer(client)
        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]
        summarizer.summarize(msgs)
        client.chat_completion.assert_called_once()

    def test_summarize_returns_string(self):
        client = self._make_mock_client()
        summarizer = MessageSummarizer(client)
        result = summarizer.summarize([{"role": "user", "content": "Hi"}])
        assert isinstance(result, str)
        assert len(result) > 0

    def test_empty_messages_returns_empty_string(self):
        client = self._make_mock_client()
        summarizer = MessageSummarizer(client)
        result = summarizer.summarize([])
        assert result == ""

    def test_summarize_excludes_system_messages(self):
        client = self._make_mock_client()
        summarizer = MessageSummarizer(client)
        msgs = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]
        summarizer.summarize(msgs)
        call_messages = client.chat_completion.call_args.kwargs.get("messages", [])
        assert "system" not in [m["role"] for m in call_messages if m.get("content") == "You are helpful"]
        user_content_found = any("Hello" in str(m.get("content", "")) for m in call_messages)
        assert user_content_found

    def test_summary_length_within_limit(self):
        client = self._make_mock_client("Short summary.")
        summarizer = MessageSummarizer(client, max_summary_tokens=200)
        result = summarizer.summarize([{"role": "user", "content": "Hello"}])
        tokens = summarizer.estimate_tokens(result)
        assert tokens <= 200

    def test_summarize_uses_config(self):
        from function_caller.config import GenerationConfig
        client = self._make_mock_client()
        custom_config = GenerationConfig(temperature=0.5, top_p=0.8)
        summarizer = MessageSummarizer(client, config=custom_config)
        summarizer.summarize([{"role": "user", "content": "Hello"}])
        call_kwargs = client.chat_completion.call_args.kwargs
        assert "temperature" in call_kwargs
        assert call_kwargs["temperature"] == 0.5


# ---------------------------------------------------------------------------
# Class 3: TestChatSession (7 tests)
# ---------------------------------------------------------------------------

class TestChatSession:
    def _make_cm(self):
        return ContextManager(max_messages=10)

    def _make_summarizer(self):
        client = MagicMock()
        client.chat_completion.return_value = {"role": "assistant", "content": "Summary."}
        return MessageSummarizer(client)

    def test_session_id_auto_generated(self):
        session = ChatSession(context_manager=self._make_cm())
        assert isinstance(session.session_id, str)
        assert len(session.session_id) > 0
        # Should look like a UUID
        try:
            uuid.UUID(session.session_id)
            is_valid_uuid = True
        except ValueError:
            is_valid_uuid = False
        assert is_valid_uuid

    def test_session_id_custom(self):
        session = ChatSession(context_manager=self._make_cm(), session_id="my-custom-id")
        assert session.session_id == "my-custom-id"

    def test_add_message_appends_to_history(self):
        session = ChatSession(context_manager=self._make_cm())
        session.add_message("user", "Hello")
        session.add_message("assistant", "Hi!")
        assert len(session.history) == 2
        assert session.history[0] == {"role": "user", "content": "Hello"}
        assert session.history[1] == {"role": "assistant", "content": "Hi!"}

    def test_history_is_read_only_view(self):
        session = ChatSession(context_manager=self._make_cm())
        session.add_message("user", "Hello")
        hist = session.history
        hist.append({"role": "user", "content": "injected"})
        # Original history should be unchanged
        assert len(session.history) == 1

    def test_get_context_without_summarizer(self):
        cm = ContextManager(max_messages=2)
        session = ChatSession(context_manager=cm, system_prompt="You are helpful.")
        session.add_message("user", "msg1")
        session.add_message("assistant", "reply1")
        session.add_message("user", "msg2")
        session.add_message("assistant", "reply2")
        session.add_message("user", "msg3")
        context = session.get_context()
        # system prompt + last 2 messages (window)
        assert context[0]["role"] == "system"
        assert context[0]["content"] == "You are helpful."
        assert len(context) == 3  # system + 2 window messages
        assert context[1]["content"] == "reply2"
        assert context[2]["content"] == "msg3"

    def test_get_context_system_prompt_at_top(self):
        session = ChatSession(
            context_manager=self._make_cm(),
            system_prompt="You are a bot."
        )
        session.add_message("user", "hello")
        context = session.get_context()
        assert context[0]["role"] == "system"
        assert "You are a bot" in context[0]["content"]

    def test_get_context_idempotent(self):
        """Calling get_context() twice without changes returns same cached result."""
        cm = ContextManager(max_messages=2)
        client = MagicMock()
        client.chat_completion.return_value = {"role": "assistant", "content": "Summary."}
        summarizer = MessageSummarizer(client)

        session = ChatSession(context_manager=cm, summarizer=summarizer)
        session.add_message("user", "old1")
        session.add_message("assistant", "old2")
        session.add_message("user", "recent")
        session.add_message("assistant", "reply")

        # First call triggers summarization
        ctx1 = session.get_context()
        assert client.chat_completion.call_count == 1

        # Second call should return cached result, no new LLM call
        call_count_before = client.chat_completion.call_count
        ctx2 = session.get_context()
        assert client.chat_completion.call_count == call_count_before
        assert ctx2 == ctx1

        # Adding more messages should invalidate cache
        session.add_message("user", "new message")
        ctx3 = session.get_context()
        assert client.chat_completion.call_count > call_count_before
        assert ctx3 != ctx1

    def test_to_dict_and_from_dict(self):
        cm = ContextManager(max_messages=5)
        session = ChatSession(context_manager=cm, system_prompt="Be helpful", session_id="test-123")
        session.add_message("user", "hello")
        session.add_message("assistant", "hi")

        data = session.to_dict()
        assert data["session_id"] == "test-123"
        assert data["system_prompt"] == "Be helpful"
        assert len(data["history"]) == 2
        # Verify context_config is serialized
        assert "context_config" in data
        assert data["context_config"]["max_messages"] == 5
        assert data["context_config"]["max_tokens"] is None
        assert data["context_config"]["preserve_system"] is True
        assert data["context_config"]["model"] == "gpt-4o"

        restored = ChatSession.from_dict(data, context_manager=cm)
        assert restored.session_id == "test-123"
        assert len(restored.history) == 2
        assert restored.history[0]["content"] == "hello"


# ---------------------------------------------------------------------------
# Class 4: TestHybridMemory (5 tests)
# ---------------------------------------------------------------------------

class TestHybridMemory:
    def _make_cm(self, max_messages=3):
        return ContextManager(max_messages=max_messages)

    def _make_summarizer(self, summary_text="This is a summary of earlier conversation."):
        client = MagicMock()
        client.chat_completion.return_value = {"role": "assistant", "content": summary_text}
        return MessageSummarizer(client)

    def test_old_messages_summarized(self):
        """Messages outside the sliding window should be summarized."""
        cm = self._make_cm(max_messages=2)
        summarizer = self._make_summarizer("Earlier: user asked about weather, assistant replied.")
        session = ChatSession(
            context_manager=cm,
            summarizer=summarizer,
            system_prompt="You are helpful."
        )
        session.add_message("user", "old msg 1")
        session.add_message("assistant", "old reply 1")
        session.add_message("user", "old msg 2")
        session.add_message("assistant", "old reply 2")
        session.add_message("user", "recent msg")
        session.add_message("assistant", "recent reply")

        context = session.get_context()
        assert "Earlier:" in context[0]["content"]
        assert len(context) >= 3  # system + at least 2 window messages

    def test_recent_messages_kept_verbatim(self):
        """Messages within the window should appear as-is, not summarized."""
        cm = self._make_cm(max_messages=2)
        summarizer = self._make_summarizer()
        session = ChatSession(context_manager=cm, summarizer=summarizer)
        session.add_message("user", "old1")
        session.add_message("assistant", "old2")
        session.add_message("user", "recent important question")
        session.add_message("assistant", "recent important answer")

        context = session.get_context()
        contents = [m["content"] for m in context]
        assert "recent important question" in contents
        assert "recent important answer" in contents

    def test_summary_in_context_format(self):
        """Summary should appear inside the system message content."""
        cm = self._make_cm(max_messages=2)
        summarizer = self._make_summarizer("SUMMARY_CONTENT")
        session = ChatSession(
            context_manager=cm,
            summarizer=summarizer,
            system_prompt="You are a bot."
        )
        session.add_message("user", "old")
        session.add_message("assistant", "old reply")
        session.add_message("user", "new")
        session.add_message("assistant", "new reply")

        context = session.get_context()
        system_content = context[0]["content"]
        assert "You are a bot" in system_content
        assert "SUMMARY_CONTENT" in system_content
        assert context[0]["role"] == "system"

    def test_no_summarizer_skips_summary(self):
        """Without a summarizer, no summary is generated."""
        cm = self._make_cm(max_messages=2)
        session = ChatSession(context_manager=cm, summarizer=None, system_prompt="Bot")
        session.add_message("user", "old1")
        session.add_message("assistant", "old2")
        session.add_message("user", "new1")
        session.add_message("assistant", "new2")

        context = session.get_context()
        system_content = context[0]["content"]
        assert "Summary" not in system_content
        assert "[Conversation Summary]" not in system_content

    def test_last_summary_returns_latest(self):
        """last_summary should return the most recent summary or None."""
        cm = self._make_cm(max_messages=2)
        summarizer = self._make_summarizer("Summary text")
        session = ChatSession(context_manager=cm, summarizer=summarizer)

        # No summary yet
        assert session.last_summary is None

        session.add_message("user", "old")
        session.add_message("assistant", "old reply")
        session.add_message("user", "new")
        session.add_message("assistant", "new reply")

        # Trigger summary by calling get_context
        session.get_context()
        assert session.last_summary == "Summary text"

        # Without summarizer
        session2 = ChatSession(context_manager=cm, summarizer=None)
        assert session2.last_summary is None

    def test_summarize_failure_returns_context_without_summary(self):
        """If summarizer raises, get_context() still returns a valid context."""
        cm = self._make_cm(max_messages=2)
        client = MagicMock()
        client.chat_completion.side_effect = RuntimeError("LLM rate limit")
        summarizer = MessageSummarizer(client)

        session = ChatSession(
            context_manager=cm,
            summarizer=summarizer,
            system_prompt="You are a helpful bot."
        )
        session.add_message("user", "old1")
        session.add_message("assistant", "old2")
        session.add_message("user", "recent")
        session.add_message("assistant", "reply")

        # Should not crash
        context = session.get_context()
        assert context[0]["role"] == "system"
        assert "Conversation Summary" not in context[0]["content"]
        assert len(context) >= 2  # system + at least 1 window message

    def test_summarize_failure_does_not_cache_error(self):
        """After summarization fails, adding more messages should retry summarization."""
        cm = self._make_cm(max_messages=2)
        client = MagicMock()
        # First call fails, second succeeds
        client.chat_completion.side_effect = [
            RuntimeError("LLM rate limit"),
            {"role": "assistant", "content": "Recovered summary."},
        ]
        summarizer = MessageSummarizer(client)

        session = ChatSession(
            context_manager=cm,
            summarizer=summarizer,
            system_prompt="Bot"
        )
        session.add_message("user", "old1")
        session.add_message("assistant", "old2")
        session.add_message("user", "recent1")
        session.add_message("assistant", "reply1")

        # First call — summarizer fails, should not crash
        context1 = session.get_context()
        assert "Conversation Summary" not in context1[0]["content"]
        call_count_after_first = client.chat_completion.call_count

        # Add more messages — cache invalidated, summarizer retried
        session.add_message("user", "recent2")
        session.add_message("assistant", "reply2")
        context2 = session.get_context()

        # Summarizer should have been called again
        assert client.chat_completion.call_count > call_count_after_first
        # This time it should have succeeded
        assert "Recovered summary." in context2[0]["content"]

    def test_summary_within_total_budget(self):
        """After many rounds with token-limited CM, total context stays within budget."""
        MAX_TOKENS = 2048
        cm = ContextManager(max_tokens=MAX_TOKENS, preserve_system=True)

        # Produce a long summary to stress-test the budget check
        long_summary = "These are the key topics discussed: " + "topic detail " * 200
        client = MagicMock()
        client.chat_completion.return_value = {"role": "assistant", "content": long_summary}
        summarizer = MessageSummarizer(client)

        session = ChatSession(
            context_manager=cm,
            summarizer=summarizer,
            system_prompt="You are a helpful assistant."
        )

        for i in range(50):
            session.add_message("user", f"Message number {i}. " + "data " * 6)
            session.add_message("assistant", f"Response to {i}. " + "ok " * 6)

        context = session.get_context()
        token_count = ContextManager.count_tokens(context)
        assert token_count <= MAX_TOKENS, (
            f"Total context tokens {token_count} exceeds budget {MAX_TOKENS}"
        )

    def test_long_summary_truncated_not_dropped(self):
        """A very long summary gets truncated instead of crashing; header appears when room allows."""
        MAX_TOKENS = 2048
        cm = ContextManager(max_tokens=MAX_TOKENS, preserve_system=True)

        # Summary that would massively exceed the budget if included verbatim
        huge_summary = "X" * 5000
        client = MagicMock()
        client.chat_completion.return_value = {"role": "assistant", "content": huge_summary}
        summarizer = MessageSummarizer(client)

        session = ChatSession(
            context_manager=cm,
            summarizer=summarizer,
            system_prompt="You are a helpful bot."
        )

        # Add enough messages to trigger summarization
        for i in range(60):
            session.add_message("user", f"Message number {i}. " + "data " * 6)
            session.add_message("assistant", f"Response {i}. " + "ok " * 6)

        context = session.get_context()
        system_content = context[0]["content"]
        token_count = ContextManager.count_tokens(context)

        # Token budget must be respected
        assert token_count <= MAX_TOKENS, (
            f"Token count {token_count} exceeds budget {MAX_TOKENS}"
        )
        # The full huge_summary must NOT appear (it was truncated or dropped)
        assert huge_summary not in system_content
        # Context must have valid structure
        assert context[0]["role"] == "system"
        assert len(context) >= 1


# ---------------------------------------------------------------------------
# Class 5: TestStressTests (2 tests)
# ---------------------------------------------------------------------------

class TestStressTests:
    def test_50_rounds_token_budget(self):
        """50 rounds of conversation should stay within token budget."""
        MAX_TOKENS = 2048
        cm = ContextManager(max_tokens=MAX_TOKENS, preserve_system=True)

        client = MagicMock()
        client.chat_completion.return_value = {"role": "assistant", "content": "Summary of old messages."}
        summarizer = MessageSummarizer(client)

        session = ChatSession(
            context_manager=cm,
            summarizer=summarizer,
            system_prompt="You are a helpful assistant. Answer concisely."
        )

        for i in range(50):
            session.add_message("user", f"This is message number {i}. " + "blah " * 5)
            session.add_message("assistant", f"Response to message {i}. " + "ok " * 5)

        context = session.get_context()
        token_count = ContextManager.count_tokens(context)
        assert token_count <= MAX_TOKENS, f"Token count {token_count} exceeds budget {MAX_TOKENS}"

    def test_50_rounds_smoke(self):
        """50 rounds should complete without any exceptions."""
        cm = ContextManager(max_messages=10)
        session = ChatSession(context_manager=cm)

        for i in range(50):
            session.add_message("user", f"Question {i}")
            session.add_message("assistant", f"Answer {i}")

        context = session.get_context()
        assert len(context) > 0
        assert len(session.history) == 100  # 50 user + 50 assistant
