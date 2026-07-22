"""ChatSession —— 结合 ContextManager 和 MessageSummarizer 的混合记忆会话。"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from session.context_manager import ContextManager

if TYPE_CHECKING:
    from session.summarizer import MessageSummarizer


class ChatSession:
    """带滑动窗口截断和可选摘要功能的对话会话。

    混合记忆策略：最近的消息保留原文（ContextManager 窗口）；
    窗口之外的旧消息通过 MessageSummarizer 压缩为摘要。
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
    # 属性
    # ------------------------------------------------------------------

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def history(self) -> list[dict]:
        """返回完整消息历史的 **副本**。"""
        return list(self._history)

    # ------------------------------------------------------------------
    # 写操作
    # ------------------------------------------------------------------

    def add_message(self, role: str, content: str) -> None:
        """向对话历史追加一条消息。"""
        self._history.append({"role": role, "content": content})
        self._history_version += 1
        self._cached_context = None

    # ------------------------------------------------------------------
    # 上下文组装（混合记忆策略）
    # ------------------------------------------------------------------

    def get_context(self) -> list[dict]:
        """组装当前上下文窗口，可选附带摘要。

        结果会被缓存，直到通过 ``add_message`` 修改历史后才重新计算。
        """
        if self._cached_context is not None:
            return self._cached_context

        # 1. 对完整历史执行 ContextManager → 窗口消息 + 保留条数
        window_messages, kept_count = self._context_manager.apply(self._history)

        # 2. 通过保留条数识别窗口之外的旧消息。
        #    kept_count 是从尾部保留的非 system 消息数量。
        older_messages: list[dict] = []
        if kept_count > 0 and kept_count < len(self._history):
            older_messages = self._history[: -kept_count]

        # 3. 如果有旧消息 且 有 summarizer → 生成摘要
        summary_text: str | None = None
        if older_messages and self._summarizer is not None:
            try:
                summary_text = self._summarizer.summarize(older_messages)
                self._last_summary = summary_text
            except Exception:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning("摘要生成失败，使用纯窗口上下文", exc_info=True)
                summary_text = None

        # 4. 组装：system_prompt + 摘要（如有）+ 窗口消息
        system_content = self._system_prompt or "You are a helpful assistant."

        if summary_text:
            system_content += f"\n\n[Conversation Summary]: {summary_text}"

        context: list[dict] = [{"role": "system", "content": system_content}]

        # 只添加非 system 的窗口消息（ContextManager 可能已包含 system 消息）
        for msg in window_messages:
            if msg.get("role") != "system":
                context.append(dict(msg))

        self._cached_context = context

        # 5. 事后预算检查：摘要是在 ContextManager 已对窗口执行 max_tokens
        #    限制之后才追加的，因此总 token 数可能超出预算。
        if self._context_manager.max_tokens is not None and summary_text:
            total = ContextManager.count_tokens(context, self._context_manager.model)
            if total > self._context_manager.max_tokens:
                base_prompt = self._system_prompt or "You are a helpful assistant."
                header = "\n\n[Conversation Summary]: "
                model = self._context_manager.model

                # 计算纯窗口消息（不含 system、不含摘要）的 token 数。
                window_only = ContextManager.count_tokens(
                    [dict(m) for m in window_messages if m.get("role") != "system"], model
                )

                # system 消息内容还能占用多少 token 预算？
                available_for_system = self._context_manager.max_tokens - window_only
                if available_for_system <= 0:
                    # 窗口已占满预算 —— 使用最精简的 system prompt
                    context = [{"role": "system", "content": base_prompt}] + [
                        dict(m) for m in window_messages if m.get("role") != "system"
                    ]
                else:
                    # 构造带 header 的 system 消息来精确测量 token 开销
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
        """将 *text* 截断到 *max_tokens* token 以内。"""
        from session._token_utils import get_encoder

        enc = get_encoder(self._context_manager.model)
        tokens = enc.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return enc.decode(tokens[:max_tokens])

    @property
    def last_summary(self) -> str | None:
        """返回最近一次生成的摘要，无摘要时返回 None。"""
        return self._last_summary

    # ------------------------------------------------------------------
    # 序列化
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """将会话状态序列化为纯字典（纯内存操作，无 I/O）。"""
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
        """从 ``to_dict()`` 生成的字典恢复 ChatSession。"""
        session = cls(
            context_manager=context_manager,
            summarizer=summarizer,
            system_prompt=data.get("system_prompt"),
            session_id=data.get("session_id"),
        )
        session._history = list(data.get("history", []))
        session._last_summary = data.get("last_summary")
        return session
