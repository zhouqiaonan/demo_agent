"""RAG 端到端编排模块。

本模块提供 :class:`RAGChain` 与 :class:`RAGAnswer`，是 rag 包的「顶层编排层」。
它把前面各环节已经实现好的组件——检索（:class:`~rag.retrievers.Retriever`）、
重排序（:class:`~rag.rerankers.ReRanker`）、Prompt 构建
（:class:`~rag.prompt.PromptBuilder`）以及 LLM 生成
（:class:`llm_client.BaseLLMClient`）——按既定顺序串接成一条完整的
「问题 → 答案」链路，并负责在整条链路中兜底处理未被各阶段异常覆盖的其它
错误（例如 LLM 调用失败）。

设计意图：

- **职责单一**：:class:`RAGChain` 本身不实现检索、重排序或 Prompt 拼装逻辑，
  只做「编排」与「错误兜底」，把具体动作委托给注入的各个组件，符合组合优于
  继承的原则。
- **可插拔重排序**：重排序器是可选的（``reranker`` 默认 ``None``），当未注入
  时直接跳过重排序环节，链路退化为「检索 → 构建 → 生成」的经典 RAG 三步曲。
- **答案可溯源**：最终返回的 :class:`RAGAnswer` 同时携带 LLM 生成的答案文本
  与来自 :class:`~rag.prompt.RAGPrompt` 的引用映射表，使上层在拿到答案后可以
  把其中的 ``[n]`` 编号反查回具体 chunk，实现答案溯源。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from llm_client import BaseLLMClient
from vector_store.document import SearchResult

from rag.exceptions import RAGChainError, RAGError
from rag.prompt import PromptBuilder, RAGPrompt
from rag.rerankers import ReRanker
from rag.retrievers import Retriever


@dataclass
class RAGAnswer:
    """一次 RAG 链路执行的最终产物。

    本类把「LLM 生成的最终答案文本」与「引用映射表」打包在一起，作为
    :class:`RAGChain.ask` 的返回结果。引用映射表直接来自
    :class:`~rag.prompt.RAGPrompt.references`，键为引用编号 ``n``（从 1 开始的
    整数），值为该编号对应 chunk 的溯源信息字典（``source`` / ``page`` /
    ``text``），从而保证答案中的 ``[n]`` 标注可以反查回具体原文片段。

    Attributes:
        answer: LLM 生成的最终答案文本。
        references: 引用映射表，与 :class:`~rag.prompt.RAGPrompt.references`
            完全一致：``{编号: {"source": ..., "page": ..., "text": ...}}``。
    """

    answer: str
    """LLM 生成的最终答案文本。"""

    references: dict[int, dict[str, Any]]
    """引用映射表：{编号: {"source": ..., "page": ..., "text": ...}}。"""


class RAGChain:
    """RAG 端到端编排链。

    本类把「检索 → 重排序 → Prompt 构建 → LLM 生成 → 返回引用」串接为一条
    完整的端到端流程，是整个 rag 包对外暴露的最上层入口。调用方只需注入
    :class:`~rag.retrievers.Retriever`、:class:`~rag.prompt.PromptBuilder` 与
    :class:`llm_client.BaseLLMClient` 三个必需组件（以及可选的重排序器
    :class:`~rag.rerankers.ReRanker`），随后调用 :meth:`ask` 即可得到带引用来源
    的最终答案。

    端到端数据流：

    1. **检索**：调用 ``self._retriever.retrieve(question, self._k)`` 得到初步
       候选 chunk 列表；
    2. **重排序**（可选）：若注入 ``reranker``，则调用
       ``self._reranker.rerank(question, chunks)`` 对候选做精细排序；
    3. **构建 Prompt**：调用 ``self._prompt_builder.build(question, chunks)``
       得到带 ``[n]`` 引用编号的完整 Prompt 与引用映射表；
    4. **生成**：调用 ``self._llm_client.chat_completion([...])`` 让 LLM 基于
       Prompt 生成答案；
    5. **返回**：把答案文本与引用映射表打包为 :class:`RAGAnswer` 返回。

    Attributes:
        _retriever: 检索组件，负责「问题 → Top-K 候选 chunk」。
        _prompt_builder: Prompt 构建组件，负责把候选 chunk 拼接为带引用编号的
            最终 Prompt，并产出引用映射表。
        _llm_client: LLM 客户端，负责基于 Prompt 生成最终答案。
        _reranker: 可选的重排序组件；为 ``None`` 时跳过重排序环节。
        _k: 检索阶段返回的候选数量。
    """

    def __init__(
        self,
        retriever: Retriever,
        prompt_builder: PromptBuilder,
        llm_client: BaseLLMClient,
        reranker: ReRanker | None = None,
        k: int = 5,
    ) -> None:
        """初始化 RAG 编排链。

        将检索、Prompt 构建、LLM 客户端与可选的重排序器注入到编排链中，并保存
        检索数量 ``k``。重排序器默认 ``None``，表示跳过重排序环节。

        Args:
            retriever: 检索组件，实现 :meth:`retrieve(query, k)` 返回候选 chunk。
            prompt_builder: Prompt 构建组件，实现 :meth:`build(question, chunks)`
                返回带引用编号的 :class:`~rag.prompt.RAGPrompt`。
            llm_client: LLM 客户端，实现 :meth:`chat_completion(messages, ...)`
                返回含 ``content`` 字段的答案字典。
            reranker: 可选的重排序组件，实现 :meth:`rerank(query, results)` 对
                候选做精细排序；传 ``None`` 时跳过重排序。
            k: 检索阶段返回的候选数量，默认 5。
        """
        self._retriever: Retriever = retriever
        """检索组件：负责「问题 → Top-K 候选 chunk」。"""

        self._prompt_builder: PromptBuilder = prompt_builder
        """Prompt 构建组件：负责拼接带引用编号的最终 Prompt 与引用映射表。"""

        self._llm_client: BaseLLMClient = llm_client
        """LLM 客户端：负责基于 Prompt 生成最终答案。"""

        self._reranker: ReRanker | None = reranker
        """可选的重排序组件；为 ``None`` 时跳过重排序环节。"""

        self._k: int = k
        """检索阶段返回的候选数量。"""

    def ask(self, question: str) -> RAGAnswer:
        """执行一次完整的端到端 RAG 问答。

        依次完成「检索 → 重排序（可选）→ Prompt 构建 → LLM 生成 → 返回引用」
        五个环节，最终返回携带答案文本与引用映射表的 :class:`RAGAnswer`。

        Args:
            question: 用户提出的自然语言问题。

        Returns:
            携带最终答案文本与引用映射表的 :class:`RAGAnswer` 实例。其中
            ``answer`` 为 LLM 生成的答案文本，``references`` 与
            :class:`~rag.prompt.RAGPrompt.references` 一致，可用于答案溯源。

        Raises:
            RetrievalError: 检索阶段失败时抛出（来自 :meth:`retrieve`，原样传播，
                不重复包装）。
            RerankError: 重排序阶段失败时抛出（来自 :meth:`rerank`，原样传播，
                不重复包装）。
            PromptBuildError: Prompt 构建阶段失败时抛出（来自 :meth:`build`，
                原样传播，不重复包装）。
            RAGChainError: 其余环节（如 LLM 调用）出现异常时，统一包装为本
                异常抛出，以区分链路级失败。
        """
        try:
            # 1. 检索：得到初步候选 chunk 列表。
            chunks: list[SearchResult] = self._retriever.retrieve(question, self._k)

            # 2. 重排序（可选）：注入 reranker 时对候选做精细排序。
            if self._reranker is not None:
                chunks = self._reranker.rerank(question, chunks)

            # 3. 构建 Prompt：得到带引用编号的完整 Prompt 与引用映射表。
            prompt: RAGPrompt = self._prompt_builder.build(question, chunks)

            # 4. 生成：调用 LLM 基于 Prompt 生成答案。
            response = self._llm_client.chat_completion(
                [{"role": "user", "content": prompt.text}]
            )

            # 5. 提取答案：dict 取其 content 字段，其它类型回退为字符串。
            if isinstance(response, dict):
                answer: str = response.get("content", "")
            else:
                answer = str(response)

        except RAGError:
            # 检索 / 重排序 / Prompt 构建等阶段已有专属异常，原样传播，
            # 避免重复包装导致调用方无法按阶段区分错误。
            raise
        except Exception as exc:
            # LLM 调用等其它异常统一包装为链路级异常。
            raise RAGChainError(f"RAG 流程执行失败：{exc}") from exc

        # 6. 返回：把答案文本与引用映射表打包为 RAGAnswer。
        return RAGAnswer(answer=answer, references=prompt.references)
