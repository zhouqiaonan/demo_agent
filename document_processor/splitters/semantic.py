"""语义分块分割器模块。

本模块实现 :class:`SemanticSplitter`，基于 **embedding 相似度变化** 切分文本，
参考 LlamaIndex 语义分块的思路，但为**自研实现**，不依赖 llama-index 库。

核心思想：语义相近的相邻句子，其 embedding 的余弦相似度较高；语义切换处
（话题转折、章节切换等），相邻句子 embedding 的相似度会显著下降。因此可
通过「相邻句子相似度序列 + 阈值」定位语义断点，从而在语义边界处把文本切
分为语义内聚的 chunk。

设计要点：

- **可插拔嵌入后端**：本类通过注入 :class:`vector_store.embeddings.base.EmbeddingClient`
  抽象接口获得嵌入能力，因此可无缝对接任意嵌入后端（OpenAI、本地
  sentence-transformers 等），不锁定具体实现。
- **批量嵌入**：对全部句子执行一次批量 ``embed`` 调用，返回与输入顺序严格
  一致的向量列表，随后仅在本地计算相似度，避免逐句往返调用。
- **相似度平滑**：为抑制单个噪声点导致的误断，用 ``buffer_size`` 对相似度
  序列做简单移动平均平滑（窗口越大越平滑、越少误断）。
- **懒依赖**：本模块仅依赖标准库 ``math`` 与已有接口，无重型第三方依赖，
  无需懒导入。
"""

from __future__ import annotations

import math

from document_processor.exceptions import DocumentSplitError
from document_processor.splitters.base import TextSplitter
from vector_store.embeddings.base import EmbeddingClient


class SemanticSplitter(TextSplitter):
    """语义分块分割器。

    先按句末标点把文本拆分为句子，再对全部句子做批量嵌入，计算相邻句子的
    余弦相似度，经 ``buffer_size`` 窗口平滑后，在相似度低于
    ``breakpoint_threshold`` 的位置断句，把连续的句子合并为语义内聚的 chunk。

    Attributes:
        _embedding: 注入的嵌入客户端，负责「句子列表 → 向量列表」的批量转换。
        _buffer_size: 相似度平滑窗口大小（正整数，越大越平滑）。
        _breakpoint_threshold: 相似度断点阈值（``[0.0, 1.0]``，越低越难断句、
            chunk 越大）。

    Example:
        >>> from vector_store.embeddings.base import EmbeddingClient
        >>> class FakeEmbedding(EmbeddingClient):
        ...     @property
        ...     def dimension(self) -> int:
        ...         return 2
        ...     def embed(self, texts: list[str]) -> list[list[float]]:
        ...         return [[1.0, 0.0] if "苹果" in t else [0.0, 1.0] for t in texts]
        >>> splitter = SemanticSplitter(embedding=FakeEmbedding(), buffer_size=1, breakpoint_threshold=0.5)
        >>> chunks = splitter.split_text("我爱吃苹果。今天天气很好。")
        >>> len(chunks) > 0
        True
    """

    def __init__(
        self,
        embedding: EmbeddingClient,
        buffer_size: int = 1,
        breakpoint_threshold: float = 0.5,
    ) -> None:
        """初始化语义分块分割器。

        Args:
            embedding: 嵌入客户端实例（实现 :class:`EmbeddingClient` 抽象接口），
                负责把一批句子转换为顺序一致的向量列表。
            buffer_size: 相似度平滑窗口大小（正整数）。对相似度序列做简单移动
                平均，``buffer_size=1`` 表示不平滑。窗口越大，相似度序列越平滑、
                越能抑制单点噪声，从而减少误断。
            breakpoint_threshold: 相似度断点阈值，取值范围 ``[0.0, 1.0]``。
                平滑后的相邻句子相似度低于该值时，判定在该位置断句。阈值越低，
                越难断句、生成的 chunk 越大；阈值越高，越易断句、chunk 越小。

        Raises:
            DocumentSplitError: 当 ``buffer_size < 1`` 或
                ``breakpoint_threshold`` 不在 ``[0.0, 1.0]`` 区间内时抛出。
        """
        if buffer_size < 1:
            raise DocumentSplitError(
                f"buffer_size 必须大于等于 1，当前值为 {buffer_size}。"
            )
        if not 0.0 <= breakpoint_threshold <= 1.0:
            raise DocumentSplitError(
                f"breakpoint_threshold 必须满足 0.0 <= breakpoint_threshold <= 1.0，"
                f"当前值为 {breakpoint_threshold}。"
            )

        self._embedding: EmbeddingClient = embedding
        """注入的嵌入客户端，负责「句子列表 → 向量列表」的批量转换。"""

        self._buffer_size: int = buffer_size
        """相似度平滑窗口大小（正整数，越大越平滑、越少误断）。"""

        self._breakpoint_threshold: float = breakpoint_threshold
        """相似度断点阈值（越低越难断句、chunk 越大）。"""

    def split_text(self, text: str) -> list[str]:
        """将单段文本按语义相似度变化切分为若干 chunk 文本。

        切分流程：

        1. 调用 :meth:`_split_sentences` 把文本拆分为句子列表。
        2. 若句子数不超过 1：``text.strip()`` 非空则返回 ``[text]``，否则返回
           空列表 ``[]``（无需嵌入）。
        3. 调用 ``self._embedding.embed(sentences)`` 对全部句子做一次批量嵌入，
           得到与句子顺序严格一致的向量列表。
        4. 计算相邻句子的余弦相似度序列 ``sims``，其中
           ``sims[i] = cosine(embeddings[i], embeddings[i+1])``。
        5. 用 ``_buffer_size`` 对 ``sims`` 做简单移动平均平滑：
           ``smoothed[i] = mean(sims[max(0, i-buffer_size+1) : i+1])``。
        6. 平滑后相似度 ``smoothed[i] < breakpoint_threshold`` 的位置 ``i``
           表示应在句子 ``i`` 与 ``i+1`` 之间断句。
        7. 按断点把句子列表切分，每个 chunk 为其包含句子的 ``"".join`` 拼接
           （句子间不加额外空格，保持原文标点与换行原样）。
        8. 过滤掉空 chunk，返回非空 chunk 列表。

        Args:
            text: 待切分的原始文本。

        Returns:
            切分得到的非空 chunk 文本列表。

        Raises:
            DocumentSplitError: 当嵌入客户端返回的向量数量与句子数量不一致、
                向量维度为 0，或分割过程抛出异常时抛出。
        """
        sentences: list[str] = self._split_sentences(text)

        if len(sentences) <= 1:
            return [text] if text.strip() else []

        try:
            embeddings: list[list[float]] = self._embedding.embed(sentences)
        except Exception as exc:
            raise DocumentSplitError(f"语义分块嵌入失败：{exc}") from exc

        if len(embeddings) != len(sentences):
            raise DocumentSplitError(
                f"嵌入客户端返回的向量数量 {len(embeddings)} 与句子数量 "
                f"{len(sentences)} 不一致。"
            )

        sims: list[float] = [
            self._cosine_similarity(embeddings[i], embeddings[i + 1])
            for i in range(len(sentences) - 1)
        ]

        smoothed: list[float] = self._smooth_similarities(sims)

        chunks: list[str] = []
        start: int = 0
        for i, sim in enumerate(smoothed):
            if sim < self._breakpoint_threshold:
                chunk: str = "".join(sentences[start : i + 1])
                if chunk.strip():
                    chunks.append(chunk)
                start = i + 1
        tail: str = "".join(sentences[start:])
        if tail.strip():
            chunks.append(tail)

        return chunks

    def _split_sentences(self, text: str) -> list[str]:
        """将文本按句末标点或换行符拆分为句子列表。

        手动逐字符遍历：遇到句末标点（``。！？；.!?;``）或换行符 ``\\n``
        即断句，标点保留在句子末尾；跳过纯空白片段（``buf.strip()`` 为空则
        丢弃）。

        Args:
            text: 待拆分句子的原始文本。

        Returns:
            拆分得到的句子列表（含句末标点），顺序与原文一致。
        """
        sentences: list[str] = []
        buf: str = ""
        for ch in text:
            buf += ch
            if ch in "。！？；.!?;\n":
                if buf.strip():
                    sentences.append(buf)
                buf = ""
        if buf.strip():
            sentences.append(buf)
        return sentences

    def _smooth_similarities(self, sims: list[float]) -> list[float]:
        """对相似度序列做简单移动平均平滑。

        平滑窗口大小为 :attr:`_buffer_size`，``buffer_size=1`` 时等价于不做
        平滑（原样返回）。对于第 ``i`` 个相似度，取其自身及其前
        ``buffer_size - 1`` 个相邻值（不足则从下标 0 起）的算术平均。

        Args:
            sims: 相邻句子余弦相似度序列。

        Returns:
            平滑后的相似度序列，长度与 ``sims`` 一致。
        """
        if self._buffer_size <= 1:
            return sims

        smoothed: list[float] = []
        for i in range(len(sims)):
            window: list[float] = sims[max(0, i - self._buffer_size + 1) : i + 1]
            smoothed.append(sum(window) / len(window))
        return smoothed

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """计算两个向量的余弦相似度。

        使用 ``math`` 模块计算范数与点积。若任一向量范数为 0（零向量），
        直接返回 ``0.0`` 以避免除零。

        Args:
            a: 第一个向量（``list[float]``）。
            b: 第二个向量（``list[float]``）。

        Returns:
            两向量的余弦相似度，取值范围 ``[-1.0, 1.0]``；任一向量为零向量
            时返回 ``0.0``。
        """
        dot: float = sum(x * y for x, y in zip(a, b))
        norm_a: float = math.sqrt(sum(x * x for x in a))
        norm_b: float = math.sqrt(sum(x * x for x in b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)
