"""递归文本分割器模块。

本模块基于 ``langchain_text_splitters`` 库实现 :class:`RecursiveTextSplitter`，
用于把长文本按一组递归分隔符切分为若干长度可控的 chunk。设计要点：

- **懒导入 langchain_text_splitters**：``langchain-text-splitters`` 依赖体积
  较大，且并非所有使用场景都需要真正执行分割（例如仅引用抽象接口或做单元
  测试时）。因此本模块**不在模块顶部 import**，而是把
  ``from langchain_text_splitters import RecursiveCharacterTextSplitter`` 延迟到
  :meth:`RecursiveTextSplitter.split_text` 方法体内执行，避免未安装依赖时
  import 即报错。

- **递归分隔符**：默认使用一组「中英混合友好」的分隔符，按优先级从粗粒度
  到细粒度依次尝试切分，依次为段落、换行、中文句号、感叹号、问号、分号、
  逗号、空格，最后退化为逐字符切分。

- **chunk 长度与重叠**：``chunk_size`` 控制每个 chunk 的目标最大长度，
  ``chunk_overlap`` 控制相邻 chunk 之间重叠的字符数（用于保留上下文，
  避免在边界处截断语义）。``chunk_overlap`` 必须严格小于 ``chunk_size``，
  否则分割器无法收敛。
"""

from __future__ import annotations

from document_processor.exceptions import DocumentSplitError
from document_processor.splitters.base import TextSplitter


class RecursiveTextSplitter(TextSplitter):
    """递归文本分割器。

    基于 ``langchain_text_splitters.RecursiveCharacterTextSplitter``，按一组
    递归分隔符把长文本切分为若干长度可控、可重叠的 chunk，适用于中文、英文
    及中英混合文本的通用场景。

    Attributes:
        _chunk_size: 每个 chunk 的目标最大长度（字符数）。
        _chunk_overlap: 相邻 chunk 之间的重叠字符数。
        _separators: 递归分隔符列表，按优先级从粗粒度到细粒度排列。

    Example:
        >>> splitter = RecursiveTextSplitter(chunk_size=100, chunk_overlap=20)
        >>> chunks = splitter.split_text("这是一段很长的中文文本。它会被递归地切分。")
        >>> len(chunks) > 0
        True
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: list[str] | None = None,
    ) -> None:
        """初始化递归文本分割器。

        Args:
            chunk_size: 每个 chunk 的目标最大长度（字符数），必须大于 0。
            chunk_overlap: 相邻 chunk 之间的重叠字符数，必须满足
                ``0 <= chunk_overlap < chunk_size``。
            separators: 递归分隔符列表。默认为
                ``["\\n\\n", "\\n", "。", "！", "？", "；", "，", " ", ""]``
                （中英混合友好，按优先级从粗粒度到细粒度排列）。

        Raises:
            DocumentSplitError: 当 ``chunk_size`` 或 ``chunk_overlap`` 取值
                非法（``chunk_size <= 0`` 或 ``chunk_overlap`` 不在
                ``[0, chunk_size)`` 区间内）时抛出。
        """
        if chunk_size <= 0:
            raise DocumentSplitError(
                f"chunk_size 必须大于 0，当前值为 {chunk_size}。"
            )
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise DocumentSplitError(
                f"chunk_overlap 必须满足 0 <= chunk_overlap < chunk_size，"
                f"当前 chunk_overlap={chunk_overlap}，chunk_size={chunk_size}。"
            )

        self._chunk_size: int = chunk_size
        """每个 chunk 的目标最大长度（字符数）。"""

        self._chunk_overlap: int = chunk_overlap
        """相邻 chunk 之间的重叠字符数。"""

        self._separators: list[str] = (
            separators
            if separators is not None
            else ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
        )
        """递归分隔符列表，按优先级从粗粒度到细粒度排列。"""

    def split_text(self, text: str) -> list[str]:
        """将单段文本按递归分隔符切分为若干 chunk 文本。

        切分流程：

        1. 方法体内**懒导入** ``langchain_text_splitters`` 的
           ``RecursiveCharacterTextSplitter``。
        2. 若输入文本去除空白后为空（``text.strip()`` 为空），直接返回空列表
           ``[]``（空进空出）。
        3. 用保存的 ``_chunk_size``、``_chunk_overlap``、``_separators``
           参数实例化 langchain 的 ``RecursiveCharacterTextSplitter``。
        4. 调用其 ``split_text(text)`` 得到 chunk 文本列表。
        5. 对非空输入，若返回空列表或分割过程中抛出异常，统一转抛为
           :class:`DocumentSplitError`。

        Args:
            text: 待切分的原始文本。

        Returns:
            切分得到的 chunk 文本列表；空/纯空白输入返回空列表。

        Raises:
            DocumentSplitError: 当非空输入的分割结果为空列表或分割过程抛出
                异常时抛出。
        """
        if not text.strip():
            # 空/纯空白输入：空进空出，返回空列表。
            return []

        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
            separators=self._separators,
        )
        try:
            chunks: list[str] = splitter.split_text(text)
        except Exception as exc:
            raise DocumentSplitError(f"递归分割文本失败：{exc}") from exc

        if not chunks:
            raise DocumentSplitError(
                "递归分割文本失败：分割结果为空列表。"
            )

        return chunks
