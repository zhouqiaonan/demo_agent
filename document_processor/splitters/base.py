"""文本分割器的抽象基类模块。

本模块定义了 document_processor 包中所有文本分割器的统一抽象接口
:class:`TextSplitter`。任何具体的分割实现（递归分割、语义分块等）都应
继承本类并实现核心的 ``split_text`` 方法，从而把文本切分为若干 chunk。

设计意图：

- ``TextSplitter`` 负责「文本 → chunk 列表」的转换，是 RAG 上游处理链路中
  紧跟文档加载之后的第二个环节。
- :meth:`split_documents` 是模板方法，在 :meth:`split_text` 之上提供了
  「文档列表 → chunk 文档列表」的高层封装：它遍历每个 :class:`Document`，
  调用 :meth:`split_text` 切分正文，并保留源文档 metadata 的同时追加局部
  的 ``chunk_index`` 索引。
- 具体实现应遵循「懒导入」原则，将重依赖（如 ``langchain_text_splitters``）
  延迟到 ``split_text`` 方法体内 import，从而在仅引用本抽象基类时无需安装
  对应依赖。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from vector_store.document import Document


class TextSplitter(ABC):
    """文本分割器的抽象基类。

    所有具体的分割实现（递归分割、语义分块等）都应继承本类，并实现抽象
    方法 :meth:`split_text`，把单段文本切分为若干 chunk 文本。

    契约约定：

    - :meth:`split_text` 接收一段纯文本，返回按内容顺序排列的 chunk 文本
      列表，供下游向量化使用。
    - **空进空出**：:meth:`split_text` 对空/纯空白输入应直接返回空列表
      ``[]``，而非抛出异常；仅当非空输入的分割结果为空（不符合预期）时
      才抛出 :class:`document_processor.exceptions.DocumentSplitError`。
    - :meth:`split_documents` 是模板方法（子类通常无需覆盖）：它调用
      :meth:`split_text` 切分每个源文档的正文，并**保留源文档 metadata**、
      同时为每个 chunk 追加局部索引 ``chunk_index``（从 0 开始）。
    - 分割过程中任何失败（配置非法、算法运行失败等）都应抛出
      :class:`document_processor.exceptions.DocumentSplitError`，以便调用方
      统一捕获。
    """

    @abstractmethod
    def split_text(self, text: str) -> list[str]:
        """将单段文本切分为若干 chunk 文本（抽象方法）。

        这是所有文本分割器的核心契约，子类必须实现。返回的 chunk 文本列表
        顺序应尽量与输入文本中的内容顺序保持一致。

        契约约定：对空/纯空白输入应直接返回空列表（空进空出），不抛出
        异常；仅当非空输入的分割结果为空（不符合预期）时抛出
        :class:`DocumentSplitError`。

        Args:
            text: 待切分的原始文本。

        Returns:
            切分得到的 chunk 文本列表；空/纯空白输入返回空列表。

        Raises:
            DocumentSplitError: 分割失败或非空输入结果不符合预期约束时抛出。
        """
        ...

    def split_documents(self, documents: list[Document]) -> list[Document]:
        """将文档列表切分为 chunk 文档列表（模板方法）。

        遍历每个 :class:`Document`，调用 :meth:`split_text` 得到 chunk 文本
        列表，并为每个 chunk 构造一个新的 :class:`Document`：其 ``text`` 为
        chunk 文本，``metadata`` 为源文档 metadata 的浅拷贝再追加局部
        ``chunk_index``（从 0 开始）；``id`` 交由 :class:`Document` 自动生成。

        设计约定：本方法保留源文档的全部 metadata（如 ``source``、``page``
        等），使下游仍能溯源每个 chunk 所属的原始文档；同时 ``chunk_index``
        仅在单个源文档内部局部递增，不跨文档累加。

        Args:
            documents: 待分割的文档列表。

        Returns:
            所有 chunk 拼接而成的文档列表，每个 chunk 为一个独立
            :class:`Document`。

        Raises:
            DocumentSplitError: 任一文档分割失败时抛出。
        """
        result: list[Document] = []
        for doc in documents:
            chunks: list[str] = self.split_text(doc.text)
            for index, chunk in enumerate(chunks):
                result.append(
                    Document(
                        text=chunk,
                        metadata={**doc.metadata, "chunk_index": index},
                    )
                )
        return result
