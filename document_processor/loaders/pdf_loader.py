"""PDF 文档加载器模块。

本模块基于 ``pypdf`` 库实现 :class:`PdfLoader`，用于把 PDF 文件按页解析为
:class:`vector_store.document.Document` 列表。设计要点：

- **懒导入 pypdf**：``pypdf`` 依赖体积较大，且并非所有使用场景都需要真正
  解析 PDF（例如仅引用抽象接口或做单元测试时）。因此本模块**不在模块顶部
  import pypdf**，而是把 ``from pypdf import PdfReader`` 延迟到
  :meth:`PdfLoader.load` 方法体内执行，避免未安装 pypdf 时 import 即报错。

- **逐页抽取 + 跳过空白页**：对 PDF 的每一页调用 ``extract_text()`` 抽取
  文本，遇到文本为空（``None`` 或去除空白后为空）的页面直接跳过，避免在
  下游产生空 chunk。

- **0-based 页码**：每页构造的 ``Document`` 元数据中 ``page`` 字段使用
  **0-based** 页码（即第一页 ``page=0``），与 LangChain 的 ``PyPDFLoader``
  保持一致，便于下游跨组件对齐页码语义。
"""

from __future__ import annotations

from pathlib import Path

from document_processor._utils import _DEFAULT_MAX_FILE_SIZE
from document_processor.exceptions import DocumentLoadError
from document_processor.loaders.base import DocumentLoader
from vector_store.document import Document


class PdfLoader(DocumentLoader):
    """PDF 文件加载器。

    将指定路径的 PDF 文件按页解析为一组 :class:`Document`，每一页对应一个
    ``Document``，其正文为该页抽取得到的文本，元数据包含 ``source``（来源
    路径）与 ``page``（0-based 页码）。

    Attributes:
        _path: 规范化后的 PDF 文件路径（``pathlib.Path``）。
        _max_file_size: 允许的最大文件字节数（默认 50 MiB，防解压炸弹）。

    Example:
        >>> loader = PdfLoader("docs/example.pdf")
        >>> docs = loader.load()
        >>> docs[0].metadata
        {'source': 'docs/example.pdf', 'page': 0}
    """

    def __init__(
        self, path: str | Path, max_file_size: int = _DEFAULT_MAX_FILE_SIZE
    ) -> None:
        """初始化 PDF 加载器。

        Args:
            path: PDF 文件的路径（字符串或 :class:`pathlib.Path`）。
            max_file_size: 允许的最大文件字节数，默认 50 MiB（防解压炸弹）。
                解析前会校验实际文件大小，超过该值将抛出
                :class:`DocumentLoadError`。
        """
        self._path: Path = Path(path)
        """规范化后的 PDF 文件路径。"""

        self._max_file_size: int = max_file_size
        """允许的最大文件字节数（防解压炸弹）。"""

    def load(self) -> list[Document]:
        """解析 PDF 文件，逐页抽取文本并返回 :class:`Document` 列表。

        解析流程：

        1. 方法体内**懒导入** ``pypdf``（``from pypdf import PdfReader``）。
        2. 校验 ``self._path`` 是否存在且为文件，否则抛出
           :class:`DocumentLoadError`。
        3. 校验文件大小不超过 ``self._max_file_size``（防解压炸弹），超限
           抛出 :class:`DocumentLoadError`。
        4. 使用 ``PdfReader`` 打开 PDF，遍历每一页并调用
           ``page.extract_text()`` 抽取文本。
        5. 跳过文本为空（``None`` 或去除空白后为空）的页面。
        6. 对每个有效页面构造一个 ``Document``，元数据包含 ``source`` 与
           ``page``（**0-based** 页码，第一页 ``page=0``）。
        7. 解析过程中任何异常都会捕获并转抛为 :class:`DocumentLoadError`。

        Returns:
            按页码递增顺序排列的文档列表；若 PDF 中所有页面均为空白，
            返回空列表。

        Raises:
            DocumentLoadError: 文件不存在、不是文件，或 PDF 解析失败时抛出。
        """
        from pypdf import PdfReader

        if not self._path.exists():
            raise DocumentLoadError(f"PDF 文件不存在：{self._path}")
        if not self._path.is_file():
            raise DocumentLoadError(f"目标路径不是文件：{self._path}")

        # 解析前校验文件大小，防止解压炸弹。
        file_size: int = self._path.stat().st_size
        if file_size > self._max_file_size:
            raise DocumentLoadError(
                f"PDF 文件大小超限：{self._path}（{file_size} 字节）"
                f"超过允许的最大值 {self._max_file_size} 字节。"
            )

        documents: list[Document] = []
        try:
            reader: PdfReader = PdfReader(str(self._path))
            for page_index, page in enumerate(reader.pages):
                text: str | None = page.extract_text()
                if text is None or not text.strip():
                    # 跳过空白页，避免产生空 chunk。
                    continue
                documents.append(
                    Document(
                        text=text,
                        metadata={
                            "source": str(self._path),
                            "page": page_index,
                        },
                    )
                )
        except DocumentLoadError:
            raise
        except Exception as exc:
            raise DocumentLoadError(
                f"解析 PDF 文件失败：{self._path}，原因：{exc}"
            ) from exc

        return documents
