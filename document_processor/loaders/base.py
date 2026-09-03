"""文档加载器的抽象基类模块。

本模块定义了 document_processor 包中所有文档加载器的统一抽象接口
:class:`DocumentLoader`。任何具体的文档加载实现（如 PDF、Word 等格式）
都应继承本类并实现核心的 ``load`` 方法，从而把外部原始文件统一解析为
:class:`vector_store.document.Document` 列表，供下游的分割、向量化与
检索环节使用。

设计意图：

- ``DocumentLoader`` 负责「原始文件 → :class:`Document` 列表」的转换，
  是 RAG 上游处理链路的第一步。
- 本抽象层统一了不同格式加载器的对外契约，上层业务只需面向
  ``DocumentLoader`` 编程，即可无缝替换不同的文件解析后端。
- 具体实现应遵循「懒导入」原则，将重依赖（如 ``pypdf`` 等）延迟到
  ``load`` 方法体内 import，从而在仅引用本抽象基类时无需安装对应依赖。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from vector_store.document import Document


class DocumentLoader(ABC):
    """文档加载器的抽象基类。

    所有具体的文件格式加载器（PDF、Word 等）都应继承本类，并实现抽象
    方法 :meth:`load`，把外部原始文件解析为 ``list[Document]``。

    契约约定：

    - :meth:`load` 返回的每个 :class:`Document`，其 ``metadata`` 应至少
      包含 ``source``（来源路径）字段，用于溯源原始文件；实现还可根据
      需要补充 ``page`` 等更细粒度的定位字段。
    - 加载过程中任何失败（文件缺失、格式不支持、内容损坏等）都应抛出
      :class:`document_processor.exceptions.DocumentLoadError`（或其子类），
      以便调用方统一捕获。
    """

    @abstractmethod
    def load(self) -> list[Document]:
        """将外部原始文件解析为 :class:`Document` 列表（抽象方法）。

        这是所有文档加载器的核心契约，子类必须实现。返回的文档列表顺序
        应尽量与原始文件中的内容顺序保持一致（例如 PDF 按页码递增）。

        Returns:
            解析得到的文档列表；若文件无有效内容，可返回空列表。

        Raises:
            DocumentLoadError: 文件不存在、格式不支持或解析失败时抛出。
        """
        ...
