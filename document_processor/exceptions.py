"""document_processor 包的自定义异常体系。

本模块定义了 document_processor 包中所有异常的基类和具体异常类型，
用于在文档加载、文本分割、元数据提取等上游处理流程中统一错误处理策略。
"""

from __future__ import annotations


class DocumentProcessorError(Exception):
    """document_processor 包所有异常的基类。

    所有 document_processor 包内抛出的异常都应继承自本类，
    以便调用方可以统一捕获和处理文档处理（加载 / 分割 / 元数据提取）
    相关的错误。
    """


class DocumentLoadError(DocumentProcessorError):
    """文档加载阶段的错误。

    适用于以下场景：
    - 目标文件不存在或路径不可访问
    - 文件格式不支持（例如传入未知的扩展名）
    - 文件内容损坏或解析失败（例如 PDF 无法解析、编码错误等）

    调用方应捕获本异常并向用户报告文档加载失败的具体原因，
    不应将本异常视为文本分割或元数据提取阶段的错误。
    """


class DocumentSplitError(DocumentProcessorError):
    """文本分割阶段的错误。

    适用于以下场景：
    - 分割器配置非法（例如 chunk_size / chunk_overlap 取值不合法）
    - 分割算法在运行过程中失败
    - 分割结果为空或不符合预期约束

    调用方应捕获本异常并向用户报告分割失败的原因，
    通常发生在文档成功加载之后、生成 chunk 之前。
    """


class MetadataExtractionError(DocumentProcessorError):
    """元数据提取阶段的错误。

    适用于以下场景：
    - 从文件名或文档结构中提取元数据时失败
    - 解析页码、章节、时间戳等结构化信息时发生异常
    - 元数据字段缺失或格式非法

    调用方应捕获本异常并向用户报告元数据提取失败的原因，
    通常发生在文档成功加载之后、构建 Document 元数据时。
    """
