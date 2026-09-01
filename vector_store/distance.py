"""向量相似度与距离计算的纯函数模块。

本模块提供向量数据库中最常用的三种度量方式（余弦相似度、欧氏距离、点积）
及其对应的枚举定义，并提供了「距离 -> 相似度分数」的映射函数，用于将
ChromaDB 等向量后端 ``query`` 接口返回的「距离」统一转换为「分数」。

核心语义约定（务必区分）：

- **相似度分数（score）**：值**越高越相似**。例如余弦相似度（范围为
  ``[-1, 1]``，通常归一化后落在 ``[0, 1]``）、点积（内积）等。
- **距离（distance）**：值**越低越近**。例如欧氏距离（``>= 0``）、
  余弦距离（``1 - cosine_similarity``，范围 ``[0, 2]``）。

二者方向相反：一个「越大越好」，一个「越小越好」。ChromaDB 的 ``query``
返回结果中的 ``distances`` 字段语义随 ``distance_metric``（collection 的
空间度量）不同而不同，因此需要 ``distance_to_score`` 将其统一映射为
「越大越相似」的分数，方便与 ``SearchResult.score`` 的语义对齐。

本模块仅依赖标准库 ``math``，所有函数均为无副作用的纯函数。
"""

from __future__ import annotations

import math
from enum import Enum


class DistanceMetric(str, Enum):
    """向量空间距离度量枚举。

    枚举的 ``value`` 与 ChromaDB 原生支持的字符串保持一致，可直接传递给
    ChromaDB collection 创建时的 ``metadata`` 参数（``{"hnsw:space": ...}``）
    或用于判断 ``query`` 返回距离的语义。

    Attributes:
        COSINE: 余弦度量，值为 ``"cosine"``。
        L2: 欧氏距离（L2 范数）度量，值为 ``"l2"``。
        IP: 内积（点积）度量，值为 ``"ip"``。
    """

    COSINE = "cosine"
    """余弦度量，对应 ChromaDB 的 ``"cosine"``。"""

    L2 = "l2"
    """欧氏距离度量，对应 ChromaDB 的 ``"l2"``。"""

    IP = "ip"
    """内积度量，对应 ChromaDB 的 ``"ip"``。"""


def _validate_lengths(a: list[float], b: list[float]) -> None:
    """校验两个浮点向量长度是否一致。

    两个向量长度不一致时抛出 :class:`ValueError`，报错信息为中文，并包含
    两个向量的实际长度，便于调用方定位问题。

    Args:
        a: 第一个浮点向量。
        b: 第二个浮点向量。

    Raises:
        ValueError: 当 ``len(a) != len(b)`` 时抛出。
    """
    if len(a) != len(b):
        raise ValueError(
            f"向量长度不一致：第一个向量长度为 {len(a)}，"
            f"第二个向量长度为 {len(b)}"
        )


def dot_product(a: list[float], b: list[float]) -> float:
    """计算两个向量的点积（内积）。

    数学公式：``dot(a, b) = sum(a_i * b_i)``，即逐元素相乘后求和。

    边界情况：
    - 两个向量长度不一致时抛出 :class:`ValueError`（中文报错信息）。
    - 空向量（长度为 0）的点积为 ``0.0``（空求和的结果）。

    Args:
        a: 第一个浮点向量。
        b: 第二个浮点向量。

    Returns:
        两个向量的点积（内积）结果。

    Raises:
        ValueError: 当两个向量长度不一致时抛出。

    Example:
        >>> dot_product([1.0, 0.0], [0.0, 1.0])
        0.0
        >>> dot_product([1.0, 2.0], [3.0, 4.0])
        11.0
    """
    _validate_lengths(a, b)
    return sum(x * y for x, y in zip(a, b))


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """计算两个向量的余弦相似度。

    数学公式：``cos(a, b) = dot(a, b) / (|a| * |b|)``，其中
    ``|a| = sqrt(sum(a_i^2))`` 为向量的欧氏范数（模长）。

    边界情况：
    - 两个向量长度不一致时抛出 :class:`ValueError`（中文报错信息）。
    - 若任一向量模长为 0（含空向量、全零向量），返回 ``0.0``，避免除零。
    - 空向量（长度为 0）模长为 0，因此返回 ``0.0``。

    Args:
        a: 第一个浮点向量。
        b: 第二个浮点向量。

    Returns:
        余弦相似度，范围为 ``[-1, 1]``；当任一向量模长为 0 时返回 ``0.0``。

    Raises:
        ValueError: 当两个向量长度不一致时抛出。

    Example:
        >>> cosine_similarity([1.0, 0.0], [0.0, 1.0])
        0.0
        >>> cosine_similarity([1.0, 0.0], [1.0, 0.0])
        1.0
    """
    _validate_lengths(a, b)
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def euclidean_distance(a: list[float], b: list[float]) -> float:
    """计算两个向量的欧氏距离（L2 距离）。

    数学公式：``d(a, b) = sqrt(sum((a_i - b_i)^2))``。

    边界情况：
    - 两个向量长度不一致时抛出 :class:`ValueError`（中文报错信息）。
    - 空向量（长度为 0）的距离为 ``0.0``：空向量逐元素差值的平方求和为 0，
      开方后仍为 0，即两个空向量视为完全重合，距离为 0。

    Args:
        a: 第一个浮点向量。
        b: 第二个浮点向量。

    Returns:
        两个向量的欧氏距离，范围为 ``>= 0``。

    Raises:
        ValueError: 当两个向量长度不一致时抛出。

    Example:
        >>> round(euclidean_distance([1.0, 0.0], [0.0, 1.0]), 6)
        1.414214
        >>> euclidean_distance([1.0, 1.0], [1.0, 1.0])
        0.0
    """
    _validate_lengths(a, b)
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def distance_to_score(metric: DistanceMetric, distance: float) -> float:
    """将「距离」映射为「相似度分数」。

    本函数用于统一不同度量下 ChromaDB ``query`` 返回的 ``distances`` 语义，
    将其转换为「越大越相似」的分数（与 ``SearchResult.score`` 对齐）。

    各度量的映射规则：

    - ``COSINE``：ChromaDB 的余弦「距离」定义为 ``1 - cosine_similarity``，
      故 ``score = 1 - distance``，结果即还原为余弦相似度，范围 ``[-1, 1]``。
    - ``IP``：ChromaDB 的 ip「距离」定义为 ``1 - 内积``，故
      ``score = 1 - distance``，结果即还原为内积。
    - ``L2``：ChromaDB 的 l2「距离」定义为**平方**欧氏距离 ``‖x−y‖²``
      （未开方），值越小越相似，故 ``score = -distance``（单调映射即可，
      仅改变方向、不改变相对顺序）。注意：本模块的
      :func:`euclidean_distance` 返回的是开方后的欧氏距离，二者数值语义
      不同，但排序结果一致。

    Args:
        metric: 距离度量类型（:class:`DistanceMetric` 成员）。
        distance: ChromaDB ``query`` 返回的距离值。

    Returns:
        映射后的相似度分数，值越大表示越相似。

    Raises:
        ValueError: 当 ``metric`` 不是 :class:`DistanceMetric` 的合法成员时抛出。

    Example:
        >>> distance_to_score(DistanceMetric.COSINE, 1.0)
        0.0
        >>> distance_to_score(DistanceMetric.L2, 0.5)
        -0.5
        >>> distance_to_score(DistanceMetric.IP, 3.0)
        -2.0
    """
    if not isinstance(metric, DistanceMetric):
        raise ValueError(f"不支持的度量类型：{metric!r}")
    if metric is DistanceMetric.COSINE:
        return 1.0 - distance
    if metric is DistanceMetric.IP:
        return 1.0 - distance
    if metric is DistanceMetric.L2:
        return -distance
    # 防御性兜底：理论上不可达，但保留以保证对所有枚举成员都有返回。
    raise ValueError(f"不支持的度量类型：{metric!r}")
