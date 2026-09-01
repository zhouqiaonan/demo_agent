"""Tests for vector_store/distance.py — 相似度与距离纯函数。"""

from __future__ import annotations

import math

import pytest

from vector_store.distance import (
    DistanceMetric,
    cosine_similarity,
    distance_to_score,
    dot_product,
    euclidean_distance,
)


class TestDotProduct:
    """测试 dot_product 点积函数。"""

    def test_orthogonal_vectors(self) -> None:
        """正交向量的点积应为 0。"""
        assert dot_product([1.0, 0.0], [0.0, 1.0]) == 0.0

    def test_known_product(self) -> None:
        """已知向量的点积应正确。"""
        assert dot_product([1.0, 2.0], [3.0, 4.0]) == 11.0

    def test_length_mismatch_raises(self) -> None:
        """长度不一致时应抛出 ValueError。"""
        with pytest.raises(ValueError):
            dot_product([1.0], [1.0, 2.0])


class TestCosineSimilarity:
    """测试 cosine_similarity 余弦相似度函数。"""

    def test_orthogonal_is_zero(self) -> None:
        """正交向量的余弦相似度应为 0。"""
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0

    def test_identical_is_one(self) -> None:
        """同向（相同）向量的余弦相似度应约为 1。"""
        assert cosine_similarity([1.0, 2.0], [1.0, 2.0]) == pytest.approx(1.0)

    def test_zero_vector_is_zero(self) -> None:
        """零向量的余弦相似度应为 0（避免除零）。"""
        assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0

    def test_length_mismatch_raises(self) -> None:
        """长度不一致时应抛出 ValueError。"""
        with pytest.raises(ValueError):
            cosine_similarity([1.0], [1.0, 2.0])


class TestEuclideanDistance:
    """测试 euclidean_distance 欧氏距离函数。"""

    def test_same_point_is_zero(self) -> None:
        """相同向量的欧氏距离应为 0。"""
        assert euclidean_distance([1.0, 1.0], [1.0, 1.0]) == 0.0

    def test_known_distance(self) -> None:
        """正交单位向量的欧氏距离应为 sqrt(2)。"""
        assert euclidean_distance([1.0, 0.0], [0.0, 1.0]) == pytest.approx(
            math.sqrt(2.0)
        )

    def test_length_mismatch_raises(self) -> None:
        """长度不一致时应抛出 ValueError。"""
        with pytest.raises(ValueError):
            euclidean_distance([1.0], [1.0, 2.0])


class TestDistanceToScore:
    """测试 distance_to_score 距离到分数的映射。"""

    def test_cosine_mapping(self) -> None:
        """COSINE 度量下 score = 1 - distance。"""
        assert distance_to_score(DistanceMetric.COSINE, 0.25) == pytest.approx(0.75)

    def test_cosine_extreme(self) -> None:
        """COSINE 度量下距离=2.0（相似度=-1.0）应映射为 -1.0。"""
        assert distance_to_score(DistanceMetric.COSINE, 2.0) == pytest.approx(-1.0)

    def test_ip_mapping(self) -> None:
        """IP 度量下 score = 1 - distance（还原内积）。"""
        assert distance_to_score(DistanceMetric.IP, 3.0) == pytest.approx(-2.0)

    def test_l2_mapping(self) -> None:
        """L2 度量下 score = -distance。"""
        assert distance_to_score(DistanceMetric.L2, 0.5) == pytest.approx(-0.5)

    def test_invalid_metric_raises(self) -> None:
        """传入非 DistanceMetric 成员时应抛出 ValueError。"""
        with pytest.raises(ValueError):
            distance_to_score("cosine", 0.5)  # 字符串不是枚举成员


class TestDistanceMetric:
    """测试 DistanceMetric 枚举值。"""

    def test_values_match_chroma(self) -> None:
        """枚举值应等于 ChromaDB 原生字符串。"""
        assert DistanceMetric.COSINE.value == "cosine"
        assert DistanceMetric.L2.value == "l2"
        assert DistanceMetric.IP.value == "ip"
