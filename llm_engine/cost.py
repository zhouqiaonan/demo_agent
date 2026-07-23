"""LLM 调用费用追踪模块。

本模块提供 CostTracker 类，用于根据各模型的官方定价计算
每次 API 调用的 token 费用，并支持按模型分组累计和汇总。
定价表参考各模型官方网站，日期：2025 年 7 月。
"""

from __future__ import annotations

import logging
from typing import Any

__all__ = ["CostTracker"]

# 模块级日志记录器
_logger: logging.Logger = logging.getLogger(__name__)


class CostTracker:
    """LLM 调用费用追踪器，按模型定价计算并累计费用。

    内部维护总费用、调用次数以及按模型分组的费用统计，
    每次 API 调用后通过 ``record_call`` 方法记录费用。

    定价表参考日期：2025 年 7 月，各模型官方网站公布价格。

    Usage::

        ct = CostTracker()
        cost = ct.record_call("gpt-4o", {
            "prompt_tokens": 1000,
            "completion_tokens": 500,
        })
        print(ct.get_summary())
    """

    # ------------------------------------------------------------------
    # 类属性：定价表（USD / 1M tokens）
    # ------------------------------------------------------------------
    PRICING: dict[str, dict[str, float]] = {
        "gpt-4o": {"prompt": 2.50, "completion": 10.00},
        "gpt-4o-mini": {"prompt": 0.15, "completion": 0.60},
        "deepseek-chat": {"prompt": 0.14, "completion": 0.28},
        "deepseek-reasoner": {"prompt": 0.55, "completion": 2.19},
    }
    """模型定价表，键为模型名称，值为包含 ``prompt`` 和 ``completion``
    单价的字典，单位为 USD/1M tokens。"""

    def __init__(self) -> None:
        """初始化 CostTracker，所有累计值归零。"""
        self._total_cost: float = 0.0
        self._call_count: int = 0
        self._cost_by_model: dict[str, float] = {}

    # ------------------------------------------------------------------
    # 公开方法
    # ------------------------------------------------------------------

    def calculate_cost(self, model: str, usage: dict[str, Any]) -> float:
        """根据模型名称和 token 用量计算单次调用费用。

        费用计算公式::

            prompt_tokens / 1e6 * prompt_price
            + completion_tokens / 1e6 * completion_price

        usage 字典中缺失的键默认计为 0。

        Args:
            model: 模型名称，必须在 ``PRICING`` 中注册。
            usage: token 用量字典，应包含 ``prompt_tokens`` 和
                ``completion_tokens`` 键（均为 ``int`` 类型）。

        Returns:
            本次调用的费用（USD）。若模型不在定价表中则返回 ``0.0``
            并记录一条 warning 日志。
        """
        pricing: dict[str, float] | None = self.PRICING.get(model)
        if pricing is None:
            _logger.warning(
                "模型 '%s' 不在定价表中，无法计算费用，返回 0.0", model
            )
            return 0.0

        prompt_tokens: int = usage.get("prompt_tokens", 0)
        completion_tokens: int = usage.get("completion_tokens", 0)

        prompt_cost: float = (prompt_tokens / 1_000_000.0) * pricing["prompt"]
        completion_cost: float = (
            completion_tokens / 1_000_000.0
        ) * pricing["completion"]

        return prompt_cost + completion_cost

    def record_call(self, model: str, usage: dict[str, Any]) -> float:
        """记录一次 API 调用并计算、累计费用。

        该方法依次完成以下操作：

        1. 调用 ``calculate_cost`` 计算本次调用费用。
        2. 将费用累加到 ``_total_cost`` 和 ``_cost_by_model[model]``。
        3. 递增 ``_call_count``。

        Args:
            model: 模型名称。
            usage: token 用量字典。

        Returns:
            本次调用的费用（USD）。
        """
        cost: float = self.calculate_cost(model, usage)

        self._total_cost += cost
        self._call_count += 1

        current: float = self._cost_by_model.get(model, 0.0)
        self._cost_by_model[model] = current + cost

        return cost

    def get_summary(self) -> dict[str, Any]:
        """返回费用摘要字典。

        包含总费用、调用次数和按模型分组的费用明细。

        Returns:
            字典，包含以下键：
            - ``total_cost``: 累计总费用（``float``）
            - ``call_count``: 累计调用次数（``int``）
            - ``by_model``: 按模型分组的费用字典（``dict[str, float]``）
        """
        return {
            "total_cost": self._total_cost,
            "call_count": self._call_count,
            "by_model": dict(self._cost_by_model),
        }

    def reset(self) -> None:
        """重置所有累计值，将总费用、调用次数和模型分组费用全部归零。"""
        self._total_cost = 0.0
        self._call_count = 0
        self._cost_by_model.clear()

    # ------------------------------------------------------------------
    # 只读属性
    # ------------------------------------------------------------------

    @property
    def total_cost(self) -> float:
        """累计总费用（USD），只读。

        Returns:
            所有已记录调用的费用总和。
        """
        return self._total_cost

    @property
    def call_count(self) -> int:
        """累计调用次数，只读。

        Returns:
            已记录的 API 调用总次数。
        """
        return self._call_count

    @property
    def cost_by_model(self) -> dict[str, float]:
        """按模型分组的累计费用（USD），返回防御性副本。

        Returns:
            一个新的字典，键为模型名称，值为该模型的累计费用。
            修改返回的字典不会影响内部状态。
        """
        return dict(self._cost_by_model)
