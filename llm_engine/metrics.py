"""LLM API 调用指标收集器。

本模块基于 prometheus_client 库，提供 LLM API 调用的监控指标收集能力，
包括调用次数、Token 使用量和调用延迟分布。
"""

from __future__ import annotations

import logging
from typing import Any

from prometheus_client import Counter, Histogram, generate_latest

# ============================================================================
# 模型名称白名单（防止 Prometheus 标签基数爆炸）
# ============================================================================

_KNOWN_MODELS: frozenset[str] = frozenset({
    "gpt-4o",
    "gpt-4o-mini",
    "deepseek-chat",
    "deepseek-reasoner",
})
"""已知模型名称白名单。不在白名单中的模型名称将被归入 ``_UNKNOWN_MODEL_LABEL``。"""

_UNKNOWN_MODEL_LABEL: str = "unknown"
"""用于未知模型的 Prometheus 标签值。"""


class LLMMetrics:
    """LLM API 调用指标收集器。

    使用 prometheus_client 的 Counter 和 Histogram 记录：
    - 调用总次数（按模型和状态分组）
    - Token 使用量（按模型和类型分组）
    - 调用延迟分布

    由于 Prometheus 要求同一进程中指标名称全局唯一，建议在应用级别
    创建单一实例并共享给所有 ``LLMEngine`` 实例。多次实例化会自动
    复用底层已注册的指标对象，仍可安全使用。

    Attributes:
        call_total: 调用次数计数器，labels 为 model + status。
        token_usage: Token 使用量计数器，labels 为 model + type。
        latency_seconds: 调用延迟直方图，label 为 model。

    Example:
        >>> metrics = LLMMetrics()
        >>> metrics.record_call(
        ...     model="gpt-4o",
        ...     tokens={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        ...     latency_ms=1234.5,
        ...     success=True,
        ... )
        >>> print(metrics.get_summary())
    """

    # ---- 类级别共享的 Prometheus 指标（仅创建一次）----
    _call_total: Counter | None = None
    _token_usage: Counter | None = None
    _latency_seconds: Histogram | None = None

    def __init__(self) -> None:
        """初始化 LLM API 调用指标收集器。

        首次实例化时向 Prometheus 注册表注册三个指标对象。
        后续实例化复用已注册的同一指标对象，避免重复注册错误。
        """
        if LLMMetrics._call_total is None:
            LLMMetrics._call_total = Counter(
                "llm_call_total",
                "LLM API 调用总次数",
                ["model", "status"],
            )
        if LLMMetrics._token_usage is None:
            LLMMetrics._token_usage = Counter(
                "llm_token_usage_total",
                "Token 使用总量",
                ["model", "type"],
            )
        if LLMMetrics._latency_seconds is None:
            LLMMetrics._latency_seconds = Histogram(
                "llm_latency_seconds",
                "API 调用延迟分布（秒）",
                ["model"],
                buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0],
            )

        self.call_total: Counter = LLMMetrics._call_total  # type: ignore[assignment]
        """Counter: 按模型和状态（success/error）统计的 API 调用次数。"""

        self.token_usage: Counter = LLMMetrics._token_usage  # type: ignore[assignment]
        """Counter: 按模型和类型（prompt/completion/total）统计的 Token 使用量。"""

        self.latency_seconds: Histogram = LLMMetrics._latency_seconds  # type: ignore[assignment]
        """Histogram: 按模型统计的 API 调用延迟分布（单位：秒）。"""

    def _sanitize_model_label(self, model: str) -> str:
        """将模型名称映射到安全的 Prometheus 标签值。

        通过在已知模型白名单中查找模型名称来防止标签基数爆炸。
        不在白名单中的模型名称将被归入 ``"unknown"`` 标签并记录警告日志。

        Args:
            model: 原始模型名称字符串。

        Returns:
            安全的标签值：白名单内的原始名称或 ``"unknown"``。
        """
        if model in _KNOWN_MODELS:
            return model
        logging.getLogger(__name__).warning(
            "模型 '%s' 不在已知白名单中，指标将归入 '%s' 标签",
            model,
            _UNKNOWN_MODEL_LABEL,
        )
        return _UNKNOWN_MODEL_LABEL

    def record_call(
        self,
        model: str,
        tokens: dict[str, Any],
        latency_ms: float,
        success: bool,
    ) -> None:
        """记录一次 LLM API 调用的指标。

        Args:
            model: 模型名称，例如 "gpt-4o"、"claude-3-opus"。
            tokens: Token 使用量字典，可能包含以下键：
                - prompt_tokens: 输入 Token 数量，缺失时默认为 0。
                - completion_tokens: 输出 Token 数量，缺失时默认为 0。
                - total_tokens: 总 Token 数量，缺失时默认为 0。
            latency_ms: 调用延迟，单位为毫秒。
            success: 调用是否成功。True 表示成功，False 表示失败。
        """
        # 确定状态标签
        status: str = "success" if success else "error"

        # 对模型名称进行白名单过滤，防止标签基数爆炸
        safe_model: str = self._sanitize_model_label(model)

        # 记录调用次数
        self.call_total.labels(model=safe_model, status=status).inc()

        # 仅在成功时记录 Token 使用量和延迟
        if success:
            # 安全获取 Token 各字段，缺失时默认为 0
            prompt_tokens: float = float(tokens.get("prompt_tokens", 0))
            completion_tokens: float = float(tokens.get("completion_tokens", 0))
            total_tokens: float = float(tokens.get("total_tokens", 0))

            # 记录 Token 使用量
            self.token_usage.labels(model=safe_model, type="prompt").inc(prompt_tokens)
            self.token_usage.labels(model=safe_model, type="completion").inc(completion_tokens)
            self.token_usage.labels(model=safe_model, type="total").inc(total_tokens)

            # 记录延迟（毫秒转秒）
            self.latency_seconds.labels(model=safe_model).observe(latency_ms / 1000.0)

    def get_summary(self) -> str:
        """获取当前指标的文本摘要。

        使用 prometheus_client.generate_latest() 生成 Prometheus 文本格式的指标摘要，
        便于程序化查询和集成到监控系统中。

        Returns:
            Prometheus 文本格式的指标字符串。
        """
        return generate_latest().decode("utf-8")

    def reset(self) -> None:
        """重置所有指标。

        清除所有 Counter 和 Histogram 的累计值，用于测试环境
        或需要手动重置指标的场景。
        """
        # 清理由 prometheus_client 注册表管理的指标数据
        self.call_total._metrics.clear()
        self.token_usage._metrics.clear()
        self.latency_seconds._metrics.clear()
