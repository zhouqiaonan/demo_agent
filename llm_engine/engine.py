"""LLM Engine 统一入口模块。

本模块提供 ``LLMEngine`` 类和其 Builder 模式的构造器 ``LLMEngineBuilder``，
将 Fallback 管理、重试策略、调用指标收集、费用追踪等子模块串联为
统一的外部调用入口。

典型用法（直接构造）::

    engine = LLMEngine(
        primary=openai_client,
        fallbacks=[deepseek_client],
        retry_config=RetryConfig(max_attempts=3),
        metrics=LLMMetrics(),
        cost_tracker=CostTracker(),
    )
    result = engine.chat([{"role": "user", "content": "Hello"}])

典型用法（Builder 模式）::

    engine = (
        LLMEngine.builder()
        .primary(openai_client)
        .add_fallback(deepseek_client)
        .with_retry(max_attempts=3, backoff="exponential")
        .with_metrics()
        .with_cost_tracking()
        .build()
    )
    result = engine.chat([{"role": "user", "content": "Hello"}])
"""

from __future__ import annotations

import time
from typing import Any

from llm_client.base import BaseLLMClient
from llm_engine.cost import CostTracker
from llm_engine.fallback import FallbackManager
from llm_engine.metrics import LLMMetrics
from llm_engine.retry import RetryConfig
from function_caller.config import GenerationConfig


# ============================================================================
# 生成参数白名单
# ============================================================================

_ALLOWED_GENERATION_PARAMS: frozenset[str] = frozenset({
    "temperature",
    "max_tokens",
    "top_p",
    "frequency_penalty",
    "presence_penalty",
    "stop",
})
"""允许透传给 ``BaseLLMClient.chat_completion`` 的 GenerationConfig 参数。"""


# ============================================================================
# LLMEngine
# ============================================================================


class LLMEngine:
    """LLM 调用统一入口，整合 Fallback、Retry、Metrics、Cost 全流程。

    本类是 llm_engine 包的对外 API 门面，内部将 ``FallbackManager``
    的降级能力与 ``LLMMetrics``（指标收集）、``CostTracker``（费用追踪）
    组合在一起，对外暴露简洁的 ``chat()`` 方法。

    支持两种构造方式：
    1. 直接调用构造函数，传入所有依赖。
    2. 通过 ``LLMEngine.builder()`` 获取 ``LLMEngineBuilder`` 进行链式构造。

    Attributes:
        primary: 主模型客户端（只读）。
        fallbacks: 备用客户端列表的防御性副本（只读）。
        metrics: 关联的 ``LLMMetrics`` 实例，可能为 ``None``。
        cost_tracker: 关联的 ``CostTracker`` 实例，可能为 ``None``。
        total_cost: 累计总费用（USD），从 cost_tracker 代理。
        total_calls: 累计调用次数，从 cost_tracker 代理。
    """

    def __init__(
        self,
        primary: BaseLLMClient,
        fallbacks: list[BaseLLMClient] | None = None,
        retry_config: RetryConfig | None = None,
        metrics: LLMMetrics | None = None,
        cost_tracker: CostTracker | None = None,
    ) -> None:
        """初始化 LLMEngine。

        所有可选组件（fallbacks、retry_config、metrics、cost_tracker）
        在未提供时使用合理的默认值或保持为 ``None``。

        Args:
            primary: 主模型客户端，必填，后续不可更换。
            fallbacks: 备用客户端列表。若未提供则仅使用主模型。
            retry_config: 单个客户端内部的重试策略配置。
                若为 ``None``，则使用默认 ``RetryConfig()``
                （最多 3 次尝试，指数退避，仅重试 ``TransientError``）。
            metrics: 已存在的 ``LLMMetrics`` 实例。若为 ``None``，
                则调用不收集指标。
            cost_tracker: 已存在的 ``CostTracker`` 实例。若为 ``None``，
                则调用不追踪费用。
        """
        self._primary: BaseLLMClient = primary
        """主模型客户端（内部存储，不可变）。"""

        self._fallbacks: list[BaseLLMClient] = (
            list(fallbacks) if fallbacks is not None else []
        )
        """备用客户端列表的副本。"""

        self._retry_config: RetryConfig = (
            retry_config if retry_config is not None else RetryConfig()
        )
        """每个客户端内部使用的重试配置。"""

        self._metrics: LLMMetrics | None = metrics
        """关联的指标收集器实例（可选）。"""

        self._cost_tracker: CostTracker | None = cost_tracker
        """关联的费用追踪器实例（可选）。"""

    # ------------------------------------------------------------------
    # 只读属性
    # ------------------------------------------------------------------

    @property
    def primary(self) -> BaseLLMClient:
        """主模型客户端。

        Returns:
            构造函数中传入的 ``BaseLLMClient`` 实例。
        """
        return self._primary

    @property
    def fallbacks(self) -> list[BaseLLMClient]:
        """备用客户端列表的防御性副本。

        Returns:
            一个新的列表，包含所有注册的备用 ``BaseLLMClient`` 实例。
        """
        return list(self._fallbacks)

    @property
    def metrics(self) -> LLMMetrics | None:
        """关联的 ``LLMMetrics`` 实例。

        Returns:
            ``LLMMetrics`` 实例，若未配置则返回 ``None``。
        """
        return self._metrics

    @property
    def cost_tracker(self) -> CostTracker | None:
        """关联的 ``CostTracker`` 实例。

        Returns:
            ``CostTracker`` 实例，若未配置则返回 ``None``。
        """
        return self._cost_tracker

    @property
    def total_cost(self) -> float:
        """累计总费用（USD），从 cost_tracker 代理。

        若未配置 ``cost_tracker``，始终返回 ``0.0``。

        Returns:
            所有已记录调用的累积费用。
        """
        if self._cost_tracker is None:
            return 0.0
        return self._cost_tracker.total_cost

    @property
    def total_calls(self) -> int:
        """累计调用次数，从 cost_tracker 代理。

        若未配置 ``cost_tracker``，始终返回 ``0``。

        Returns:
            所有已记录的 API 调用总次数。
        """
        if self._cost_tracker is None:
            return 0
        return self._cost_tracker.call_count

    # ------------------------------------------------------------------
    # 静态方法：Builder 模式入口
    # ------------------------------------------------------------------

    @staticmethod
    def builder() -> "LLMEngineBuilder":
        """返回 Builder 实例，展示 Builder 模式用法。

        Returns:
            一个新的 ``LLMEngineBuilder`` 实例，用于链式构造 ``LLMEngine``。

        Example:
            >>> engine = (
            ...     LLMEngine.builder()
            ...     .primary(mock_client)
            ...     .with_retry(max_attempts=1)
            ...     .build()
            ... )
        """
        return LLMEngineBuilder()

    # ------------------------------------------------------------------
    # 核心调用方法
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: list[dict[str, Any]],
        config: GenerationConfig | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """执行 LLM 聊天调用，整合 Fallback、Retry、Metrics、Cost 全流程。

        内部流程：
        1. 将主模型和备用客户端打包传入 ``FallbackManager``。
        2. 通过 ``execute_with_callbacks`` 执行带回调钩子的调用：
           - ``on_success``：记录 metrics（通过 ``record_call``）和
             cost（通过 ``record_call``），并捕获最终结果。
           - ``on_failure``：累计尝试次数。
           - ``on_switch``：不需要额外处理（FallbackManager 已记录）。
        3. 组装标准化响应字典。

        Args:
            messages: 符合 OpenAI Chat Completions 格式的消息列表，
                每个元素为 ``{"role": str, "content": str}`` 格式。
            config: 可选的 ``GenerationConfig`` 生成参数配置。
                若提供，会通过 ``to_dict()`` 转换为关键字参数合并到
                ``**kwargs`` 中。
            **kwargs: 透传给 ``BaseLLMClient.chat_completion`` 的额外参数，
                如 ``temperature``、``max_tokens`` 等。

        Returns:
            标准化响应字典，包含以下键：
            - ``content`` (str): 模型回复文本。
            - ``model`` (str): 实际使用的模型名称。
            - ``usage`` (dict): Token 用量信息。
            - ``attempts`` (int): 总尝试次数（含重试和 Fallback 切换）。
            - ``cost`` (float): 本次调用费用（USD）。
            - ``finish_reason`` (str): 完成原因。
            - ``raw_response`` (dict): 客户端原始返回的完整字典。

        Raises:
            NonRetryableError: 任意客户端返回不可重试错误（如认证失败）。
            AllModelsExhaustedError: 所有模型（含备用）均在重试后失败。
        """
        # ---- 合并 GenerationConfig 与 kwargs ----
        all_kwargs: dict[str, Any] = dict(kwargs)
        if config is not None:
            config_dict: dict[str, Any] = config.to_dict()
            for key, value in config_dict.items():
                if key in _ALLOWED_GENERATION_PARAMS:
                    all_kwargs[key] = value

        # ---- 构造 FallbackManager ----
        all_clients: list[BaseLLMClient] = [self._primary] + self._fallbacks
        fallback_manager: FallbackManager = FallbackManager(
            clients=all_clients,
            retry_config=self._retry_config,
        )

        # ---- 回调中使用的可变容器（用于跨回调传递状态）----
        captured_client: list[BaseLLMClient | None] = [None]
        """捕获最终成功的客户端实例。"""

        captured_cost: list[float] = [0.0]
        """捕获本次调用的费用（USD）。"""

        attempts: list[int] = [0]
        """累计总尝试次数（含各级重试和客户端切换）。"""

        # ---- 定义回调 ----
        def on_failure(
            _client: BaseLLMClient,
            _error: Exception,
            _attempt_count: int,
        ) -> None:
            """每次尝试失败时回调，累计尝试次数并记录失败指标。

            Args:
                _client: 当前客户端（未使用，仅用于签名匹配）。
                _error: 异常对象（未使用）。
                _attempt_count: 当前客户端内的尝试序号（未使用）。
            """
            attempts[0] += 1

            # 记录失败的调用指标
            if self._metrics is not None:
                self._metrics.record_call(
                    model=_client.model_name,
                    tokens={},
                    latency_ms=0.0,
                    success=False,
                )

        def on_success(
            client: BaseLLMClient,
            response: dict[str, Any],
            _attempt_count: int,
        ) -> None:
            """调用成功时回调，记录 metrics、cost 并捕获最终客户端。

            Args:
                client: 成功响应的客户端实例。
                response: ``chat_completion`` 返回的原始响应字典。
                _attempt_count: 当前客户端内的尝试序号（未使用）。
            """
            attempts[0] += 1
            captured_client[0] = client

            usage: dict[str, Any] = response.get("usage", {})

            # 记录调用指标
            if self._metrics is not None:
                elapsed_ms: float = (time.perf_counter() - start_time) * 1000
                self._metrics.record_call(
                    model=client.model_name,
                    tokens=usage,
                    latency_ms=elapsed_ms,
                    success=True,
                )

            # 记录调用费用
            if self._cost_tracker is not None:
                captured_cost[0] = self._cost_tracker.record_call(
                    client.model_name, usage
                )

        # ---- 执行带回调的 Fallback 调用 ----
        start_time: float = time.perf_counter()
        raw_response: dict[str, Any] = fallback_manager.execute_with_callbacks(
            messages=messages,
            on_success=on_success,
            on_failure=on_failure,
            **all_kwargs,
        )

        # ---- 确定最终使用的客户端（回调已设置 captured_client）----
        actual_client: BaseLLMClient = captured_client[0] or self._primary

        # ---- 组装标准化响应 ----
        return {
            "content": raw_response.get("content", ""),
            "model": raw_response.get("model", actual_client.model_name),
            "usage": raw_response.get("usage", {}),
            "attempts": attempts[0],
            "cost": captured_cost[0],
            "finish_reason": raw_response.get("finish_reason", "unknown"),
            "raw_response": raw_response,
        }

    # ------------------------------------------------------------------
    # 流式接口（预留）
    # ------------------------------------------------------------------

    def stream(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> None:
        """流式调用（暂不支持 Fallback——流式模式下仅使用主模型）。

        Args:
            messages: 符合 OpenAI Chat Completions 格式的消息列表。
            **kwargs: 透传给 ``BaseLLMClient.stream_completion`` 的额外参数。

        Raises:
            NotImplementedError: 当前版本尚未实现流式调用。
        """
        raise NotImplementedError(
            "流式调用暂未实现，请使用 chat() 方法"
        )


# ============================================================================
# LLMEngineBuilder
# ============================================================================


class LLMEngineBuilder:
    """LLMEngine 的流式构造器（Builder 模式）。

    提供链式 API 逐步配置 ``LLMEngine`` 的各项参数，
    最后通过 ``build()`` 方法产出完整的 ``LLMEngine`` 实例。

    Example:
        >>> engine = (
        ...     LLMEngineBuilder()
        ...     .primary(client)
        ...     .add_fallback(backup)
        ...     .with_retry(max_attempts=5, backoff="linear")
        ...     .with_metrics()
        ...     .with_cost_tracking()
        ...     .build()
        ... )
    """

    def __init__(self) -> None:
        """初始化 Builder，所有字段默认未设置。"""
        self._primary: BaseLLMClient | None = None
        """主模型客户端，build() 前必须设置。"""

        self._fallbacks: list[BaseLLMClient] = []
        """备用客户端累积列表。"""

        self._retry_kwargs: dict[str, Any] = {}
        """传递给 ``RetryConfig`` 的关键字参数字典。"""

        self._metrics: LLMMetrics | None = None
        """用户提供的或自动创建的 ``LLMMetrics`` 实例。"""

        self._auto_metrics: bool = False
        """标记：是否应在 build() 时自动创建 LLMMetrics。"""

        self._cost_tracker: CostTracker | None = None
        """用户提供的或自动创建的 ``CostTracker`` 实例。"""

        self._auto_cost_tracker: bool = False
        """标记：是否应在 build() 时自动创建 CostTracker。"""

    # ------------------------------------------------------------------
    # 链式配置方法
    # ------------------------------------------------------------------

    def primary(self, client: BaseLLMClient) -> "LLMEngineBuilder":
        """设置主模型客户端。

        Args:
            client: ``BaseLLMClient`` 实例，作为 LLM 调用的首选模型。

        Returns:
            当前 ``LLMEngineBuilder`` 实例，支持链式调用。
        """
        self._primary = client
        return self

    def add_fallback(self, client: BaseLLMClient) -> "LLMEngineBuilder":
        """添加一个备用客户端。

        可多次调用以注册多个备用模型。备用模型按注册顺序依次尝试。

        Args:
            client: ``BaseLLMClient`` 实例，当主模型失败时用作备用。

        Returns:
            当前 ``LLMEngineBuilder`` 实例，支持链式调用。
        """
        self._fallbacks.append(client)
        return self

    def with_retry(
        self,
        max_attempts: int = 3,
        backoff: str = "exponential",
        min_wait: float = 1.0,
        max_wait: float = 60.0,
    ) -> "LLMEngineBuilder":
        """配置重试策略。

        Args:
            max_attempts: 最大尝试次数（包含首次调用），默认 3。
            backoff: 退避策略，支持 ``"exponential"`` | ``"fixed"`` | ``"linear"``，
                默认 ``"exponential"``。
            min_wait: 最小等待时间（秒），默认 1.0。
            max_wait: 最大等待时间（秒），默认 60.0。

        Returns:
            当前 ``LLMEngineBuilder`` 实例，支持链式调用。
        """
        self._retry_kwargs = {
            "max_attempts": max_attempts,
            "backoff": backoff,
            "min_wait": min_wait,
            "max_wait": max_wait,
        }
        return self

    def with_metrics(
        self,
        metrics: LLMMetrics | None = None,
    ) -> "LLMEngineBuilder":
        """配置指标收集器。

        若未传入已有实例，则在 ``build()`` 时自动创建一个新的
        ``LLMMetrics`` 实例。
        注意：由于 Prometheus 指标名称需全局唯一，同一进程中只能存在
        一个 ``LLMMetrics`` 实例——若已通过直接构造方式创建，
        请将同一实例传入 Builder 以避免重复注册错误。

        Args:
            metrics: 已存在的 ``LLMMetrics`` 实例。若为 ``None``，
                则标记为在 ``build()`` 时自动创建。

        Returns:
            当前 ``LLMEngineBuilder`` 实例，支持链式调用。
        """
        if metrics is not None:
            self._metrics = metrics
        else:
            self._auto_metrics = True
        return self

    def with_cost_tracking(
        self,
        tracker: CostTracker | None = None,
    ) -> "LLMEngineBuilder":
        """配置费用追踪器。

        若未传入已有实例，则在 ``build()`` 时自动创建一个新的
        ``CostTracker`` 实例。

        Args:
            tracker: 已存在的 ``CostTracker`` 实例。若为 ``None``，
                则标记为在 ``build()`` 时自动创建。

        Returns:
            当前 ``LLMEngineBuilder`` 实例，支持链式调用。
        """
        if tracker is not None:
            self._cost_tracker = tracker
        else:
            self._auto_cost_tracker = True
        return self

    # ------------------------------------------------------------------
    # 构建方法
    # ------------------------------------------------------------------

    def build(self) -> LLMEngine:
        """构建并返回配置好的 ``LLMEngine`` 实例。

        使用 Builder 中累积的所有配置创建 ``LLMEngine``。
        对于标记为自动创建的 ``LLMMetrics`` / ``CostTracker``，
        在此时延迟实例化，避免在 Builder 链式调用期间触发副作用
        （如 Prometheus 指标重复注册）。

        Returns:
            配置完毕的 ``LLMEngine`` 实例。

        Raises:
            ValueError: 如果未调用 ``primary()`` 设置主模型客户端。
        """
        if self._primary is None:
            raise ValueError(
                "必须调用 .primary(client) 设置主模型客户端后才能 build()"
            )

        # 延迟创建 metrics（避免 Prometheus 重复注册）
        metrics: LLMMetrics | None = self._metrics
        if self._auto_metrics and metrics is None:
            metrics = LLMMetrics()

        # 延迟创建 cost_tracker
        cost_tracker: CostTracker | None = self._cost_tracker
        if self._auto_cost_tracker and cost_tracker is None:
            cost_tracker = CostTracker()

        return LLMEngine(
            primary=self._primary,
            fallbacks=self._fallbacks if self._fallbacks else None,
            retry_config=RetryConfig(**self._retry_kwargs),
            metrics=metrics,
            cost_tracker=cost_tracker,
        )
