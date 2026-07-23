"""
llm_engine 演示脚本

展示多模型 Fallback + Retry + Metrics + Cost Tracking 的完整工作流。

运行方式::

    python examples/llm_engine_demo.py

然后打开浏览器访问 http://localhost:9090/metrics 查看 Prometheus 指标。
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from prometheus_client import start_http_server

from llm_engine import (
    AllModelsExhaustedError,
    CostTracker,
    LLMEngine,
    LLMEngineBuilder,
    LLMMetrics,
    RetryConfig,
    TransientError,
)


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _make_mock_client(model_name: str) -> MagicMock:
    """创建一个具有 model_name 和 chat_completion 属性的 mock 客户端。

    Args:
        model_name: 模型名称（如 "gpt-4o"、"deepseek-chat"），用于 CostTracker 定价匹配。

    Returns:
        配置好的 MagicMock 实例，模拟 ``BaseLLMClient`` 接口。
    """
    client: MagicMock = MagicMock()
    client.model_name = model_name
    return client


def _make_success_response(
    content: str = "Hello!",
    model: str = "gpt-4o",
    usage: dict[str, int] | None = None,
) -> dict[str, Any]:
    """构造一个标准的成功响应字典，格式与 ``BaseLLMClient.chat_completion`` 一致。

    Args:
        content: 模型回复文本内容。
        model: 实际使用的模型名称。
        usage: Token 用量字典，缺失时使用默认值。

    Returns:
        符合 chat_completion 返回格式的字典，包含 role、content、model、usage、finish_reason。
    """
    if usage is None:
        usage = {
            "prompt_tokens": 350,
            "completion_tokens": 150,
            "total_tokens": 500,
        }
    return {
        "role": "assistant",
        "content": content,
        "model": model,
        "usage": usage,
        "finish_reason": "stop",
    }


def _format_cost(cost: float) -> str:
    """格式化费用为美元字符串。

    Args:
        cost: 费用金额（USD）。

    Returns:
        格式化为 6 位小数的美元字符串，如 "$0.008750"。
    """
    return f"${cost:,.6f}"


def _print_separator(title: str) -> None:
    """打印带标题的分隔线。

    Args:
        title: 分隔线标题文本。
    """
    print(f"\n{'─' * 52}")
    print(f"  {title}")
    print(f"{'─' * 52}\n")


# ---------------------------------------------------------------------------
# 演示函数
# ---------------------------------------------------------------------------


def demo_direct_construction() -> None:
    """演示 1: 直接构造 LLMEngine。

    创建主模型 mock（OpenAI gpt-4o），通过直接调用 ``LLMEngine`` 构造函数
    创建引擎实例，然后调用 ``chat()`` 展示正常流程。

    展示点：
    - 直接构造 LLMEngine 的简洁方式
    - chat() 返回的标准化响应字典结构
    """
    print("🚀 演示 1: 直接构造 LLMEngine")
    print("-" * 40)

    # ---- 创建 mock 客户端 ----
    # 模拟 OpenAI gpt-4o 客户端，model_name 用于 CostTracker 定价匹配
    primary: MagicMock = _make_mock_client("gpt-4o")
    primary.chat_completion.return_value = _make_success_response(
        content="你好！我是 GPT-4o，很高兴为你服务！",
        model="gpt-4o",
        usage={"prompt_tokens": 350, "completion_tokens": 150, "total_tokens": 500},
    )

    # ---- 直接构造 LLMEngine ----
    # 方式：LLMEngine(primary=..., fallbacks=[...], metrics=..., cost_tracker=...)
    metrics: LLMMetrics = LLMMetrics()
    cost_tracker: CostTracker = CostTracker()

    engine: LLMEngine = LLMEngine(
        primary=primary,
        retry_config=RetryConfig(max_attempts=1, backoff="fixed", min_wait=0.01),
        metrics=metrics,
        cost_tracker=cost_tracker,
    )

    # ---- 发起调用 ----
    messages: list[dict[str, str]] = [
        {"role": "user", "content": "Hello from direct construction!"}
    ]
    response: dict[str, Any] = engine.chat(messages)

    # ---- 打印结果 ----
    print(f"  📝 请求: {messages[0]['content']}")
    print(f"  ✅ 响应: {response['content']}")
    print(f"  🤖 模型: {response['model']}")
    print(f"  📊 Token 用量: {response['usage']}")
    print(f"  🔄 尝试次数: {response['attempts']}")
    print(f"  💰 本次费用: {_format_cost(response['cost'])}")
    print(f"  🏁 完成原因: {response['finish_reason']}")

    cost_summary: dict[str, Any] = cost_tracker.get_summary()
    print(f"\n  📈 累计统计: 总调用 {cost_summary['call_count']} 次，"
          f"总费用 {_format_cost(cost_summary['total_cost'])}")

    # 重置 metrics，避免影响后续演示
    metrics.reset()
    cost_tracker.reset()


def demo_builder_pattern() -> None:
    """演示 2: Builder 模式构造 LLMEngine。

    使用 ``LLMEngine.builder()`` 链式 API 配置并构建 ``LLMEngine`` 实例，
    展示 Builder 模式的优雅用法。

    展示点：
    - 链式调用 .primary() → .add_fallback() → .with_retry() → .with_metrics() → .build()
    - Builder 自动创建 LLMMetrics / CostTracker 实例
    """
    print("\n🚀 演示 2: Builder 模式构造 LLMEngine")
    print("-" * 40)

    # ---- 创建 mock 客户端 ----
    primary: MagicMock = _make_mock_client("gpt-4o")
    primary.chat_completion.return_value = _make_success_response(
        content="GPT-4o 通过 Builder 模式响应！",
        model="gpt-4o",
        usage={"prompt_tokens": 200, "completion_tokens": 100, "total_tokens": 300},
    )

    backup: MagicMock = _make_mock_client("deepseek-chat")
    backup.chat_completion.return_value = _make_success_response(
        content="DeepSeek 备用响应",
        model="deepseek-chat",
        usage={"prompt_tokens": 200, "completion_tokens": 100, "total_tokens": 300},
    )

    # ---- Builder 模式构造 ----
    # 链式配置：主模型 → 备用模型 → 重试策略 → 指标 → 费用 → 构建
    engine: LLMEngine = (
        LLMEngine.builder()
        .primary(primary)
        .add_fallback(backup)
        .with_retry(max_attempts=1, backoff="fixed", min_wait=0.01)
        .with_metrics()
        .with_cost_tracking()
        .build()
    )

    # ---- 发起调用 ----
    messages: list[dict[str, str]] = [
        {"role": "user", "content": "Hello from Builder pattern!"}
    ]
    response: dict[str, Any] = engine.chat(messages)

    # ---- 打印结果 ----
    print(f"  📝 请求: {messages[0]['content']}")
    print(f"  ✅ 响应: {response['content']}")
    print(f"  🤖 模型: {response['model']}")
    print(f"  🔄 尝试次数: {response['attempts']}")
    print(f"  💰 本次费用: {_format_cost(response['cost'])}")

    print(f"\n  🏗️  Builder 配置一览:")
    print(f"      主模型: {engine.primary.model_name}")
    print(f"      备用模型: {[c.model_name for c in engine.fallbacks]}")
    print(f"      Metrics: {'✅ 已启用' if engine.metrics else '❌ 未启用'}")
    print(f"      Cost Tracker: {'✅ 已启用' if engine.cost_tracker else '❌ 未启用'}")

    # 打印费用摘要
    if engine.cost_tracker:
        summary: dict[str, Any] = engine.cost_tracker.get_summary()
        print(f"\n  📈 累计统计: 总调用 {summary['call_count']} 次，"
              f"总费用 {_format_cost(summary['total_cost'])}")

    # 重置，避免影响后续演示
    if engine.metrics:
        engine.metrics.reset()
    if engine.cost_tracker:
        engine.cost_tracker.reset()


def demo_fallback_with_retry() -> None:
    """演示 3: 主模型失败 → 降级到备用模型（含 metrics + cost）。

    展示两个场景：
    1. **重试成功**：主模型前 2 次抛出 ``TransientError``，第 3 次成功
       —— 演示 tenacity 自动重试能力。
    2. **降级切换**：主模型全部 3 次尝试均失败，自动切换到备用模型成功
       —— 演示完整的 Fallback 流程。

    展示点：
    - TransientError 触发自动重试
    - 重试耗尽后自动切换到备用模型
    - metrics 记录每次尝试的模型和状态
    - cost 追踪只计入最终成功的调用
    """
    print("\n🚀 演示 3: Fallback + Retry 完整流程")
    print("=" * 40)

    # ---- 创建共享的 metrics 和 cost_tracker ----
    metrics: LLMMetrics = LLMMetrics()
    cost_tracker: CostTracker = CostTracker()

    # ==================================================================
    # 场景 A: 重试成功 —— 主模型前 2 次失败，第 3 次成功
    # ==================================================================
    _print_separator("📋 场景 A: 主模型重试成功")

    primary_a: MagicMock = _make_mock_client("gpt-4o")
    # side_effect: 前 2 次抛出 TransientError，第 3 次返回成功
    primary_a.chat_completion.side_effect = [
        TransientError("网络超时"),
        TransientError("速率限制"),
        _make_success_response(
            content="重试成功！第 3 次尝试终于连通了。",
            model="gpt-4o",
            usage={"prompt_tokens": 500, "completion_tokens": 200, "total_tokens": 700},
        ),
    ]

    # 配置重试：最多 3 次尝试，指数退避
    engine_a: LLMEngine = LLMEngine(
        primary=primary_a,
        retry_config=RetryConfig(max_attempts=3, backoff="exponential", min_wait=0.01, max_wait=0.1),
        metrics=metrics,
        cost_tracker=cost_tracker,
    )

    messages: list[dict[str, str]] = [{"role": "user", "content": "重试场景测试"}]
    try:
        response_a: dict[str, Any] = engine_a.chat(messages)

        print(f"  📝 请求: {messages[0]['content']}")
        print(f"  ⚠️  第 1 次尝试: TransientError('网络超时') → tenacity 自动重试")
        print(f"  ⚠️  第 2 次尝试: TransientError('速率限制') → tenacity 自动重试")
        print(f"  ✅ 第 3 次尝试: 成功！")
        print(f"  📨 响应: {response_a['content']}")
        print(f"  🤖 最终模型: {response_a['model']}")
        print(f"  🔄 总尝试次数: {response_a['attempts']}")
        print(f"  💰 费用: {_format_cost(response_a['cost'])}")
    except AllModelsExhaustedError as exc:
        print(f"  ❌ 不期望的失败: {exc}")

    # ==================================================================
    # 场景 B: 降级切换 —— 主模型全失败，切换到 DeepSeek 备用模型
    # ==================================================================
    _print_separator("📋 场景 B: 主模型全失败 → 降级到备用模型")

    primary_b: MagicMock = _make_mock_client("gpt-4o")
    # 主模型始终抛出 TransientError（重试耗尽后切换到备用模型）
    primary_b.chat_completion.side_effect = TransientError("主模型持续超时")

    backup_b: MagicMock = _make_mock_client("deepseek-chat")
    backup_b.chat_completion.return_value = _make_success_response(
        content="主模型不可用，已切换到 DeepSeek 备用模型。",
        model="deepseek-chat",
        usage={"prompt_tokens": 300, "completion_tokens": 150, "total_tokens": 450},
    )

    engine_b: LLMEngine = LLMEngine(
        primary=primary_b,
        fallbacks=[backup_b],
        retry_config=RetryConfig(max_attempts=3, backoff="exponential", min_wait=0.01, max_wait=0.1),
        metrics=metrics,
        cost_tracker=cost_tracker,
    )

    messages_b: list[dict[str, str]] = [{"role": "user", "content": "降级场景测试"}]
    try:
        response_b: dict[str, Any] = engine_b.chat(messages_b)

        print(f"  📝 请求: {messages_b[0]['content']}")
        print(f"  ❌ 主模型 gpt-4o: 3 次重试全部失败 (TransientError)")
        print(f"  🔄 自动切换: gpt-4o → deepseek-chat")
        print(f"  ✅ 备用模型成功: {response_b['content']}")
        print(f"  🤖 最终模型: {response_b['model']}")
        print(f"  🔄 总尝试次数: {response_b['attempts']}")
        print(f"  💰 费用: {_format_cost(response_b['cost'])}")
    except AllModelsExhaustedError as exc:
        print(f"  ❌ 所有模型均已耗尽: {exc}")

    # ==================================================================
    # 打印汇总
    # ==================================================================
    _print_separator("📊 汇总统计")
    cost_summary: dict[str, Any] = cost_tracker.get_summary()
    print(f"  累计调用次数: {cost_summary['call_count']}")
    print(f"  累计总费用: {_format_cost(cost_summary['total_cost'])}")
    print(f"  按模型费用: ")
    for model_name, cost in cost_summary["by_model"].items():
        print(f"    - {model_name}: {_format_cost(cost)}")

    print(f"\n  📋 Metrics 摘要 (Prometheus 格式前 500 字符):")
    metrics_text: str = metrics.get_summary()
    print(f"    {metrics_text[:500]}{'...' if len(metrics_text) > 500 else ''}")


def demo_metrics_server() -> None:
    """演示 4: 启动 Prometheus metrics HTTP 端点。

    启动 Prometheus HTTP 服务暴露 ``/metrics`` 端点，做一次额外调用
    产生指标数据，然后打印 metrics 摘要。

    注意：
    - 此函数在脚本末尾调用，确保不阻塞前面的输出
    - ``start_http_server`` 在后台 daemon 线程运行，不会阻塞主线程
    """
    print("\n🚀 演示 4: Prometheus Metrics 服务器")
    print("-" * 40)

    # ---- 启动 Prometheus HTTP 服务器 ----
    start_http_server(9090)
    print("  🌐 Metrics 服务器已启动")
    print("  📍 端点: http://localhost:9090/metrics")
    print("  💡 打开浏览器或运行 `curl http://localhost:9090/metrics` 查看指标")

    # ---- 做一次额外调用以产生指标数据 ----
    print("\n  🔧 产生一条示例指标数据...")
    client: MagicMock = _make_mock_client("gpt-4o")
    client.chat_completion.return_value = _make_success_response(
        content="Metrics 演示调用成功！",
        model="gpt-4o",
        usage={"prompt_tokens": 200, "completion_tokens": 80, "total_tokens": 280},
    )

    metrics: LLMMetrics = LLMMetrics()
    cost_tracker: CostTracker = CostTracker()

    engine: LLMEngine = LLMEngine(
        primary=client,
        retry_config=RetryConfig(max_attempts=1, backoff="fixed", min_wait=0.01),
        metrics=metrics,
        cost_tracker=cost_tracker,
    )

    response: dict[str, Any] = engine.chat(
        [{"role": "user", "content": "metrics demo"}]
    )
    print(f"  ✅ 调用完成: {response['content'][:40]}...")
    print(f"  💰 费用: {_format_cost(response['cost'])}")

    # ---- 打印 metrics 摘要 ----
    print(f"\n  📋 当前 Prometheus 指标:")
    metrics_text: str = metrics.get_summary()
    for line in metrics_text.split("\n"):
        if line.strip() and not line.startswith("#"):
            print(f"    {line}")
        elif line.startswith("# HELP") or line.startswith("# TYPE"):
            print(f"    {line}")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print()
    print("=" * 56)
    print("    🤖 llm_engine 多模型 Fallback + 监控演示")
    print("=" * 56)

    demo_direct_construction()
    demo_builder_pattern()
    demo_fallback_with_retry()
    demo_metrics_server()

    print()
    print("=" * 56)
    print("    ✅ 演示完成！")
    print("    📍 Metrics 端点: http://localhost:9090/metrics")
    print("=" * 56)
    print()
