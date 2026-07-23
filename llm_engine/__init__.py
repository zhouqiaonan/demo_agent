"""LLM Engine 包入口。

提供 LLM 调用编排、重试逻辑、模型路由切换、费用追踪等核心能力。
当前导出自定义异常体系、重试策略配置、备用切换管理、
调用指标收集、按模型定价的费用追踪以及统一入口 ``LLMEngine``。
"""

from llm_engine.cost import CostTracker
from llm_engine.engine import LLMEngine, LLMEngineBuilder
from llm_engine.exceptions import (
    AllModelsExhaustedError,
    LLMEngineError,
    NonRetryableError,
    TransientError,
)
from llm_engine.fallback import FallbackManager
from llm_engine.metrics import LLMMetrics
from llm_engine.retry import RetryConfig, create_retry_decorator

__all__ = [
    "LLMEngineError",
    "TransientError",
    "NonRetryableError",
    "AllModelsExhaustedError",
    "RetryConfig",
    "create_retry_decorator",
    "FallbackManager",
    "LLMMetrics",
    "CostTracker",
    "LLMEngine",
    "LLMEngineBuilder",
]
