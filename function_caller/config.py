"""生成配置模块 — 提供 GenerationConfig 数据类及常用预设。"""

from dataclasses import dataclass, field


@dataclass
class GenerationConfig:
    """LLM 生成参数配置。

    封装了 OpenAI 兼容 API 的采样参数，并提供三个常用预设配置。
    """

    temperature: float = 0.7
    """温度参数，控制输出的随机性 (0.0–2.0)。"""

    top_p: float = 1.0
    """核采样参数，控制输出的多样性 (0.0–1.0)。"""

    frequency_penalty: float = 0.0
    """频率惩罚，降低重复词的概率 (-2.0–2.0)。"""

    presence_penalty: float = 0.0
    """存在惩罚，鼓励谈论新话题 (-2.0–2.0)。"""

    max_tokens: int | None = None
    """最大生成 token 数。"""

    stop: list[str] | None = None
    """停止序列列表。"""

    # ------------------------------------------------------------------
    # 预设配置
    # ------------------------------------------------------------------

    @classmethod
    def code(cls) -> "GenerationConfig":
        """确定性代码生成预设 — 低温度、低核采样，输出稳定可控。

        适用于：代码补全、代码生成、翻译、结构化输出。
        """
        return cls(temperature=0.0, top_p=0.1)

    @classmethod
    def chat(cls) -> "GenerationConfig":
        """平衡对话预设 — 中等温度、全核采样，兼具连贯性与多样性。

        适用于：日常对话、问答、文本摘要。
        """
        return cls(temperature=0.7, top_p=1.0)

    @classmethod
    def creative(cls) -> "GenerationConfig":
        """创意写作预设 — 高温度、稍紧核采样、添加惩罚项，鼓励原创性。

        适用于：故事创作、头脑风暴、营销文案。
        """
        return cls(
            temperature=1.0,
            top_p=0.95,
            frequency_penalty=0.3,
            presence_penalty=0.1,
        )

    # ------------------------------------------------------------------
    # 序列化
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """将配置转为字典，仅包含非 None 值，便于 **kwargs 透传。"""
        result: dict = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
        }
        if self.max_tokens is not None:
            result["max_tokens"] = self.max_tokens
        if self.stop is not None:
            result["stop"] = self.stop
        return result

    # ------------------------------------------------------------------
    # 展示
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """展示配置摘要，若匹配预设则附带名称。"""
        # 尝试匹配预设名称
        preset_name = self._match_preset()
        parts = [
            f"temperature={self.temperature}",
            f"top_p={self.top_p}",
        ]
        if self.frequency_penalty != 0.0:
            parts.append(f"frequency_penalty={self.frequency_penalty}")
        if self.presence_penalty != 0.0:
            parts.append(f"presence_penalty={self.presence_penalty}")
        if self.max_tokens is not None:
            parts.append(f"max_tokens={self.max_tokens}")
        if self.stop is not None:
            parts.append(f"stop={self.stop}")

        label = f"[{preset_name}] " if preset_name else ""
        return f"GenerationConfig.{preset_name or 'custom'}({', '.join(parts)})"

    def _match_preset(self) -> str:
        """检查当前配置是否匹配某个预设。"""
        presets = {
            "code": self.code(),
            "chat": self.chat(),
            "creative": self.creative(),
        }
        for name, preset in presets.items():
            if self == preset:
                return name
        return ""
