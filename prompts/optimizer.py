"""提示词优化器模块 — 基于评估集的迭代式提示词自动优化。

提供 EvalCase、OptimizationResult 数据结构，以及 PromptOptimizer 类，
通过 LLM 生成变体、评估、分析改进的循环来优化提示词模板。
"""

from dataclasses import dataclass, field
import re
import json

from prompts.template import PromptTemplate
from function_caller.config import GenerationConfig

# LLM 输出用做模板时的最大允许长度
_MAX_TEMPLATE_LENGTH = 8192

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalCase:
    """评估用例 — 表示一个输入-期望输出对。

    Attributes:
        input: 模版中 ``{{ text }}`` 变量将接收的输入。
        expected: 期望的输出/结果。
        metadata: 可选的额外元数据字典。
    """

    input: str
    expected: str
    metadata: dict = field(default_factory=dict)

    MAX_INPUT_LENGTH = 4096
    MAX_EXPECTED_LENGTH = 2048

    def __post_init__(self):
        if len(self.input) > self.MAX_INPUT_LENGTH:
            raise ValueError(f"EvalCase input exceeds max length of {self.MAX_INPUT_LENGTH} chars")
        if len(self.expected) > self.MAX_EXPECTED_LENGTH:
            raise ValueError(f"EvalCase expected exceeds max length of {self.MAX_EXPECTED_LENGTH} chars")
        if "\x00" in self.input or "\x00" in self.expected:
            raise ValueError("EvalCase fields must not contain null bytes")


@dataclass(frozen=True)
class OptimizationResult:
    """优化结果 — 包含最佳模板、迭代历史和迭代次数。

    Attributes:
        best_template: 在所有迭代中得分最高的提示词模板。
        history: 每次迭代的详细信息列表，每项为 dict。
        iterations: 实际运行的迭代次数（可能因提前停止而少于请求次数）。
    """

    best_template: PromptTemplate
    history: list[dict]
    iterations: int


# ---------------------------------------------------------------------------
# PromptOptimizer
# ---------------------------------------------------------------------------


class PromptOptimizer:
    """评估集驱动的迭代式提示词优化器。

    通过 LLM 生成提示词变体，在评估集上评分，分析结果并改进，
    循环迭代直至收敛或达到最大迭代次数。

    Args:
        client: 实现 ``chat_completion(messages, tools, model, **kwargs)`` 的 LLM 客户端。
        config: GenerationConfig 实例，默认使用 GenerationConfig.code() 预设。
    """

    def __init__(self, client, config=None):
        self.client = client
        self.config = config if config is not None else GenerationConfig.code()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def optimize(
        self,
        seed_template,
        eval_set,
        metric,
        iterations=5,
        variants_per_iter=3,
        template_input_key="text",
    ):
        """执行迭代式提示词优化。

        Args:
            seed_template: 初始 PromptTemplate，作为优化的起点。
            eval_set: EvalCase 列表，用于评估每个变体的质量。
            metric: 评分函数 ``(prediction: str, expected: str) -> float``，
                    返回 0.0 到 1.0 之间的分数。
            iterations: 最大迭代次数，必须 >= 1。
            variants_per_iter: 每次迭代生成的变体数量。
            template_input_key: 模板中接收 ``case.input`` 的变量名，默认 ``"text"``。

        Returns:
            OptimizationResult — 包含最佳模板、历史和迭代次数。

        Raises:
            ValueError: eval_set 为空或 iterations < 1。
        """
        # ---- 验证输入 ----
        if not eval_set:
            raise ValueError("eval_set 不能为空")
        if iterations < 1:
            raise ValueError("iterations 必须 >= 1")
        if variants_per_iter < 1:
            raise ValueError(f"variants_per_iter 必须 >= 1，实际值为 {variants_per_iter}")

        history: list[dict] = []
        current_template = seed_template
        best_overall_template = seed_template
        best_overall_score = -1.0

        for i in range(iterations):
            # ---- 生成变体 ----
            variants = self._generate_variants(
                current_template.template_str, variants_per_iter
            )

            # ---- 评估每个变体 ----
            variant_scores: list[float] = []
            for variant_text in variants:
                score = self._evaluate_variant(variant_text, eval_set, metric, template_input_key)
                variant_scores.append(score)

            # ---- 找到最佳变体 ----
            best_idx = max(range(len(variants)), key=lambda j: variant_scores[j])
            best_score = variant_scores[best_idx]

            # ---- 记录历史 ----
            history.append({
                "iteration": i + 1,
                "variants": [
                    {"text": v, "score": s}
                    for v, s in zip(variants, variant_scores)
                ],
                "best_variant_idx": best_idx,
                "best_score": best_score,
            })

            # ---- 更新全局最佳 ----
            if best_score > best_overall_score:
                best_overall_score = best_score
                best_overall_template = PromptTemplate(variants[best_idx])

            # ---- 提前停止 ----
            if best_score >= 1.0:
                break

            # ---- 分析结果并生成下一轮起点 ----
            if i < iterations - 1:
                next_text = self._analyze_and_improve(
                    variants[best_idx],
                    best_score,
                    eval_set,
                    metric,
                    template_input_key,
                )
                current_template = PromptTemplate(next_text)

        # ---- 返回结果 ----
        return OptimizationResult(
            best_template=best_overall_template,
            history=history,
            iterations=len(history),
        )

    # ------------------------------------------------------------------
    # Private: variant generation
    # ------------------------------------------------------------------

    def _generate_variants(self, template_str: str, count: int) -> list[str]:
        """通过 LLM 生成提示词变体列表。

        发送系统提示要求生成指定数量的变体，并在模板不同的方面给出变化。
        """
        system_prompt = (
            f"你是一个提示词优化器。请基于当前提示词生成 {count} 个变体。"
            "每个变体应在措辞、结构或指令精度上有所不同。"
            "用纯文本列出每个变体，用编号分隔。\n"
            "格式：\n"
            "Variant 1: <变体内容>\n"
            "Variant 2: <变体内容>\n"
            "..."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"当前提示词：\n{template_str}"},
        ]

        response = self.client.chat_completion(
            messages=messages,
            **self.config.to_dict(),
        )

        content = response.get("content", "") or ""
        variants = self._parse_variants(content, count)

        # 如果解析出的变体数不足，用原文补全
        while len(variants) < count:
            variants.append(template_str)

        return variants[:count]

    @staticmethod
    def _parse_variants(raw: str, expected_count: int) -> list[str]:
        """从 LLM 回复中解析变体列表。

        支持的格式：
        - "Variant N: ..." (按行解析)
        - 若解析失败，返回 [raw.strip()]
        """
        # 尝试按 "Variant N:" 模式分割
        lines = raw.strip().split("\n")
        parsed = []

        # 模式 1: "Variant N: <content>"
        for line in lines:
            m = re.match(r"^(?:Variant|变体)\s*\d+\s*[:：]\s*(.+)", line.strip())
            if m:
                parsed.append(m.group(1).strip())

        if parsed:
            return parsed

        # 模式 2: 尝试按编号分割 (1. , 2. , etc)
        numbered_pattern = re.compile(r"^\d+[.、)]\s+")
        current = []
        for line in lines:
            stripped = line.strip()
            if numbered_pattern.match(stripped):
                if current:
                    parsed.append("\n".join(current))
                current = [re.sub(numbered_pattern, "", stripped, count=1)]
            elif stripped:
                current.append(stripped)
        if current:
            parsed.append("\n".join(current))

        if parsed:
            return parsed

        # Fallback: return whole response as single variant
        return [raw.strip()]

    # ------------------------------------------------------------------
    # Private: evaluation
    # ------------------------------------------------------------------

    def _evaluate_variant(
        self,
        variant_text: str,
        eval_set: list[EvalCase],
        metric,
        input_key: str = "text",
    ) -> float:
        """在评估集上评估单个变体，返回平均分数。"""
        if not eval_set:
            return 0.0

        scores: list[float] = []
        for case in eval_set:
            try:
                # 用 variant_text 创建临时模板，传入 case.input 渲染
                temp = PromptTemplate(variant_text)
                prediction = temp.render(**{input_key: case.input})
            except ValueError:
                # 如果缺少变量，分数为 0
                scores.append(0.0)
                continue

            s = metric(prediction, case.expected)
            scores.append(s)

        return sum(scores) / len(scores)

    # ------------------------------------------------------------------
    # Private: analysis & improvement
    # ------------------------------------------------------------------

    def _analyze_and_improve(
        self,
        best_text: str,
        best_score: float,
        eval_set: list[EvalCase],
        metric,
        input_key: str = "text",
    ) -> str:
        """分析评估结果，请求 LLM 给出改进后的提示词。

        将最佳/最差表现样例（带分数）发送给 LLM，请求分析并直接给出改进后的模板。
        """
        # 收集每个 case 的分数
        case_scores = []
        temp = PromptTemplate(best_text)
        for case in eval_set:
            try:
                prediction = temp.render(**{input_key: case.input})
                score = metric(prediction, case.expected)
            except ValueError:
                score = 0.0
            case_scores.append((case, score))

        # 排序：分高的在前
        case_scores.sort(key=lambda x: x[1], reverse=True)

        # 构建分析请求
        best_cases = case_scores[:3]
        worst_cases = case_scores[-3:] if len(case_scores) > 3 else []

        best_str = "\n".join(
            f"- 输入: {c.input!r}, 期望: {c.expected!r}, 分数: {s:.3f}"
            for c, s in best_cases
        )
        worst_str = "\n".join(
            f"- 输入: {c.input!r}, 期望: {c.expected!r}, 分数: {s:.3f}"
            for c, s in worst_cases
        )

        system_prompt = (
            "你是一个提示词优化器。分析评估结果并生成改进后的提示词模板。"
            "你的回复必须直接包含改进后的提示词模板，"
            "用 ``{{ variable }}`` 语法保留变量占位符。"
            "只需给出改进后的模板文本，不要附加额外解释。"
        )

        user_message = (
            f"当前提示词模板：\n{best_text}\n\n"
            f"平均得分：{best_score:.3f}\n\n"
            f"表现出色的样例：\n{best_str}\n\n"
            f"表现不佳的样例：\n{worst_str}\n\n"
            "请分析并直接给出改进后的模板文本（保留变量占位符语法）。"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        response = self.client.chat_completion(
            messages=messages,
            **self.config.to_dict(),
        )

        content = response.get("content", "") or ""
        improved = content.strip()

        # 在使用 LLM 输出作为模板之前进行验证
        if "\x00" in improved:
            improved = improved.replace("\x00", "")
        if len(improved) > _MAX_TEMPLATE_LENGTH:
            improved = improved[:_MAX_TEMPLATE_LENGTH]
        if not improved or len(improved) < 10:
            return best_text  # 回退：拒绝可疑的过短输出

        # 尝试从回复中提取模板文本（移除可能的 Markdown 代码块包装）
        code_block_match = re.search(r"```(?:prompt|text)?\s*\n(.*?)\n```", content, re.DOTALL)
        if code_block_match:
            improved = code_block_match.group(1).strip()

        return improved if improved else best_text
