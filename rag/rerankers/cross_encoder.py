"""基于交叉编码器（CrossEncoder）的重排序器模块。

本模块实现 :class:`CrossEncoderReRanker`，它通过 HuggingFace 的
sentence-transformers 库加载一个交叉编码器（CrossEncoder）模型，对初步检索
得到的候选片段进行精排。

**交叉编码器（cross-encoder）与双塔编码器（bi-encoder）的区别**：

- bi-encoder（如 ``all-MiniLM-L6-v2``）将 query 与文档**分别独立编码**为向量，
  再通过向量相似度（内积 / 余弦）计算相关性。优点是文档向量可离线预计算、
  检索速度快，适合大规模候选集的粗排；缺点是无法捕捉 query 与文档之间的
  细粒度交互，精度相对有限。
- cross-encoder 将 query 与文档**拼接后一起编码**，通过完整的注意力机制直接
  建模二者之间的交叉交互，输出一个标量相关性分数。优点是精度显著更高，适合
  对粗排得到的 Top-K 候选做精排；缺点是每对 (query, document) 都要完整前向
  计算一次，无法预计算，速度较慢，不适合全量候选集。

因此典型用法是：先用 bi-encoder 粗排召回 Top-K，再用本类对 Top-K 做精排。

**懒导入与懒加载设计（重要）**：

- ``sentence_transformers`` 依赖 PyTorch（torch），安装体积巨大，且交叉编码器
  模型需要在**联网环境**下从 HuggingFace 下载。为避免未安装依赖 / 无法联网的
  用户在 import 或构造阶段就报错，本模块**不在模块顶部导入**
  ``sentence_transformers``，而是把导入与模型加载都延迟到**首次真正调用**
  :meth:`rerank` 且候选列表非空时才触发。
- 若在懒加载过程中发生任何异常（含 ``ImportError``、模型下载失败等），都会被
  统一转抛为 :class:`rag.exceptions.RerankError`，并保留原始异常作为 cause。
"""

from __future__ import annotations

from typing import Any

from rag.exceptions import RerankError
from rag.rerankers.base import ReRanker
from vector_store.document import SearchResult


class CrossEncoderReRanker(ReRanker):
    """基于交叉编码器（CrossEncoder）的重排序器。

    默认使用 ``cross-encoder/ms-marco-MiniLM-L-6-v2`` 模型，对粗排候选结果
    按 query 与文档的精细相关性重新排序，返回按相关度降序排列的结果列表。

    设计说明：

    - **懒导入 + 懒加载**：``sentence_transformers`` 的导入与模型实例化都延迟到
      首次真正调用 :meth:`rerank` 且候选列表非空时才触发，避免未安装 torch /
      无法联网下载模型的用户在 import 或构造阶段报错。详见模块级 docstring。
    - **空输入处理**：候选列表 ``results`` 为空时直接返回 ``[]``，不触发模型
      加载。
    - **分数语义**：cross-encoder 输出的相关性分数越高越相关，因此写入
      ``SearchResult.score``（越高越相关）；同时写入 ``distance = -score`` 以
      保持 vector_store 中「距离越低越近」的约定。

    Attributes:
        _model_name: 使用的交叉编码器模型名称（用于内部加载）。
        _model: 懒加载的 CrossEncoder 模型实例；为 ``None`` 表示尚未加载。
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> None:
        """初始化交叉编码器重排序器。

        注意：本构造方法**不会**导入或加载模型（懒加载），因此无需安装
        sentence-transformers 也能安全构造实例。

        Args:
            model_name: 交叉编码器模型名称，默认
                ``"cross-encoder/ms-marco-MiniLM-L-6-v2"``。
        """
        self._model_name: str = model_name
        """使用的交叉编码器模型名称。"""

        self._model: Any | None = None
        """懒加载的 CrossEncoder 模型实例；为 ``None`` 表示尚未加载。"""

    def rerank(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        """对初步检索结果按交叉编码器相关性分数重新排序。

        懒加载交叉编码器模型后，将 ``query`` 与每个候选文档文本拼接成
        ``(query, document)`` pair，交由 ``CrossEncoder.predict`` 一次性打分，
        再按分数降序重排并重写 ``score`` / ``distance`` 字段。

        空输入处理约定：``results`` 为空列表 ``[]`` 时直接返回 ``[]``，不触发
        模型加载。

        Args:
            query: 查询文本（通常为用户的自然语言问题）。
            results: 初步检索得到的候选结果列表，通常按粗排分数降序。

        Returns:
            按交叉编码器相关性分数降序排列的 :class:`SearchResult` 列表，其中
            每个结果的 ``score`` 为 cross-encoder 相关性分（越高越相关）、
            ``distance = -score``（保持「越低越近」语义），``id`` / ``text`` /
            ``metadata`` 沿用原结果。

        Raises:
            RerankError: 懒加载模型失败（如未安装 sentence-transformers、无法
                联网下载模型等）或模型推理异常时抛出。
        """
        if not results:
            return []

        model = self._load_model()

        try:
            pairs: list[list[str]] = [[query, result.text] for result in results]
            scores = model.predict(pairs)

            ranked: list[SearchResult] = sorted(
                (
                    SearchResult(
                        id=result.id,
                        text=result.text,
                        metadata=result.metadata,
                        score=float(score),
                        distance=-float(score),
                    )
                    for result, score in zip(results, scores)
                ),
                key=lambda item: item.score,
                reverse=True,
            )
        except Exception as exc:  # noqa: BLE001 - 统一转抛 RerankError
            # 推理阶段（pairs 构造 / predict / 排序重建）异常统一转抛 RerankError，
            # 避免被 RAGChain 误包装成 RAGChainError，保证调用方能按阶段定位。
            raise RerankError(
                f"交叉编码器推理失败（模型名：{self._model_name}）：请检查输入"
                "文本是否合法，或稍后重试。"
            ) from exc

        return ranked

    def _load_model(self) -> Any:
        """懒加载交叉编码器（CrossEncoder）模型实例。

        首次调用时才执行 ``import sentence_transformers`` 并实例化
        ``CrossEncoder``，随后缓存到 ``self._model`` 复用。若导入或模型下载
        过程中发生任何异常（含 ``ImportError``、网络错误、模型不存在等），
        统一转抛为 :class:`RerankError`，并保留原始异常作为 cause。

        Returns:
            已加载的 CrossEncoder 模型实例。

        Raises:
            RerankError: 未安装 sentence-transformers 或无法联网下载模型时抛出。
        """
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except Exception as exc:  # noqa: BLE001 - 统一转抛 RerankError
                raise RerankError(
                    "交叉编码器重排序器初始化失败：请先安装 sentence-transformers"
                    "（pip install sentence-transformers），并确保处于可联网环境"
                    "以下载交叉编码器模型。"
                ) from exc

            try:
                self._model = CrossEncoder(self._model_name)
            except Exception as exc:  # noqa: BLE001 - 统一转抛 RerankError
                raise RerankError(
                    f"交叉编码器模型加载失败（模型名：{self._model_name}）："
                    "请确认模型名称正确，并确保处于可联网环境以下载模型权重。"
                ) from exc

        return self._model
