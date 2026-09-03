"""chunk_size / chunk_overlap A/B 实验脚本（独立可运行）。

实验目的
--------
对比 ``RecursiveTextSplitter`` 在不同 ``chunk_size`` 与 ``chunk_overlap``
组合下对检索召回率的影响，并用 matplotlib 画出召回率曲线，输出到
``examples/chunking_ab.png``。

运行方式::

    python examples/chunking_ab_test.py

依赖：需先安装 chromadb 与 matplotlib
（``pip install chromadb matplotlib``）。

自监督召回率设计
----------------
本实验使用 :class:`tests.fakes.HashEmbedding` 作为确定性伪嵌入后端：它用
SHA-256 把文本映射为确定性向量。**完全相同的文本 → 完全相同向量**；而文本
哪怕只差一个字 → 哈希雪崩导致向量完全不同（余弦相似度趋近 0）。

因此产生了一个真实可观察的趋势：

- **chunk_size 越小（越接近单句）**：切出的 chunk 文本与 query 句子完全一致，
  HashEmbedding 给出相同向量、相似度为 1，检索时"包含该句子的 chunk"必然排
  第一，命中率接近 1。
- **chunk_size 越大（一个 chunk 含多句）**：query 句子只是 chunk 文本的
  一部分，二者的哈希向量完全不同、相似度趋近 0，检索结果近似随机，命中率
  下降到约 ``k / chunk 总数``。

因此 chunk_size 越小、自监督召回率越高，本实验正是要量化并可视化这一趋势。

HashEmbedding 的局限
--------------------
哈希雪崩使「包含自己的 chunk」无法通过语义相似被区分出来，只能反映
「query 与 chunk 文本是否逐字相等」这一硬性条件，与真实语义嵌入模型
（如 OpenAI / sentence-transformers）的行为不同。本实验仅用于展示分块
参数对「自监督可命中性」的机械影响，不能替代真实嵌入模型的评估。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

# 将项目根目录加入 sys.path，保证 `python examples/chunking_ab_test.py`
# 从任意目录运行都能 import 到本项目包（document_processor / vector_store /
# tests）。
_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---- 依赖保护：chromadb 未安装时打印友好提示并退出 ----
try:
    import chromadb
except ImportError:
    print("未检测到 chromadb，请先安装：pip install chromadb")
    sys.exit(1)

# 在 import pyplot 之前切换到非交互的 Agg 后端，避免无显示环境下报错。
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from document_processor import RecursiveTextSplitter
from tests.fakes import HashEmbedding
from vector_store.chroma_store import ChromaStore
from vector_store.document import Document, SearchResult


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

_N_SENTENCES: int = 200
"""合成语料中的句子总数。"""

_DIMENSION: int = 64
"""伪嵌入向量的维度。"""

_TOP_K: int = 3
"""自监督检索时返回的候选数量（命中判定在 top-k 内即算命中）。"""

_PARAM_GRID: list[tuple[int, int]] = [
    (50, 0),
    (100, 0),
    (200, 0),
    (200, 50),
    (400, 0),
    (400, 100),
]
"""待对比的 ``(chunk_size, chunk_overlap)`` 参数网格。"""

_OUTPUT_PNG: Path = Path(__file__).resolve().parent / "chunking_ab.png"
"""召回率曲线图的输出路径（相对脚本所在目录，避免依赖运行时的 cwd）。"""


@dataclass
class _RunResult:
    """单组参数的一次实验运行结果。"""

    chunk_size: int
    """该组的 chunk_size。"""

    chunk_overlap: int
    """该组的 chunk_overlap。"""

    recall: float
    """该组的自监督召回率（命中句子数 / 句子总数）。"""

    chunk_count: int
    """该组切分得到的 chunk 总数。"""


# ---------------------------------------------------------------------------
# 语料构造
# ---------------------------------------------------------------------------


def _build_sentences(n: int) -> list[str]:
    """生成 n 条确定性的合成句子，用于构造实验语料。

    每条句子的形式为
    ``"主题{i % 10}：这是第 {i} 个句子，围绕主题{i % 10}展开的叙述内容。"``
    （``i`` 从 0 到 ``n-1``）。主题按 ``i % 10`` 循环，使语料兼具重复主题与
    局部差异。

    Args:
        n: 句子数量。

    Returns:
        长度为 n 的句子文本列表，其中第 i 个元素对应 ``i`` 号句子。
    """
    return [
        f"主题{i % 10}：这是第 {i} 个句子，围绕主题{i % 10}展开的叙述内容。"
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# 单组参数评估
# ---------------------------------------------------------------------------


def _evaluate(
    chunk_size: int,
    chunk_overlap: int,
    sentences: list[str],
    full_text: str,
) -> _RunResult:
    """对一组 ``(chunk_size, chunk_overlap)`` 执行自监督召回评估。

    评估流程：

    1. 用 ``RecursiveTextSplitter`` 把 ``full_text`` 切分为若干 chunk。
    2. 新建一个内存版 ``ChromaStore``（注入 :class:`HashEmbedding`），
       写入全部 chunk。
    3. 对每个句子 ``s``，先确定"正确答案 chunk"——即文本中包含 ``s`` 的
       第一个 chunk；再用 ``s`` 作为 query 检索 top-``k``，若正确答案 chunk
       的 ``id`` 出现在返回结果里，记为命中。
    4. ``recall = 命中数 / 句子总数``。

    Args:
        chunk_size: 递归分割器的目标 chunk 长度。
        chunk_overlap: 相邻 chunk 之间的重叠字符数。
        sentences: 全部句子列表（用于逐句自监督检索）。
        full_text: 由句子拼接成的全文（作为分割输入）。

    Returns:
        该组参数的实验运行结果。
    """
    splitter: RecursiveTextSplitter = RecursiveTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks: list[Document] = splitter.split_documents([Document(text=full_text)])

    store: ChromaStore = ChromaStore(
        embedding=HashEmbedding(dimension=_DIMENSION),
        collection_name=f"ab_{uuid4().hex[:8]}",
        client=chromadb.Client(),
    )
    store.add_documents(chunks)

    hits: int = 0
    for s in sentences:
        # 找到文本中包含该句子的第一个 chunk 作为"正确答案"。
        correct: Document | None = next(
            (c for c in chunks if s in c.text), None
        )
        if correct is None:
            # 句子未被任何 chunk 完整包含（理论上不应发生），视为未命中。
            continue

        results: list[SearchResult] = store.query(s, k=_TOP_K)
        correct_id: str | None = correct.id
        if correct_id is not None and correct_id in [r.id for r in results]:
            hits += 1

    recall: float = hits / len(sentences)
    return _RunResult(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        recall=recall,
        chunk_count=len(chunks),
    )


# ---------------------------------------------------------------------------
# 结果输出
# ---------------------------------------------------------------------------


def _print_table(results: list[_RunResult]) -> None:
    """打印各参数组的召回率对比表格。"""
    header: str = f"{'chunk_size':>10} {'chunk_overlap':>13} {'chunk 数':>10} {'recall':>10}"
    print("=" * len(header))
    print(header)
    print("=" * len(header))
    for r in results:
        print(
            f"{r.chunk_size:>10} {r.chunk_overlap:>13} "
            f"{r.chunk_count:>10} {r.recall:>10.3f}"
        )
    print("=" * len(header))


def _configure_chinese_font() -> None:
    """尝试把 matplotlib 字体配置为支持中文的字体，避免中文显示为方框。

    依次在系统已注册字体中查找常见的中文字体；若均不存在则退回默认字体，
    仅打印一条提示（图片仍能生成，只是中文可能显示为方框）。
    """
    from matplotlib import font_manager

    preferred: list[str] = [
        "PingFang SC",
        "Hiragino Sans GB",
        "Arial Unicode MS",
        "Heiti SC",
        "Songti SC",
        "STHeiti",
        "SimHei",
        "Microsoft YaHei",
    ]
    available: set[str] = {f.name for f in font_manager.fontManager.ttflist}
    for name in preferred:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name]
            plt.rcParams["axes.unicode_minus"] = False
            return
    print("警告：未找到可用的中文字体，图片中的中文可能显示为方框。")


def _plot(results: list[_RunResult]) -> None:
    """用 matplotlib 画出召回率随 chunk_size 变化的曲线并保存 PNG。

    横轴为 ``chunk_size``，纵轴为 ``recall``；不同的 ``chunk_overlap``
    使用不同的标记 / 颜色连成独立的曲线。

    Args:
        results: 各组参数的实验运行结果列表。
    """
    _configure_chinese_font()

    overlaps: list[int] = sorted({r.chunk_overlap for r in results})
    markers: list[str] = ["o", "s", "^", "D", "v", "*", "P", "X"]

    fig, ax = plt.subplots(figsize=(8, 6))
    for idx, ov in enumerate(overlaps):
        subset: list[_RunResult] = sorted(
            (r for r in results if r.chunk_overlap == ov),
            key=lambda r: r.chunk_size,
        )
        xs: list[int] = [r.chunk_size for r in subset]
        ys: list[float] = [r.recall for r in subset]
        ax.plot(
            xs,
            ys,
            marker=markers[idx % len(markers)],
            linewidth=2,
            markersize=8,
            label=f"chunk_overlap={ov}",
        )

    ax.set_xlabel("chunk_size（字符数）")
    ax.set_ylabel("自监督召回率 recall")
    ax.set_title("不同 chunk_size / chunk_overlap 下的自监督召回率对比")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    ax.set_ylim(bottom=0.0, top=1.05)

    fig.tight_layout()
    fig.savefig(_OUTPUT_PNG, dpi=150)
    plt.close(fig)
    print(f"召回率曲线已保存到：{_OUTPUT_PNG}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def main() -> None:
    """执行 A/B 实验主流程：构造语料 → 遍历参数网格 → 打印表格 → 绘图。"""
    sentences: list[str] = _build_sentences(_N_SENTENCES)
    full_text: str = "\n".join(sentences)

    results: list[_RunResult] = []
    for chunk_size, chunk_overlap in _PARAM_GRID:
        result: _RunResult = _evaluate(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            sentences=sentences,
            full_text=full_text,
        )
        results.append(result)
        print(
            f"chunk_size={chunk_size:>3}, chunk_overlap={chunk_overlap:>3} "
            f"→ chunk 数={result.chunk_count:>3}, recall={result.recall:.3f}"
        )

    print()
    _print_table(results)
    _plot(results)


if __name__ == "__main__":
    main()
