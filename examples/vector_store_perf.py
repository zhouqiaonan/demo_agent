"""vector_store 性能演示脚本（独立可运行）。

演示 :class:`ChromaStore` + :class:`HashEmbedding` 的写入吞吐（QPS）与
自监督召回率：构造 1000 条确定性文档，计时 ``add_documents`` 插入并计算
QPS，再抽样 50 条做自监督检索（查询向量 == 文档自身向量），统计召回命中率。

运行方式::

    python examples/vector_store_perf.py

依赖：需先安装 chromadb（``pip install chromadb``）。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from uuid import uuid4

# 将项目根目录加入 sys.path，保证 `python examples/vector_store_perf.py`
# 直接从任意目录运行都能 import 到本项目包（vector_store / tests）。
_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---- 依赖保护：chromadb 未安装时打印友好提示并退出 ----
try:
    import chromadb
except ImportError:
    print("未检测到 chromadb，请先安装：pip install chromadb")
    sys.exit(1)

from tests.fakes import HashEmbedding
from vector_store.chroma_store import ChromaStore
from vector_store.document import Document


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

_DOC_COUNT: int = 1000
"""性能测试的文档总数。"""

_DIMENSION: int = 64
"""嵌入向量的维度。"""

_SAMPLE_STEP: int = 20
"""召回抽样的步长（1000 条中每 20 条抽 1 条，共 50 条）。"""


def _build_documents(n: int) -> list[Document]:
    """构造 n 条确定性的性能测试文档。

    Args:
        n: 文档数量。

    Returns:
        文本为 ``性能测试文档 {i} 号``、id 为 ``perf-{i}``、
        metadata 为 ``{"idx": i}`` 的文档列表。
    """
    return [
        Document(
            text=f"性能测试文档 {i} 号",
            metadata={"idx": i},
            id=f"perf-{i}",
        )
        for i in range(n)
    ]


def main() -> None:
    """执行性能测试主流程：插入计时 + 自监督召回统计。"""
    embedding = HashEmbedding(dimension=_DIMENSION)
    client = chromadb.Client()
    store = ChromaStore(
        embedding=embedding,
        collection_name=f"perf_{uuid4().hex[:8]}",
        client=client,
    )

    docs: list[Document] = _build_documents(_DOC_COUNT)

    # ---- 计时插入 ----
    start: float = time.perf_counter()
    store.add_documents(docs, chunk_size=100, concurrency=4)
    elapsed: float = time.perf_counter() - start
    qps: float = _DOC_COUNT / elapsed

    # ---- 抽样自监督召回 ----
    sample_indices: list[int] = list(range(0, _DOC_COUNT, _SAMPLE_STEP))
    hits: int = 0
    for idx in sample_indices:
        doc: Document = docs[idx]
        results = store.query(doc.text, k=5)
        if doc.id in [r.id for r in results]:
            hits += 1
    recall: float = hits / len(sample_indices)

    # ---- 打印报告 ----
    print(f"插入 {_DOC_COUNT} 条文档耗时：{elapsed:.3f} 秒")
    print(f"QPS：{qps:.2f} 条/秒")
    print(f"自监督召回率：{recall:.3f}（{hits}/{len(sample_indices)}）")


if __name__ == "__main__":
    main()
