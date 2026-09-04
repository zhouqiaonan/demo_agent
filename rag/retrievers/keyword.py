"""基于 BM25 的关键词检索器实现。

本模块提供 :class:`KeywordRetriever`，是 :class:`Retriever` 抽象接口的一个具体
实现：它手写实现了经典的 BM25（Best Matching 25）排序算法，用字符 bigram 做
中文友好的无依赖分词，把「给定查询返回 Top-K 相关 chunk」的动作转化为对文档
词频统计的纯 Python 计算，不依赖任何外部检索库（如 ``rank_bm25``）或分词库
（如 ``jieba``），便于理解算法原理。

设计要点：

- **无依赖分词**：中文不像英文有空格天然分词，为避免引入重量级分词依赖，采用
  **字符 bigram**（相邻两字成词）作为「词项」单位。bigram 无需词典与训练数据，
  对任意语言文本都通用，且能捕捉中文双字词的大部分局部语义（如「苹果」会
  被整体命中）。
- **BM25 打分**：对查询中的每个 bigram，用 IDF（逆文档频率）加权其词频贡献，
  并用词频饱和参数 ``k1`` 与长度归一化参数 ``b`` 修正极端值，得到文档与查询的
  相关度分数。
- **纯内存索引**：构造时一次性遍历文档，统计每个 bigram 在每个文档中的词频
  （TF）与含该词的文档数（DF），形成倒排式索引，供 ``retrieve`` 阶段快速计算。
- **距离语义映射**：BM25 本身只产出「越大越相关」的分数，没有原生「距离」概念；
  为与 :class:`SearchResult.distance`「越低越近」的约定保持一致，命中结果用
  ``distance = -score`` 填充，使上层可统一按 distance 升序理解结果。

BM25 算法原理简述：

对查询 q 与文档 d，其相关度分数定义为查询中每个词项贡献之和：

    score(d, q) = Σ_{term∈q} IDF(term) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * |d| / avgdl))

其中各部分含义：

- ``tf``：词项 ``term`` 在文档 ``d`` 中出现的次数（词频）。
- ``|d|``：文档 ``d`` 的长度（bigram 数量），``avgdl`` 为全部文档的平均长度。
- ``IDF(term) = ln(1 + (N - df + 0.5) / (df + 0.5))``：逆文档频率，衡量词项的
  区分度——出现在越少文档中的词越稀有，信息量越大，权重越高；出现在几乎所有
  文档中的常见词（如「的」「我们」对应的 bigram）权重趋近于 0。
- ``k1``：词频饱和参数，控制词频对分数的影响上限。词频越高贡献越大，但增益
  递减（饱和），避免某词在文档中重复出现时分数无限增长。
- ``b``：长度归一化参数（``0 <= b <= 1``），控制文档长度对词频的惩罚程度。
  ``b`` 越大，长文档的 tf 被稀释得越厉害，从而避免长文档仅因「词多」而占优。
"""

from __future__ import annotations

import math

from vector_store.document import Document, SearchResult

from rag.exceptions import RetrievalError
from rag.retrievers.base import Retriever


def _bigrams(text: str) -> list[str]:
    """把文本切分为字符 bigram（中文友好的无依赖分词）。

    对文本做首尾去空白后，按相邻两字为一个词项切分：长度为 ``n`` 的文本会产生
    ``n - 1`` 个 bigram。长度小于 2 的文本（含空文本）按特殊情形处理。

    Args:
        text: 待切分的原始文本。

    Returns:
        字符 bigram 列表，顺序与文本中出现的顺序一致。空文本返回空列表，长度
        为 1 的文本返回仅含该单个字符的列表。

    Example:
        >>> _bigrams("苹果好吃")
        ['苹果', '果好', '好吃']
        >>> _bigrams(" ")
        []
        >>> _bigrams("A")
        ['A']
    """
    text = text.strip()
    if len(text) < 2:
        return [text] if text else []
    return [text[i : i + 2] for i in range(len(text) - 1)]


class KeywordRetriever(Retriever):
    """基于 BM25 算法的关键词检索器实现。

    本类是对 :class:`Retriever` 抽象接口的关键词检索实现：在构造时对传入文档
    建立字符 bigram 的词频倒排索引，检索时按 BM25 公式计算每个文档与查询的
    相关度分数，返回分数降序排列的 Top-K 命中结果。

    BM25 是对经典 TF-IDF 的改进：它保留了 IDF 对稀有词的高权重，同时引入
    ``k1``（词频饱和）与 ``b``（长度归一化）两个自由参数，使打分对极端词频与
    极端文档长度更鲁棒，是信息检索领域应用最广泛的关键词排序算法之一。

    Attributes:
        _documents: 原始文档列表，与索引中的 ``doc_idx`` 一一对应。
        _k1: 词频饱和参数，默认 1.5。
        _b: 长度归一化参数，默认 0.75。
        _doc_bigrams: 每个文档的 bigram 列表（``doc_bigrams[i]`` 即第 i 篇
            文档的 bigram）。
        _tf: 词项词频索引（``term -> {doc_idx -> 该文档中 term 的出现次数}``）。
        _df: 词项文档频率索引（``term -> 含该 term 的文档数量``）。
        _doc_len: 每个文档的长度（bigram 数量）。
        _avgdl: 所有文档的平均长度；文档列表为空时为 0.0。
    """

    def __init__(self, documents: list[Document], k1: float = 1.5, b: float = 0.75) -> None:
        """初始化关键词检索器并构建字符 bigram 倒排索引。

        遍历 ``documents``，对每篇文档切分 bigram，统计：
        每篇文档的 bigram 列表、每个词项在各文档中的词频（TF）、每个词项的
        文档频率（DF）、每篇文档的长度与平均长度，供 ``retrieve`` 阶段计算
        BM25 分数。

        Args:
            documents: 待建立索引的文档列表。
            k1: BM25 词频饱和参数，控制词频对分数的贡献上限，默认 1.5。
            b: BM25 长度归一化参数（``0 <= b <= 1``），控制文档长度对词频的
                惩罚程度，默认 0.75。

        Raises:
            RetrievalError: ``k1 <= 0``（会导致除零 / NaN）或 ``b`` 不在
                ``[0.0, 1.0]`` 区间时抛出。
        """
        if k1 <= 0:
            raise RetrievalError(
                f"BM25 词频饱和参数 k1 必须为正数，当前传入 {k1}。"
            )
        if not 0.0 <= b <= 1.0:
            raise RetrievalError(
                f"BM25 长度归一化参数 b 必须在 [0.0, 1.0] 区间内，当前传入 {b}。"
            )

        self._documents: list[Document] = documents
        """原始文档列表，与索引中的 doc_idx 一一对应。"""

        self._k1: float = k1
        """BM25 词频饱和参数。"""

        self._b: float = b
        """BM25 长度归一化参数。"""

        self._doc_bigrams: list[list[str]] = []
        """每个文档的 bigram 列表。"""

        self._tf: dict[str, dict[int, int]] = {}
        """词项词频索引：term -> {doc_idx -> 词频}。"""

        self._df: dict[str, int] = {}
        """词项文档频率索引：term -> 含该词项的文档数。"""

        self._doc_len: list[int] = []
        """每个文档的长度（bigram 数量）。"""

        for doc_idx, document in enumerate(documents):
            doc_bigrams = _bigrams(document.text)
            self._doc_bigrams.append(doc_bigrams)
            self._doc_len.append(len(doc_bigrams))

            # 统计本单篇文档内的词频（去重后再累加 DF，保证每词每文档只计一次）。
            local_tf: dict[str, int] = {}
            for term in doc_bigrams:
                local_tf[term] = local_tf.get(term, 0) + 1

            for term, freq in local_tf.items():
                self._tf.setdefault(term, {})[doc_idx] = freq
                self._df[term] = self._df.get(term, 0) + 1

        if self._doc_len:
            self._avgdl: float = sum(self._doc_len) / len(self._doc_len)
            """所有文档的平均长度（bigram 数量）。"""
        else:
            self._avgdl: float = 0.0
            """文档列表为空时平均长度置为 0.0。"""

    def retrieve(self, query: str, k: int = 5) -> list[SearchResult]:
        """根据查询文本计算 BM25 分数，返回最相关的 ``k`` 条结果。

        流程：

        1. 若索引为空（构造时未传入文档），直接返回空列表。
        2. 对查询文本切分 bigram 并去重（BM25 对每个词项只计一次贡献）。
        3. 逐文档累加每个词项的 ``IDF * tf 饱和项`` 得到 BM25 分数。
        4. 按分数降序排序，仅保留分数大于 0 的文档（与查询无任何共同 bigram
           的文档分数为 0，不应作为命中返回），取前 ``k`` 个。
        5. 每个命中构造 :class:`SearchResult`，其中 ``distance = -score``，
           以维持「distance 越低越近」的语义一致性。

        说明：
            BM25 本身只定义「越大越相关」的分数，没有原生距离概念；本实现用
            ``distance = -score`` 把分数映射为单调反向的伪距离，便于上层与
            向量检索（距离越低越近）统一理解。

        Args:
            query: 查询文本（通常为用户的自然语言问题）。
            k: 返回的最相关结果数量，默认 5。

        Returns:
            按 BM25 分数降序排列的 :class:`SearchResult` 列表（分数越高越靠前）。
            若索引为空或所有文档分数均为 0，返回空列表。

        Raises:
            RetrievalError: ``k`` 非正数时抛出（k 必须为正整数）。
        """
        if k <= 0:
            raise RetrievalError(f"检索数量 k 必须为正整数，当前传入 {k}。")

        if not self._documents:
            return []

        # 查询词项去重：BM25 对每个 distinct term 只计一次。
        q_terms: set[str] = set(_bigrams(query))
        if not q_terms:
            return []

        n: int = len(self._documents)

        # 预计算每个查询词项的 IDF，避免在文档循环内重复计算。
        term_idf: dict[str, float] = {}
        for term in q_terms:
            df = self._df.get(term, 0)
            # 词项从未在任何文档中出现时 df 为 0，IDF 为 ln(1 + (N + 0.5)/0.5)，
            # 但因其在文档中词频也必为 0，不会贡献分数；这里仍按公式计算保持一致。
            term_idf[term] = math.log(1.0 + (n - df + 0.5) / (df + 0.5))

        k1: float = self._k1
        b: float = self._b
        avgdl: float = self._avgdl

        scored: list[tuple[float, int]] = []
        for doc_idx in range(n):
            doc_len = self._doc_len[doc_idx]
            # 长度归一化因子：b=0 时不考虑长度，b=1 时完全按长度比例归一化。
            length_norm = 1.0 - b + b * (doc_len / avgdl) if avgdl else 1.0

            score = 0.0
            for term in q_terms:
                tf = self._tf.get(term, {}).get(doc_idx, 0)
                if tf == 0:
                    continue
                # tf 饱和项：tf 越大贡献越大，但增益递减并趋于 (k1 + 1)。
                tf_component = (tf * (k1 + 1.0)) / (tf + k1 * length_norm)
                score += term_idf[term] * tf_component

            if score > 0.0:
                scored.append((score, doc_idx))

        if not scored:
            return []

        # 按分数降序排序；分数相同时保持文档原始顺序（稳定排序）。
        scored.sort(key=lambda item: item[0], reverse=True)

        results: list[SearchResult] = []
        for score, doc_idx in scored[:k]:
            document = self._documents[doc_idx]
            results.append(
                SearchResult(
                    id=document.id or "",
                    text=document.text,
                    metadata=document.metadata,
                    score=score,
                    distance=-score,
                )
            )
        return results
