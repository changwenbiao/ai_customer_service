"""混合检索模块：dense（OpenAI API Embeddings）+ sparse（fastembed Qdrant/bm25），Qdrant RRF 融合。

设计要点：
- dense 向量继续用平台/租户配置的 OpenAIEmbeddings（API 调用，维度由配置决定）
- sparse 向量用 fastembed 的 Qdrant/bm25 模型（本地 CPU 推理，纯统计 BM25，支持中文 multilingual tokenizer）
- collection 用 named vectors：{"dense": dense_vec, "sparse": sparse_vec}
- 检索用 qdrant_client.query_points + FusionQuery(RRF) + 两个 Prefetch
- HybridStore 接口兼容 QdrantVectorStore 的 asimilarity_search_with_score，便于替换
"""
import logging
from typing import Optional

import jieba
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from langchain_core.documents import Document

from config import (
    SPARSE_MODEL_NAME, DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME,
    HYBRID_PREFETCH_K,
)

logger = logging.getLogger("agent.hybrid")

# sparse 模型进程内单例（BM25 模型无状态，所有租户共享）
_sparse_model: Optional[SparseTextEmbedding] = None


def _tokenize_for_bm25(text: str) -> str:
    """BM25 前的中文预分词。

    fastembed 的 SimpleTokenizer 只按空白/标点切词，中文无空格会被当成一个 token。
    用 jieba 把中文切成词并用空格连接，让 SimpleTokenizer 能正确拆分。
    索引和查询必须走同一套分词，否则 token 对不上。
    """
    return " ".join(jieba.cut(text))


def get_sparse_model() -> SparseTextEmbedding:
    """懒加载 sparse BM25 模型。首次调用时下载/加载模型到内存。

    - disable_stemmer=True：SnowballStemmer 不支持中文，且 jieba 已切词，无需再词干提取。
    - language 任意即可（disable_stemmer 时不加载语言相关 stopwords/stemmer）。
    """
    global _sparse_model
    if _sparse_model is None:
        logger.info("⏳ loading sparse model: %s", SPARSE_MODEL_NAME)
        _sparse_model = SparseTextEmbedding(
            model_name=SPARSE_MODEL_NAME,
            language="english",
            disable_stemmer=True,
        )
        logger.info("✅ sparse model loaded: %s", SPARSE_MODEL_NAME)
    return _sparse_model


def make_sparse_vector(text: str) -> qmodels.SparseVector:
    """用 BM25 模型把文本转成 Qdrant SparseVector（indices + values）。

    Qdrant/bm25 由 Qdrant 服务端计算 IDF，客户端只需输出词频稀疏向量。
    中文先经 jieba 分词再喂给 BM25（见 _tokenize_for_bm25）。
    """
    model = get_sparse_model()
    tokenized = _tokenize_for_bm25(text)
    result = list(model.embed([tokenized]))
    sv = result[0]
    return qmodels.SparseVector(indices=list(sv.indices), values=list(sv.values))


class HybridStore:
    """混合检索 Store，接口与 QdrantVectorStore.asimilarity_search_with_score 对齐。

    mode:
      - "hybrid"：dense + sparse 两路 RRF 融合（默认）
      - "dense"  ：仅稠密向量检索
      - "sparse" ：仅 BM25 稀疏向量检索
    """

    def __init__(self, qdrant_client: QdrantClient, collection_name: str, embeddings,
                 dense_name: str = DENSE_VECTOR_NAME, sparse_name: str = SPARSE_VECTOR_NAME):
        self.client = qdrant_client
        self.collection_name = collection_name
        self.embeddings = embeddings
        self.dense_name = dense_name
        self.sparse_name = sparse_name

    async def asimilarity_search_with_score(self, query: str, k: int = 3, mode: str = "hybrid"):
        """返回 [(Document, score), ...]，score 为 RRF 融合分（hybrid）或原始相似度。"""
        # 1) dense 向量（OpenAI API，异步）
        dense_vec = await self.embeddings.aembed_query(query)

        # 2) sparse 向量（本地 CPU，同步但很快）
        sparse_vec = make_sparse_vector(query)

        # 3) 构造 Prefetch 列表
        prefetch = []
        if mode in ("hybrid", "dense"):
            prefetch.append(qmodels.Prefetch(
                query=qmodels.NearestQuery(nearest=list(dense_vec)),
                using=self.dense_name,
                limit=HYBRID_PREFETCH_K,
            ))
        if mode in ("hybrid", "sparse"):
            prefetch.append(qmodels.Prefetch(
                query=qmodels.NearestQuery(nearest=sparse_vec),
                using=self.sparse_name,
                limit=HYBRID_PREFETCH_K,
            ))

        # 4) 单路直接查（拿真实相似度分数），多路用 RRF 融合
        # 注意：单路若也套 FusionQuery(RRF)，Qdrant 会把分数归一化成 rank 倒数
        #       （rank1→0.5, rank2→0.333...），导致 Top-1 分数恒为 0.5，阈值扫描失效。
        if mode == "dense":
            resp = self.client.query_points(
                collection_name=self.collection_name,
                query=qmodels.NearestQuery(nearest=list(dense_vec)),
                using=self.dense_name,
                limit=k,
                with_payload=True,
            )
        elif mode == "sparse":
            resp = self.client.query_points(
                collection_name=self.collection_name,
                query=sparse_vec,
                using=self.sparse_name,
                limit=k,
                with_payload=True,
            )
        else:  # hybrid
            resp = self.client.query_points(
                collection_name=self.collection_name,
                query=qmodels.FusionQuery(fusion=qmodels.Fusion.RRF),
                prefetch=prefetch,
                limit=k,
                with_payload=True,
            )

        # 5) 转成 (Document, score) 列表，与 QdrantVectorStore 接口对齐
        results = []
        for point in resp.points:
            payload = point.payload or {}
            metadata = payload.get("metadata") or {}
            doc = Document(
                page_content=payload.get("page_content", ""),
                metadata=metadata,
            )
            results.append((doc, float(point.score)))
        return results
