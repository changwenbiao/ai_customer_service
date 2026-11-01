"""Cross-encoder rerank：bge-reranker-v2-m3。

对 dense/hybrid 返回的 top-N 候选重新打分排序，取 top-k 返回。
模型由 FlagEmbedding 提供，首次加载会从 HuggingFace 下载 ~1.2GB。
"""
import logging

from config import RERANK_CANDIDATE_N

logger = logging.getLogger("agent.rerank")

_reranker = None
RERANK_MODEL_NAME = "BAAI/bge-reranker-v2-m3"  # 中文 rerank 标杆，多语言


def get_reranker():
    """懒加载 reranker 单例。失败返回 None（调用方降级为不重排）。"""
    global _reranker
    if _reranker is None:
        try:
            from FlagEmbedding import FlagReranker
            # use_fp16=True 仅在有 GPU 时加速；纯 CPU 必须设 False，否则可能报错或更慢
            _reranker = FlagReranker(RERANK_MODEL_NAME, use_fp16=False)
            logger.info("✅ reranker loaded: %s", RERANK_MODEL_NAME)
        except Exception as e:
            logger.warning("⚠️ reranker 不可用，降级为不重排: %s", str(e)[:150])
    return _reranker


def rerank_items(query: str, items: list, top_k: int, candidate_n: int = RERANK_CANDIDATE_N) -> list:
    """对检索结果重排。

    Args:
        query: 用户问题
        items: 检索返回的 [{doc_id, title, content, score}]，已按原分数降序
        top_k: 最终返回条数
        candidate_n: 只对前 candidate_n 个重排（cross-encoder 是 O(N)，全量太慢）

    Returns:
        重排后的 top_k 条，score 替换为 cross-encoder 分数
    """
    rk = get_reranker()
    if rk is None or not items:
        return items[:top_k]

    candidates = items[:candidate_n]
    pairs = [[query, it["content"]] for it in candidates]
    # compute_score 输入 list[pair] 返回 list[float]（对数似然，越大越相关）
    scores = rk.compute_score(pairs)
    if not isinstance(scores, list):
        scores = [scores]
    for it, sc in zip(candidates, scores):
        it["score"] = float(sc)
    candidates.sort(key=lambda x: -x["score"])
    return candidates[:top_k]
