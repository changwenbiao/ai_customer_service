"""
P4-3：Agent 工具函数模块
从 main.py 拆分，包含 search_kb、query_order、transfer_to_human 三个工具。
采用 app 注入模式：在 main.py 中调用 set_app(app) 注入全局实例。
"""
import asyncio
import re
from contextvars import ContextVar

from langchain_core.tools import tool
from langgraph.types import interrupt
from config import ENABLE_CITATION_GUARD, REJECT_REPLY

# Redis 检索缓存暂时关闭：kb_cache_get / kb_cache_set 的调用已注释（重新打开时恢复导入即可）
from cache import kb_singleflight
from config import GO_BUSINESS_API, RETRIEVER_TOP_K, ENABLE_RERANK, RERANK_CANDIDATE_N, \
    RETRIEVAL_SCORE_THRESHOLD, ENABLE_RETRIEVAL_SCORE_THRESHOLD
from db_pool import _pool_conn

# 模块级 app 引用，由 main.py 注入
_app = None
_logger = None
_tenant_var = None
_pending_ticket_var = None
_thread_var = None

# R12：单次请求内累积的引用来源；ContextVar 保证并发请求互不污染
_citations_var: ContextVar[list[dict]] = ContextVar("kb_citations", default=[])


def reset_citations() -> None:
    """每次 /agent/chat 或 /agent/chat/stream 入口调用一次"""
    _citations_var.set([])


def get_citations() -> list[dict]:
    """拷贝一份，避免外部修改污染 ContextVar"""
    return list(_citations_var.get())


def _append_citations(pairs: list) -> list[int]:
    """把本次命中的 (Document, score) 追加进引用列表，返回它们在正文中的编号。

    跨多次 search_kb 调用继续编号（start = 当前已有数量 + 1）。
    同一 doc_id+page_content 重复命中时去重（保留第一次的编号）。
    """
    current = _citations_var.get()
    seen_keys = {(c["doc_id"], c.get("snippet", "")) for c in current}
    new_indices = []
    for doc, score in pairs:
        meta = doc.metadata or {}
        key = (meta.get("doc_id", ""), (doc.page_content or "")[:200])
        if key in seen_keys:
            continue
        seen_keys.add(key)
        idx = len(current) + 1
        current.append({
            "index": idx,
            "doc_id": meta.get("doc_id", ""),
            "title": meta.get("title", ""),
            "section": meta.get("section", ""),
            "snippet": (doc.page_content or "")[:200],
            "score": round(float(score), 4),
        })
        new_indices.append(idx)
    _citations_var.set(current)
    return new_indices


# R12 Step 5：后置引用校验正则（提取回答里所有 [n] 编号）
_CITE_RE = re.compile(r"\[(\d+)\]")


def citation_guard(answer: str, citations: list[dict]) -> tuple[str, bool]:
    """L3 后置引用校验。

    返回 (最终回答, 是否被降级为拒答)。降级条件：
    1. 回答为空 → 拒答
    2. citations 非空（本轮调用过 search_kb）但回答里没有任何 [n] → 拒答
    3. 回答里的 [n] 编号超出 citations 范围 → 拒答（编造来源）
    """
    if not ENABLE_CITATION_GUARD:
        return answer, False

    if not answer:
        return REJECT_REPLY, True

    # 本轮没有检索到任何资料：可能是寒暄、订单查询等非知识库问题 → 放行
    if not citations:
        return answer, False

    valid = {c["index"] for c in citations}
    used = {int(n) for n in _CITE_RE.findall(answer)}

    # 有资料但没有任何引用 → 视为无依据回答
    if not used:
        return REJECT_REPLY, True
    # 引用了不存在的编号 → 视为编造来源
    if not used.issubset(valid):
        return REJECT_REPLY, True
    return answer, False


def set_app(app, logger, tenant_var, pending_ticket_var=None, thread_var=None):
    """注入全局 app、logger、tenant_var（在 main.py 中调用）"""
    global _app, _logger, _tenant_var, _pending_ticket_var, _thread_var
    _app = app
    _logger = logger
    _tenant_var = tenant_var
    _pending_ticket_var = pending_ticket_var
    _thread_var = thread_var


async def _record_kb_hits(tid: str, thread_id: str, query: str, pairs: list) -> None:
    """KB 闭环：检索结果落库（异步旁路，失败不影响对话）。
    pairs 为 store.asimilarity_search_with_score() 返回的 (Document, score) 元组列表：
    - pairs 为空 → 插入一条未命中记录（doc_id 为空，score=0）
    - 非空 → 每个命中文档一条记录，score 为真实相似度（COSINE：越大越相似）
    """
    try:
        if pairs:
            rows = [
                (tid, thread_id, doc.metadata.get("doc_id", ""), doc.metadata.get("title", ""), query,
                 float(score))
                for doc, score in pairs
            ]
        else:
            rows = [(tid, thread_id, "", "", query, 0.0)]
        async with _pool_conn(_app) as conn:
            async with conn.cursor() as cur:
                await cur.executemany(
                    "INSERT INTO kb_hits (tenant_id, thread_id, doc_id, doc_title, query, score) "
                    "VALUES (%s, %s, %s, %s, %s, %s)",
                    rows,
                )
    except Exception as e:
        if _logger:
            _logger.warning(f"[KB Hit] 记录失败 tenant={tid}: {str(e)[:120]}")


@tool
async def search_kb(query: str) -> str:
    """搜索知识库，获取关于平台、课程、政策等官方信息。
    检索结果中的每条资料都带编号 [1][2]...，回答时必须在对应结论后标注同样的编号，
    编号必须来自资料，禁止虚构编号，禁止标注不存在的编号。
    若工具返回"知识库暂未就绪"，说明知识库当前不可用：请如实告知用户"知识库暂时无法访问"，
    不要编造或猜测产品价格、政策、库存等具体信息，并建议用户稍后重试或转接人工客服。
    Args:
        query: 搜索问题或关键词
    """
    tid = _tenant_var.get() or ""
    thread_id = (_thread_var.get() if _thread_var is not None else "") or ""

    # ===== Redis 检索缓存暂时关闭：仅注释读/写缓存调用，结构保留，重新打开时取消注释即可 =====
    # 需求10：检索缓存改为 Redis（多实例共享，失败自动降级为直接回源）
    redis_client = getattr(_app.state, "redis_client", None)
    # cached = await kb_cache_get(redis_client, tid, query)
    # if cached is not None:
    #     _logger.debug(f"[KB Cache HIT] tenant={tid}, query={query[:30]}")
    #     return cached

    # 需求3改造：绕过 retriever 直调 store，拿真实相似度分数落库 kb_hits。
    # retriever.ainvoke() 接口契约固定返回 list[Document]，分数会被丢弃
    # （langchain_core VectorStoreRetriever 源码为证），因此 retriever 链路已停用。
    store = _app.state.tenant_stores.get(tid)
    if store is None:
        return "知识库暂未就绪，请稍后再试。"

    # 需求10：击穿防护——热点 key 失效瞬间，同实例并发请求只放一个回源（进程内锁，不涉 Redis，保留）
    async with kb_singleflight(tid, query):
        # 双检：等锁期间可能已被其他请求回源并写入缓存
        # cached = await kb_cache_get(redis_client, tid, query)
        # if cached is not None:
        #     _logger.debug(f"[KB Cache HIT-after-lock] tenant={tid}, query={query[:30]}")
        #     return cached
        try:
            # rerank 开启时：hybrid 取 top-N 候选，再用 cross-encoder 重排取 top-k
            if ENABLE_RERANK:
                from rerank import rerank_items
                raw_pairs = await store.asimilarity_search_with_score(query, k=RERANK_CANDIDATE_N)
                items = [{"doc_id": d.metadata.get("doc_id", ""),
                          "title": d.metadata.get("title", ""),
                          "content": d.page_content,
                          "score": float(s),
                          "_idx": i} for i, (d, s) in enumerate(raw_pairs)]
                # compute_score 是 CPU 同步推理，放线程池避免阻塞事件循环
                reranked = await asyncio.to_thread(rerank_items, query, items,
                                                   RETRIEVER_TOP_K, RERANK_CANDIDATE_N)
                # 按 rerank 后的顺序，用 _idx 找回原始 Document
                pairs = [(raw_pairs[it["_idx"]][0], it["score"]) for it in reranked]
            else:
                pairs = await store.asimilarity_search_with_score(query, k=RETRIEVER_TOP_K)
        except Exception as e:
            # 检索期降级：embedding 调用失败（key 错误/服务不可用）不抛穿对话，如实告知
            _logger.warning(f"[KB Degrade] tenant={tid} 检索失败，降级为纯LLM: {str(e)[:150]}")
            return "知识库暂未就绪，请稍后再试。"
        # 相似度阈值过滤（ENABLE_RETRIEVAL_SCORE_THRESHOLD 开关默认打开）：低于阈值的 chunk 剔除，
        # 全被滤掉则走"未找到相关信息"分支（模型据此拒答）。
        # 开关打开后阈值始终生效；默认 0 时 dense/hybrid 分数均为正，自然不过滤（rerank 对数似然可为负，会滤负分）。
        if ENABLE_RETRIEVAL_SCORE_THRESHOLD and pairs:
            filtered = [(d, s) for d, s in pairs if s >= RETRIEVAL_SCORE_THRESHOLD]
            if len(filtered) < len(pairs):
                _logger.info(
                    f"[KB Threshold] tenant={tid} query={query[:30]} 阈值{RETRIEVAL_SCORE_THRESHOLD:.3f} "
                    f"过滤 {len(pairs) - len(filtered)}/{len(pairs)} 条")
            pairs = filtered
        docs = [doc for doc, _score in pairs]
        if not docs:
            result = "未找到相关信息，建议转接人工客服。"
        else:
            # ===== R12 命中：编号文本给模型 + 结构化引用给程序 =====
            nums = _append_citations(pairs)
            result = "\n\n".join(
                f"[{n}] {doc.page_content}"
                for n, (doc, _s) in zip(nums, pairs)
            )

        # KB 闭环：检索结果落库（命中明细含真实分数 / 未命中查询）
        await _record_kb_hits(tid, thread_id, query, pairs)

        # await kb_cache_set(redis_client, tid, query, result)
        # _logger.debug(f"[KB Cache MISS] tenant={tid}, query={query[:30]}, cached")
        return result


@tool
async def query_order(order_id: str) -> str:
    """根据订单号查询订单状态和详情，调用Go业务系统获取真实订单。
    Args:
        order_id: 订单号，如 ORD-2024-001
    """
    http_client = _app.state.http_client
    try:
        resp = await http_client.get(f"{GO_BUSINESS_API}/api/business/order?order_id={order_id}", timeout=5)
        resp.raise_for_status()
        data = resp.json()
        return str(data.get("data", "未获取订单数据"))
    except Exception as e:
        _logger.warning(f"query_order error: {e}")
        return f"查询订单业务异常：{str(e)}"


@tool
async def transfer_to_human(reason: str) -> str:
    """将用户转接给人工客服，触发LangGraph HITL中断。
    Args:
        reason: 转接原因
    """
    # 待处理工单已存在（排队中未接管）：不再重复触发中断，正常提示等待，人工接入后自动转交
    if _pending_ticket_var is not None and _pending_ticket_var.get():
        return ("您已提交人工客服，正在等待人工接入。人工客服接入后我会将完整对话转交，"
                "排队期间我仍可继续为您解答问题。")
    approval = interrupt({
        "action": "transfer_to_human",
        "reason": reason,
        "message": f"用户请求转接人工客服，原因：{reason}。是否转接？"
    })
    if approval.get("confirmed"):
        return (f"已为您转接人工客服，预计等待 {approval.get('wait_time', 3)} 分钟。"
                f"工单号：{approval.get('ticket_id', 'N/A')}")
    return "转接已取消，我继续为您服务。"
