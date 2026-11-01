"""知识库异步索引 worker：消费 Redis 队列，执行分片 → embedding → Qdrant 写入。

- 队列：Redis List `cs:kb:index:queue`，消息 {"doc_id": "...", "tenant_id": "..."}
- 上传/重索引只入队并标记 status=0，本 worker 异步完成向量化后更新 status=1/2
- 幂等：处理前先按 doc_id 删除旧 Qdrant points，重复消费不产生重复向量
- 崩溃兜底：main.py 启动时扫描 status=0 记录重新入队
"""
import asyncio
import json
import logging
import re
import uuid

from langchain_text_splitters import MarkdownHeaderTextSplitter
from qdrant_client.http import models as qmodels

from agent_factory import ensure_tenant_vector, get_tenant_embeddings
from cache import clear_kb_cache
from config import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME, ENABLE_CHUNK_TITLE_PREFIX
from db_ops import collection_name
from db_pool import _pool_conn
from hybrid_search import make_sparse_vector
from routers_kb import KB_INDEX_QUEUE, read_kb_file_text

logger = logging.getLogger("agent.kb_worker")

MAX_TEXT_CHARS = 1_000_000  # 单文档参与分片的文本上限（约 1MB 字符），超长截断防 embedding 超时

# 实验 E：标题路径前缀分片（Contextual Retrieval）
# 1) 先按 markdown 标题切段，保留每段的「文档标题 / 小节标题」元数据
#    （strip_headers=True 去掉原文标题行，标题信息统一由前缀承载，避免内容里重复出现标题）
# 2) 每段再用 RecursiveCharacterTextSplitter（即 app.state.text_splitter）切小块
# 3) 每个 chunk 前置「[文档标题 > 小节标题]」路径，让 dense embedding 和 sparse BM25 都带上结构信息
_HEADER_SPLITTER = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "doc_title"), ("##", "section")],
    strip_headers=True,
)
_HEADING_LINE_RE = re.compile(r"^\s*#.*$", re.MULTILINE)


def split_chunks_with_title_prefix(text: str, title: str, sub_splitter) -> list:
    """标题感知分片 + 标题路径前缀；无标题或切分结果为空时回退为原分片逻辑。"""
    sections = _HEADER_SPLITTER.split_text(text)
    if not sections:
        return sub_splitter.create_documents([text])
    chunks = []
    for sec in sections:
        # 文档标题优先取 markdown 一级标题，缺失时用 DB 里的 title
        doc_title = (sec.metadata.get("doc_title") or title or "").strip()
        section = (sec.metadata.get("section") or "").strip()
        prefix = f"[{doc_title} > {section}]" if section else f"[{doc_title}]"
        for sub in sub_splitter.split_documents([sec]):
            # 只有标题行、没有正文的段（如文档开头的一级标题）不生成 chunk，避免纯标题垃圾分片
            body = _HEADING_LINE_RE.sub("", sub.page_content).strip()
            if not body:
                continue
            sub.page_content = f"{prefix}\n{sub.page_content}"
            sub.metadata["section"] = section
            chunks.append(sub)
    if not chunks:
        return sub_splitter.create_documents([text])
    return chunks


async def _index_one(app, doc_id: str, tenant_id: str) -> None:
    """处理单个文档：分片 → embedding → Qdrant → 更新状态（失败标记 status=2）"""
    # 1) 读 DB 记录（content 列已废弃，全文以共享卷原文件为唯一数据源）
    async with _pool_conn(app) as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT title, file_name FROM kb_documents WHERE doc_id = %s AND tenant_id = %s",
                (doc_id, tenant_id))
            row = await cur.fetchone()
    if not row:
        logger.warning("KB worker: doc not found %s/%s, skip", tenant_id, doc_id)
        return
    title, file_name = row[0], (row[1] or "")
    content = read_kb_file_text(tenant_id, doc_id, file_name).strip()
    if not content:
        await _mark_failed(app, doc_id, tenant_id, "原文件缺失或无法解析，请重新上传")
        return

    try:
        # 2) 标记索引中（status=3，前端进度弹窗可显示"索引中"）
        async with _pool_conn(app) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "UPDATE kb_documents SET status = 3, error_msg = NULL, updated_at = CURRENT_TIMESTAMP "
                    "WHERE doc_id = %s AND tenant_id = %s",
                    (doc_id, tenant_id))

        # 3) 幂等：先删旧向量（防重复消费/重索引残留）
        # 注意：langchain_qdrant 写入时 metadata 嵌套在 payload.metadata 下，过滤 key 必须带前缀
        try:
            app.state.qdrant_client.delete(
                collection_name=collection_name(tenant_id),
                points_selector=qmodels.FilterSelector(filter=qmodels.Filter(must=[
                    qmodels.FieldCondition(key="metadata.doc_id", match=qmodels.MatchValue(value=doc_id))
                ])),
            )
        except Exception:
            pass

        # 4) 分片（超长截断）：实验 E 开启时按「标题切段 + 标题路径前缀」分片，否则走基线切分
        text = content[:MAX_TEXT_CHARS]
        if ENABLE_CHUNK_TITLE_PREFIX:
            chunks = split_chunks_with_title_prefix(text, title, app.state.text_splitter)
        else:
            chunks = app.state.text_splitter.create_documents([text])
        if not chunks:
            await _mark_failed(app, doc_id, tenant_id, "文档分片结果为空")
            return
        texts = [ch.page_content for ch in chunks]
        sections = [ch.metadata.get("section", "") for ch in chunks]
        ids = [str(uuid.uuid4()) for _ in range(len(texts))]

        # 5) 生成 dense + sparse 向量 + 写入 Qdrant（named vectors）
        vs = await ensure_tenant_vector(app, tenant_id)
        if vs is None:
            await _mark_failed(app, doc_id, tenant_id, "向量模型/向量库不可用，索引失败（请检查后台模型配置）")
            return
        # dense 向量：OpenAI Embeddings（异步批量）
        tenant_emb = await get_tenant_embeddings(app, tenant_id)
        dense_vecs = await tenant_emb.aembed_documents(texts)
        # sparse 向量：fastembed Qdrant/bm25（本地 CPU，逐 chunk 生成）
        sparse_vecs = [make_sparse_vector(t) for t in texts]

        col = collection_name(tenant_id)
        points = [
            qmodels.PointStruct(
                id=ids[i],
                vector={
                    DENSE_VECTOR_NAME: dense_vecs[i],
                    SPARSE_VECTOR_NAME: sparse_vecs[i],
                },
                payload={
                    "page_content": texts[i],
                    "metadata": {"doc_id": doc_id, "title": title, "section": sections[i],
                                 "source": "kb", "tenant_id": tenant_id},
                },
            )
            for i in range(len(texts))
        ]
        app.state.qdrant_client.upsert(collection_name=col, points=points)

        # 6) 更新状态
        async with _pool_conn(app) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "UPDATE kb_documents SET chunk_count = %s, status = 1, error_msg = NULL, updated_at = CURRENT_TIMESTAMP "
                    "WHERE doc_id = %s AND tenant_id = %s",
                    (len(texts), doc_id, tenant_id))
        # tenant_retrievers 缓存已停用（retriever 链路废弃：search_kb 直调 store 拿相似度分数，
        # 详见 agent_factory.get_tenant_retriever 注释），索引完成无需再失效 retriever
        # app.state.tenant_retrievers.pop(tenant_id, None)
        # 需求4：新向量此刻生效，清该租户检索缓存（旧分片内容不能再命中）
        # 需求10：缓存已搬到 Redis（多实例共享），失效需 await
        n = await clear_kb_cache(app.state.redis_client, tenant_id)
        logger.info("KB cache invalidated tenant=%s n=%d", tenant_id, n)
        logger.info("KB worker indexed tenant=%s doc_id=%s chunks=%d", tenant_id, doc_id, len(texts))
    except Exception as e:
        logger.exception("KB worker index error doc_id=%s", doc_id)
        await _mark_failed(app, doc_id, tenant_id, str(e))


async def _mark_failed(app, doc_id: str, tenant_id: str, err: str) -> None:
    try:
        async with _pool_conn(app) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "UPDATE kb_documents SET status = 2, error_msg = %s, updated_at = CURRENT_TIMESTAMP "
                    "WHERE doc_id = %s AND tenant_id = %s",
                    (err[:500], doc_id, tenant_id))
    except Exception:
        pass


async def index_worker_loop(app, worker_id: int) -> None:
    """单 worker 循环：BRPOP 消费队列，处理失败不影响后续消息"""
    r = getattr(app.state, "redis_client", None)
    if r is None:
        logger.error("KB worker[%d]: redis client unavailable, worker exits", worker_id)
        return
    logger.info("🔧 KB index worker[%d] started", worker_id)
    while True:
        try:
            item = await r.blpop(KB_INDEX_QUEUE, timeout=5)
            if item is None:
                continue
            try:
                data = json.loads(item[1])
                await _index_one(app, data.get("doc_id", ""), data.get("tenant_id", ""))
            except Exception as e:
                logger.warning("KB worker[%d] task error: %s", worker_id, e)
        except asyncio.CancelledError:
            logger.info("KB worker[%d] cancelled", worker_id)
            raise
        except Exception as e:
            logger.warning("KB worker[%d] loop error: %s", worker_id, e)
            await asyncio.sleep(2)


async def requeue_pending(app) -> int:
    """启动兜底：扫描 status=3（上次处理中崩溃）的记录重新入队；
    注意：status=0 是"待手动索引"（上传后不自动入队），不在此处自动消费。"""
    n = 0
    try:
        async with _pool_conn(app) as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT doc_id, tenant_id FROM kb_documents WHERE status = 3")
                rows = await cur.fetchall()
        r = getattr(app.state, "redis_client", None)
        if r is None:
            return 0
        for doc_id, tenant_id in rows:
            await r.lpush(KB_INDEX_QUEUE, json.dumps({"doc_id": doc_id, "tenant_id": tenant_id}))
            n += 1
        if n:
            logger.info("KB worker: requeued %d interrupted docs (status=3)", n)
    except Exception as e:
        logger.warning("KB requeue pending error: %s", e)
    return n
