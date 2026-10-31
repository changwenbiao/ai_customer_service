"""
P2-2 / 需求10：知识库检索缓存（Redis 版，多实例共享）

原实现是模块级 dict（进程内）：多副本部署时各存一份，KB 变更后清缓存只清当前实例，
其他实例继续返回旧答案（需求 4 的一致性缺陷在多实例下会被放大）。
改为 Redis 后天然共享，并复用 app.state.redis_client（Redis 是项目强制依赖）。

- key:   cs:kb:cache:{tenant_id}:{md5(query)}
- value: 检索结果文本（可直接拼进 Prompt 的那段）
- TTL:   KB_CACHE_TTL（默认 300s，环境变量可覆盖；Redis 自动过期，无需额外清理任务）
- 失效:  SCAN 前缀匹配 + 批量 DEL（不用 KEYS，避免阻塞 Redis 单线程）
- 击穿:  kb_singleflight 按 key 粒度的进程内锁，同实例同一问题只回源一次

降级策略：Redis 读写异常一律视为「未命中/写入失败」，对话继续回源 Qdrant，不影响可用性。
"""
import asyncio
import hashlib
import logging
import os
from contextlib import asynccontextmanager

logger = logging.getLogger("agent.cache")

# 检索结果缓存有效期（秒）；调试可设 KB_CACHE_TTL=10，生产建议 300
KB_CACHE_TTL = int(os.getenv("KB_CACHE_TTL", "300"))
_KEY_PREFIX = "cs:kb:cache:"

# 击穿防护锁：key -> asyncio.Lock（进程内，不跨实例）
_kb_locks: dict[str, asyncio.Lock] = {}


def kb_cache_key(tenant_id: str, query: str) -> str:
    """生成缓存 key：cs:kb:cache:{tenant_id}:{md5(query)}"""
    q_hash = hashlib.md5(query.strip().lower().encode("utf-8")).hexdigest()
    return f"{_KEY_PREFIX}{tenant_id}:{q_hash}"


async def kb_cache_get(redis_client, tenant_id: str, query: str):
    """读缓存；未命中 / Redis 不可用 / 异常 → None"""
    if redis_client is None:
        return None
    try:
        return await redis_client.get(kb_cache_key(tenant_id, query))
    except Exception as e:
        logger.warning("KB cache get failed tenant=%s: %s", tenant_id, str(e)[:120])
        return None


async def kb_cache_set(redis_client, tenant_id: str, query: str, result: str) -> None:
    """写缓存（SETEX）；Redis 异常静默降级（下次请求重新回源）"""
    if redis_client is None:
        return
    try:
        await redis_client.setex(kb_cache_key(tenant_id, query), KB_CACHE_TTL, result)
    except Exception as e:
        logger.warning("KB cache set failed tenant=%s: %s", tenant_id, str(e)[:120])


@asynccontextmanager
async def kb_singleflight(tenant_id: str, query: str):
    """按 key 粒度的单飞锁（缓存击穿防护）。

    热点 key 失效瞬间，大量并发请求会同时穿透到 Qdrant（每次都调 embedding，烧钱且拖慢）。
    用法是「双检 + 回源」：

        cached = await kb_cache_get(...)
        if cached is not None:
            return cached
        async with kb_singleflight(tid, query):
            cached = await kb_cache_get(...)   # 等锁期间可能已被其他请求写入
            if cached is not None:
                return cached
            ...回源并写缓存...

    注意：锁只在进程内，多实例部署时每个实例最多回源一次（跨实例单飞需分布式锁，
    对本场景收益不大：多一次 embedding 调用 ≠ 数据不一致）。
    """
    key = kb_cache_key(tenant_id, query)
    lock = _kb_locks.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _kb_locks[key] = lock
    async with lock:
        yield
    # 释放后无持有者则移除，避免 _kb_locks 随查询量无限增长。
    # （极小概率竞态：恰好有等待者即将拿到锁时被移除，最坏情况只是多一把锁，不影响正确性）
    if not lock.locked():
        _kb_locks.pop(key, None)


async def clear_kb_cache(redis_client, tenant_id: str | None = None) -> int:
    """失效检索缓存，返回删除条数。

    tenant_id 为空  → 清全部（模型/向量配置变更时用）
    tenant_id 非空  → 只清该租户（KB 文档变更时用，不误伤其他租户）
    key 格式为 "cs:kb:cache:{tenant_id}:{hash}"，按前缀匹配。
    """
    if redis_client is None:
        return 0
    pattern = f"{_KEY_PREFIX}{tenant_id}:*" if tenant_id else f"{_KEY_PREFIX}*"
    deleted = 0
    batch = []
    try:
        # SCAN 游标迭代而非 KEYS：KEYS 会一次性遍历全 keyspace 并阻塞 Redis 单线程
        async for key in redis_client.scan_iter(match=pattern, count=100):
            batch.append(key)
            if len(batch) >= 100:
                deleted += await redis_client.delete(*batch)
                batch.clear()
        if batch:
            deleted += await redis_client.delete(*batch)
    except Exception as e:
        logger.warning("KB cache clear failed tenant=%s: %s", tenant_id or "*", str(e)[:120])
    return deleted
