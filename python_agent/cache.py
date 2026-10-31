"""
P4-3：知识库检索缓存模块
从 main.py 拆分，提供 Qdrant 检索结果的内存缓存。
缓存 key：tenant_id + query(MD5)，TTL 5 分钟。
"""
import hashlib
import time as _time

# 检索结果缓存有效期（秒）；调试用短 TTL，生产建议 300
KB_CACHE_TTL = 10

# key: "tenant_id:query_hash", value: (expire_time, result)
_kb_cache = {}


def _kb_cache_key(tenant_id: str, query: str) -> str:
    """生成缓存key：tenant_id + query的MD5哈希"""
    q_hash = hashlib.md5(query.strip().lower().encode('utf-8')).hexdigest()
    return f"{tenant_id}:{q_hash}"


def _kb_cache_get(tenant_id: str, query: str):
    """获取缓存，过期返回None"""
    key = _kb_cache_key(tenant_id, query)
    item = _kb_cache.get(key)
    if item and item[0] > _time.time():
        return item[1]
    if item:
        del _kb_cache[key]  # 清理过期缓存
    return None


def _kb_cache_set(tenant_id: str, query: str, result: str):
    """设置缓存"""
    key = _kb_cache_key(tenant_id, query)
    _kb_cache[key] = (_time.time() + KB_CACHE_TTL, result)


def _kb_cache_cleanup():
    """清理所有过期缓存（定期调用）"""
    now = _time.time()
    expired = [k for k, v in _kb_cache.items() if v[0] <= now]
    for k in expired:
        del _kb_cache[k]
    return len(expired)


def clear_kb_cache(tenant_id: str | None = None):
    """清空知识库检索结果缓存。
    tenant_id 为 None → 全清（模型/向量配置变更时用）；
    指定 tenant_id → 只清该租户（KB 文档变更时用，不误伤其他租户）。
    key 格式为 "tenant_id:md5(query)"，按前缀匹配。返回清理条数。
    """
    if tenant_id is None:
        n = len(_kb_cache)
        _kb_cache.clear()
        return n
    prefix = f"{tenant_id}:"
    keys = [k for k in _kb_cache if k.startswith(prefix)]
    for k in keys:
        del _kb_cache[k]
    return len(keys)
