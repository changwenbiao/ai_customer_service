"""
P4-3：配置管理模块
从 main.py 拆分，集中管理所有环境变量和常量配置。
"""
import os


def get_env_or_raise(key: str) -> str:
    """获取环境变量，未配置则抛出异常"""
    value = os.getenv(key)
    if not value:
        raise RuntimeError(f"环境变量 {key} 未配置，请检查 .env 文件")
    return value


# ====================== 数据库 & 外部服务 ======================
POSTGRES_DSN = get_env_or_raise("POSTGRES_DSN")
QDRANT_URL = get_env_or_raise("QDRANT_URL")
QDRANT_COLLECTION_NAME = "customer_service_knowledge"
GO_BUSINESS_API = os.getenv("GO_BUSINESS_API", "http://127.0.0.1:8080")

# 内部接口鉴权 Token：调用 Go 后端 /api/agent/* 接口时必须携带
INTERNAL_API_TOKEN = os.getenv("INTERNAL_API_TOKEN", "")

# ====================== 模型配置 ======================
# 【模型配置单一数据源】模型参数统一由后台「模型提供商/租户模型配置」维护（DB），
# 不从 .env 读取模型 Key/Base/Model（旧 AGNES/DASHSCOPE 环境变量迁移桥已删除）。

# 生产环境置为True，屏蔽接口原始异常详情
PROD_MODE = os.getenv("PROD_MODE", "false").lower() == "true"

# 模型 API 配置
# - max_retries: ChatOpenAI 对 429/超时/连接错误做指数退避重试（默认 3 次额外重试，共 4 次尝试）
# - timeout: 单次请求总超时（秒），包括连接、首 token、整个响应。glm-4.5-flash 长对话易超 30s，默认 60
# - stream_chunk_timeout: SSE 每个 chunk 的间隔超时（秒），流到一半卡住时早发现早重试
MODEL_MAX_RETRIES = int(os.getenv("MODEL_MAX_RETRIES", "3"))
MODEL_TIMEOUT = int(os.getenv("MODEL_TIMEOUT", "60"))
MODEL_STREAM_CHUNK_TIMEOUT = int(os.getenv("MODEL_STREAM_CHUNK_TIMEOUT", "30"))
# - max_tokens: 模型单次输出上限（token）。glm-4.5-flash 最大输出 96K，客服场景 1024 够了（省钱 + 防冗长）
MODEL_MAX_TOKENS = int(os.getenv("MODEL_MAX_TOKENS", "1024"))
# - context_window: 模型最大输入 token。glm-4.5-flash 官方 128K（阿里云百炼部署上限 96K）
CONTEXT_WINDOW_TOKENS = int(os.getenv("CONTEXT_WINDOW_TOKENS", "131072"))

# ====================== 会话回收配置 ======================
# 过期天数 / 定时清理间隔(秒, 0=关闭)
SESSION_RETENTION_DAYS = int(os.getenv("SESSION_RETENTION_DAYS", "7"))
SESSION_CLEANUP_INTERVAL = int(os.getenv("SESSION_CLEANUP_INTERVAL", str(6 * 3600)))

TEXT_SPLITTER_CHUNK_SIZE = 200
TEXT_SPLITTER_CHUNK_OVERLAP = 30
# 分片策略（实验 E：标题路径前缀）：是否给每个 chunk 前置「文档标题 > 小节标题」路径
# 开启后 embedding/BM25 都带上结构信息；关闭回退为纯 RecursiveCharacterTextSplitter（基线）。
# 改这个开关必须重新导入知识库（kb_import.py）才会生效，因为分片是索引期动作。
ENABLE_CHUNK_TITLE_PREFIX = os.getenv("ENABLE_CHUNK_TITLE_PREFIX", "true").lower() == "true"
RETRIEVER_TOP_K = 3
# 检索相似度阈值过滤开关：默认开启；关闭后 search_kb 不做任何分数过滤
ENABLE_RETRIEVAL_SCORE_THRESHOLD = os.getenv("ENABLE_RETRIEVAL_SCORE_THRESHOLD", "true").lower() == "true"
# 检索相似度阈值：低于该分数的 chunk 被过滤，全部被滤掉则返回"未找到相关信息"（模型据此拒答）。
# 开关打开后阈值始终生效（无"等于0不过滤"特殊分支），默认 0——dense/hybrid 分数均为正时自然不过滤，
# 但 rerank 链路对数似然可为负，默认 0 也会滤掉负分 chunk。
# 注意分数量纲随检索模式变化：dense 为 cosine 相似度（实测有答案 top-1 约 0.5~0.8），
# hybrid/RRF 为融合分（约 0.5~1.0），rerank 为 cross-encoder 对数似然（可正可负，慎用阈值）。
RETRIEVAL_SCORE_THRESHOLD = float(os.getenv("RETRIEVAL_SCORE_THRESHOLD", "0"))

# ====================== 混合检索配置 ======================
# dense 向量名（OpenAI Embeddings）/ sparse 向量名（fastembed BM25）
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
# sparse 检索模型：Qdrant/bm25 是纯统计 BM25，CPU 友好，支持中文（multilingual tokenizer）
SPARSE_MODEL_NAME = "Qdrant/bm25"
# 融合策略：rff（Reciprocal Rank Fusion），无需分数归一化
HYBRID_FUSION = "rrf"
# hybrid 检索的候选池大小（RRF 前每个检索器各取这么多，融合后取 top-k）
HYBRID_PREFETCH_K = 20

# ====================== Rerank 配置 ======================
# 是否启用 cross-encoder 重排（bge-reranker-v2-m3）。默认开启，构建镜像时已预下载模型
ENABLE_RERANK = os.getenv("ENABLE_RERANK", "true").lower() == "true"
# rerank 候选池大小：先 hybrid 取 top-N，再 rerank 取 top-k
# CPU 单次前向约 3~5s，N 直接决定耗时；10 候选实测 30~50s 会顶到 Go 侧超时，故默认 5
RERANK_CANDIDATE_N = int(os.getenv("RERANK_CANDIDATE_N", "5"))

# ====================== 默认 Prompt ======================
DEFAULT_SYSTEM_PROMPT = """你是智能客服系统 的智能客服"小Q"。
## 你的职责
1. 热情接待每一位用户，用"您"称呼
2. 关于平台信息、课程内容、政策等问题，使用 search_kb 查询
3. 关于订单查询，使用 query_order 工具
4. 遇到无法解决的问题，使用 transfer_to_human 转接人工
## 行为准则
- 回答简洁，每次 2-3 句话
- 不知道的就查询知识库，查不到就诚实告知
- 保持友好亲切的语气
## 服务信息
- 每次回复结尾必须附上一行：智能客服 客服中心 | 工作时间 9:00-18:00"""
