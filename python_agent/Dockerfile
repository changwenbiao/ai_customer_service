# syntax=docker/dockerfile:1.4
# -------------------------- 构建阶段 builder --------------------------
FROM python:3.13-slim AS builder

# 模型下载源：默认走国内镜像，可用 --build-arg HF_ENDPOINT=https://huggingface.co 覆盖
ARG HF_ENDPOINT=https://hf-mirror.com
ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TZ=Asia/Shanghai \
    HF_HOME=/opt/huggingface \
    HF_ENDPOINT=${HF_ENDPOINT} \
    FASTEMBED_CACHE_PATH=/opt/fastembed

WORKDIR /build

# 仅builder安装tzdata，这里现在已经不需要复制时区文件到运行阶段，如果你后续想保留复制时区逻辑可以保留这行
RUN apt-get update && apt-get install -y --no-install-recommends tzdata \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# pip 缓存挂载：依赖层失效时仍可复用已下载的 wheel，避免重复下载
# 先装 CPU 版 torch（~200MB，比默认 CUDA 版 ~2GB 小很多），再装其余依赖
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    pip install --prefix=/install torch --index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cpu \
    && pip install --prefix=/install -r requirements.txt

# 预下载 bge-reranker-v2-m3 模型：先落到 BuildKit 缓存，再复制进镜像层
# 这样即使依赖层变化，也不会重新走网络下载
RUN --mount=type=cache,target=/opt/hf-cache,sharing=locked \
    HF_HOME=/opt/hf-cache PYTHONPATH=/install/lib/python3.13/site-packages python -c \
    "from FlagEmbedding import FlagReranker; FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=False)" \
    && mkdir -p /opt/huggingface \
    && cp -a /opt/hf-cache/. /opt/huggingface/

# 预下载 fastembed 的 BM25 模型到固定缓存目录，运行时完全离线
RUN --mount=type=cache,target=/opt/fastembed-cache,sharing=locked \
    FASTEMBED_CACHE_PATH=/opt/fastembed-cache PYTHONPATH=/install/lib/python3.13/site-packages python -c \
    "from fastembed import SparseTextEmbedding; SparseTextEmbedding(model_name='Qdrant/bm25', language='english', disable_stemmer=True)" \
    && mkdir -p /opt/fastembed \
    && cp -a /opt/fastembed-cache/. /opt/fastembed/

# 源码放最后：改代码不会让上面的依赖和模型层失效
COPY . .

# -------------------------- 运行阶段 runtime（最终镜像） --------------------------
FROM python:3.13-slim

# 运行阶段不继承 builder 的 ENV；两个模型的缓存均已预下载，运行时强制离线
ARG HF_ENDPOINT=https://hf-mirror.com
ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TZ=Asia/Shanghai \
    HF_HOME=/opt/huggingface \
    HF_ENDPOINT=${HF_ENDPOINT} \
    FASTEMBED_CACHE_PATH=/opt/fastembed \
    HF_HUB_OFFLINE=1

WORKDIR /app

# 复制python依赖包
COPY --from=builder /install /usr/local
# 复制项目源码
COPY --from=builder /build /app
# 复制预下载的模型缓存（reranker + BM25）
COPY --from=builder /opt/huggingface /opt/huggingface
COPY --from=builder /opt/fastembed /opt/fastembed

EXPOSE 8000
CMD ["python","main.py"]
