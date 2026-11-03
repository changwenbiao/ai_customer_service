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
# 先装 CPU 版 torch（~200MB，比默认 CUDA 版 ~2GB 小很多），再装其余依赖
RUN pip install --no-cache-dir --prefix=/install torch --index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cpu \
    && pip install --no-cache-dir --prefix=/install -r requirements.txt

COPY . .

# 预下载 bge-reranker-v2-m3 模型，避免运行时首次调用才下载
# pip 用 --prefix=/install 安装，需指定 PYTHONPATH 才能 import
RUN PYTHONPATH=/install/lib/python3.13/site-packages python -c \
    "from FlagEmbedding import FlagReranker; FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=False)"

# 预下载 fastembed 的 BM25 模型到固定缓存目录，运行时完全离线
RUN PYTHONPATH=/install/lib/python3.13/site-packages python -c \
    "from fastembed import SparseTextEmbedding; SparseTextEmbedding(model_name='Qdrant/bm25', language='english', disable_stemmer=True)"

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
