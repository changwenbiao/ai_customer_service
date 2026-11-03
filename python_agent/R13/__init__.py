"""R13 上下文工程：Step 1（基线观测）。

本目录代码仅做「观测 + 压测 + 基线查询」，不修改 Agent 运行时行为。
Step 2+ 的 token 预算、分层裁剪、槽位、摘要注入放在后续迭代里。

可直接执行的脚本：
    python3 R13/context_stress.py   # 30 轮 SSE 压测 + 事实回忆校验
    python3 R13/token_baseline.py   # 查 token_usage 基线 + CSV 导出
"""
from __future__ import annotations

import time
from pathlib import Path

# R13 代码根（__init__.py 自己所在目录）
R13_DIR = Path(__file__).resolve().parent
AGENT_DIR = R13_DIR.parent
REPO_ROOT = AGENT_DIR.parent


def make_thread_id(tenant: str = "t_demo", tag: str = "ctx") -> str:
    """生成 R13 压测用的线程 ID：`{tenant}:{tag}_{yyyymmdd}_{unix}`。

    每次压测用独立线程，避免与历史压测或线上会话混在一起。
    """
    date = time.strftime("%Y%m%d")
    return f"{tenant}:{tag}_{date}_{int(time.time())}"


# Step 1 压测里用来做事实回忆的黄金样本；脚本会把它们塞进
# 第 3 轮用户消息，然后在第 25 轮检查 AI 回答里是否还能准确复述。
# 如果这就 FAIL，说明模型本身不记事实，Step 5 的槽位机制就成了刚需。
FACT_SAMPLES: dict[str, str] = {
    "ORD-2024-001": "我的订单号是 ORD-2024-001，帮我查一下状态和金额",
}


def check_fact_recall(answer: str, marker: str, label: str = "") -> tuple[bool, str]:
    """检查回答里是否还能看到 marker（如订单号），返回 (是否通过, 摘要)。"""
    hit = marker in (answer or "")
    snippet = (answer or "").replace("\n", " ")[:160]
    tag = label or marker
    return hit, f"{tag}={'PASS' if hit else 'FAIL'} | {snippet}"
