#!/usr/bin/env python3
"""R13 Step 1：30 轮 SSE 上下文压测脚本。

作用：在不修改任何运行时代码的前提下，先画出一条「每轮 prompt token 如何随轮数膨胀」的基线曲线。
直接调 Agent 的 SSE 接口（/agent/chat/stream），token_usage 由 Agent 自己落库。

用法：
    # 直接跑（Agent 默认在 127.0.0.1:8000 启动）
    python3 python_agent/R13/context_stress.py

    # 指定轮数 / 线程前缀 / Agent 地址
    python3 python_agent/R13/context_stress.py --rounds 30 --tag ctx1 --base-url http://127.0.0.1:8000

    # 跑完自动打印基线曲线（复用 token_baseline 的查询逻辑）
    python3 python_agent/R13/context_stress.py --show-baseline --window 131072

环境变量：
    AGENT_BASE_URL   默认 http://127.0.0.1:8000
    AGENT_TIMEOUT_S  单轮总超时（秒），默认 180

输出：
    控制台：每轮 TTFT / 总耗时 / 答案长度 / 事实回忆
    控制台（--show-baseline）：token_usage 曲线
    可选：--save-csv <path> 把每轮数据写成 CSV
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
import time
from pathlib import Path

import httpx

# 让 R13/__init__.py 可 import（无论脚本从哪里启动都能找到它）
_AGENT_DIR = Path(__file__).resolve().parents[1]
if str(_AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(_AGENT_DIR))

from R13 import FACT_SAMPLES, check_fact_recall, make_thread_id  # noqa: E402


DEFAULT_QUESTIONS = [
    "退换货政策是什么？",
    "发货时效多久？",
    "支持哪些支付方式？",
    "会员权益有哪些？",
    "发票怎么开？",
    "运费怎么算？",
    "可以修改收货地址吗？",
    "售后维修怎么申请？",
]


def build_messages(rounds: int) -> list[str]:
    """按轮数生成测试消息：

    - 第 3 轮埋关键事实（订单号）
    - 第 25 轮回收：检查 AI 是否还能复述
    - 其他轮在 FAQ 里轮流取
    """
    out: list[str] = []
    for i in range(1, rounds + 1):
        if i == 3:
            out.append(FACT_SAMPLES["ORD-2024-001"])
        elif i == 25:
            out.append("刚才第 3 轮那个订单，现在是什么状态？金额多少？")
        else:
            out.append(DEFAULT_QUESTIONS[(i - 1) % len(DEFAULT_QUESTIONS)])
    return out


async def run_one_round(
    client: httpx.AsyncClient,
    base_url: str,
    thread_id: str,
    message: str,
    timeout_s: float,
) -> dict:
    """发起一次 SSE 请求，返回 {round, ttft, total_s, answer_len, events}。"""
    t0 = time.perf_counter()
    first_chunk_s: float | None = None
    full = ""
    events: list[dict] = []
    error: str | None = None
    interrupted = False

    try:
        async with client.stream(
            "POST",
            f"{base_url}/agent/chat/stream",
            json={"thread_id": thread_id, "user_message": message},
            timeout=httpx.Timeout(timeout_s),
        ) as resp:
            async for raw_line in resp.aiter_lines():
                if not raw_line.startswith("data:"):
                    continue
                try:
                    evt = json.loads(raw_line[5:].strip())
                except json.JSONDecodeError:
                    continue
                events.append(evt)
                etype = evt.get("type")
                if etype == "chunk":
                    if first_chunk_s is None:
                        first_chunk_s = time.perf_counter() - t0
                    full += evt.get("content", "")
                elif etype == "end":
                    full = evt.get("content", full)
                    break
                elif etype == "error":
                    error = evt.get("msg", "unknown error")
                    break
                elif etype == "interrupt":
                    interrupted = True
    except httpx.HTTPError as e:
        error = f"HTTPError: {e}"
    except asyncio.TimeoutError:
        error = "timeout"

    total_s = time.perf_counter() - t0
    return {
        "ttft_s": round(first_chunk_s or 0.0, 3),
        "total_s": round(total_s, 3),
        "answer_len": len(full),
        "answer": full,
        "error": error,
        "interrupt": interrupted,
        "events_count": len(events),
    }


async def run_stress(args: argparse.Namespace) -> None:
    base_url = (args.base_url or os.getenv("AGENT_BASE_URL") or "http://127.0.0.1:8000").rstrip("/")
    timeout_s = float(args.timeout or os.getenv("AGENT_TIMEOUT_S", "180"))
    rounds = args.rounds
    thread_id = args.thread_id or make_thread_id(args.tenant, args.tag)
    messages = build_messages(rounds)

    print(f"▶ R13 上下文压测  thread={thread_id}  rounds={rounds}  base={base_url}  timeout={timeout_s}s")
    print()

    rows: list[dict] = []
    fact_marker = "ORD-2024-001"

    async with httpx.AsyncClient() as client:
        for idx, text in enumerate(messages, 1):
            result = await run_one_round(client, base_url, thread_id, text, timeout_s)
            fact_check = ""
            if idx == 25 and not result["error"]:
                ok, snippet = check_fact_recall(result["answer"], fact_marker, "事实回忆")
                result["fact_recall_pass"] = ok
                fact_check = f"  |  {snippet}"

            status = "ERR" if result["error"] else ("INTR" if result["interrupt"] else "OK")
            print(
                f"[{idx:02d}] {status:<4}  ttft={result['ttft_s']:6.3f}s  "
                f"total={result['total_s']:6.3f}s  answer_len={result['answer_len']:4d}  "
                f"q={text[:18]}{fact_check}"
            )
            rows.append({
                "round": idx,
                "ttft_s": result["ttft_s"],
                "total_s": result["total_s"],
                "answer_len": result["answer_len"],
                "error": result["error"] or "",
                "interrupt": result["interrupt"],
                "fact_recall_pass": result.get("fact_recall_pass"),
                "thread_id": thread_id,
                "question": text,
            })

            # 轮间等待：默认 0.3s 太猛容易 429，按需调大
            if idx < rounds:
                await asyncio.sleep(args.interval)

    # 汇总
    valid = [r for r in rows if not r["error"]]
    total_rounds = len(rows)
    failed = total_rounds - len(valid)
    avg_ttft = sum(r["ttft_s"] for r in valid) / len(valid) if valid else 0
    p95_ttft = sorted(r["ttft_s"] for r in valid)[int(len(valid) * 0.95) - 1] if len(valid) >= 20 else (max((r["ttft_s"] for r in valid), default=0))
    fact_pass = [r for r in rows if r.get("fact_recall_pass") is True]

    print()
    print("─" * 60)
    print(f"压测完成  rounds={total_rounds}  failed={failed}  avg_ttft={avg_ttft:.3f}s  p95_ttft={p95_ttft:.3f}s")
    if fact_pass:
        print(f"事实回忆  ORD-2024-001 → PASS（第 25 轮还能看到）")
    else:
        fail_row = next((r for r in rows if r["round"] == 25), None)
        if fail_row and not fail_row["error"]:
            print(f"事实回忆  ORD-2024-001 → FAIL（第 25 轮回答里没有）")
        elif fail_row and fail_row["error"]:
            print(f"事实回忆  第 25 轮请求失败：{fail_row['error']}")

    if args.save_csv:
        _save_csv(args.save_csv, rows)

    # 可选：自动查 token_usage 基线
    if args.show_baseline:
        print()
        print("⏳ 等待 2s 让 Agent 把 token_usage 全部落库...")
        await asyncio.sleep(2)
        await _print_token_baseline(thread_id, args.window)


def _save_csv(path: str, rows: list[dict]) -> None:
    fieldnames = ["round", "ttft_s", "total_s", "answer_len", "error",
                  "interrupt", "fact_recall_pass", "thread_id", "question"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\n📄 每轮数据已写入 {path}")


async def _print_token_baseline(thread_id: str, window: int) -> None:
    """查 token_usage 曲线。DSN 查找顺序与 Agent 完全对齐（main.py 启动顺序）。

    必须 async：调用方 run_stress 已经在一个 event loop 里了，
    嵌套 asyncio.run() 会抛 "cannot be called from a running event loop"。
    """
    from dotenv import load_dotenv

    # 和 main.py 完全一致的加载顺序：根 .env → python_agent/.env → python_agent/.env.local（override）
    REPO_ROOT = _AGENT_DIR.parent
    load_dotenv(REPO_ROOT / ".env")
    load_dotenv(_AGENT_DIR / ".env")
    load_dotenv(_AGENT_DIR / ".env.local", override=True)

    import psycopg  # noqa: F401  # 仅用来探测数据库是否可达

    # 真实 Agent 用 POSTGRES_DSN（config.py），兼容常见别名
    dsn = (
        os.getenv("POSTGRES_DSN")
        or os.getenv("PSYCOPG_DSN")
        or os.getenv("DATABASE_URL")
    )
    if not dsn:
        print(
            "⚠️ 未找到 POSTGRES_DSN（也没 DATABASE_URL / PSYCOPG_DSN）。\n"
            "   Agent 启动时是怎么拿到 DSN 的？可能在 .env.local 或 PyCharm Run Configuration 里，\n"
            "   请把同一个 DSN 写到 python_agent/.env.local 或 shell 导出 POSTGRES_DSN。"
        )
        return

    try:
        async with await psycopg.AsyncConnection.connect(dsn) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT id, created_at, input_tokens, output_tokens, model "
                    "FROM token_usage WHERE thread_id = %s ORDER BY id",
                    (thread_id,),
                )
                rows = await cur.fetchall()
    except Exception as e:
        print(f"⚠️ 查 token_usage 失败：{e}")
        return

    if not rows:
        print(f"⚠️ token_usage 里找不到 thread_id={thread_id} 的记录。")
        return

    print(f"\n📊 token_usage 基线（{len(rows)} 条，window={window}）")
    print(f"{'#':>3}  {'input':>8}  {'output':>7}  {'ratio':>5}  {'model':<24}")
    print("-" * 50)
    peak_input = 0
    for i, (tid, created_at, inp, out, model) in enumerate(rows, 1):
        ratio = round((inp or 0) / window, 2) if window else 0
        peak_input = max(peak_input, inp or 0)
        print(f"{i:>3}  {inp or 0:>8d}  {out or 0:>7d}  {ratio:>5.2f}  {(model or '')[:24]:<24}")
    print("-" * 50)
    print(f"峰值 input_token = {peak_input}  峰值占比 = {peak_input / window:.2f}" if window else f"峰值 input_token = {peak_input}")


def main() -> None:
    p = argparse.ArgumentParser(description="R13 Step 1：30 轮 SSE 上下文压测")
    p.add_argument("--rounds", type=int, default=30, help="压测轮数（默认 30）")
    p.add_argument("--base-url", default=None, help="Agent SSE 地址（默认读 AGENT_BASE_URL 或 http://127.0.0.1:8000）")
    p.add_argument("--thread-id", default=None, help="自定义 thread_id（默认自动生成，每次独立）")
    p.add_argument("--tenant", default="t_demo", help="自动生成 thread_id 时的租户前缀")
    p.add_argument("--tag", default="ctx", help="自动生成 thread_id 时的 tag 前缀")
    p.add_argument("--timeout", type=float, default=None, help="单轮超时秒数（默认 180）")
    p.add_argument("--show-baseline", action="store_true", help="压测完成后自动查 token_usage 曲线")
    p.add_argument("--window", type=int, default=131072, help="模型上下文窗口（算 ratio 用，默认 131072=glm-4.5-flash 128K）")
    p.add_argument("--save-csv", default=None, help="把每轮数据写成 CSV")
    p.add_argument(
        "--interval", type=float, default=1.5,
        help="轮间等待秒数（默认 1.5；小模型/限流严可改 3~5，调大后更稳但更慢）",
    )

    args = p.parse_args()
    asyncio.run(run_stress(args))


if __name__ == "__main__":
    main()
