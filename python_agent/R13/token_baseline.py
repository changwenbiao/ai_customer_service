#!/usr/bin/env python3
"""R13 Step 1：token_usage 基线查询工具。

独立于压测脚本，用来：
  1) 查某线程的每轮 token 曲线；
  2) 查最近一天所有线程的峰值（定位异常线程）；
  3) 把结果导出成 CSV 方便做图表 / 对比表。

用法：
    # 查刚才那次压测的线程
    python3 python_agent/R13/token_baseline.py --thread-id t_demo:ctx_20261103_123456

    # 查最近 7 天所有线程（找异常）
    python3 python_agent/R13/token_baseline.py --days 7 --min-ratio 0.6

    # 导出成 CSV（配合 --thread-id 或 --days）
    python3 python_agent/R13/token_baseline.py --days 7 --export-csv /tmp/r13_baseline.csv

环境变量：
    DATABASE_URL 或 PSYCOPG_DSN   postgres://user:pass@host:5432/dbname
    R13_WINDOW                    上下文窗口（算 ratio 用，默认 131072）
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# 让 dotenv 能从 python_agent/.env 和 .env.local 里读 DATABASE_URL
_AGENT_DIR = Path(__file__).resolve().parents[1]
if str(_AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(_AGENT_DIR))


async def _query_by_thread(conn, thread_id: str, window: int) -> list[dict]:
    """单线程曲线：每行一条 token_usage 记录，按时间正序。"""
    async with conn.cursor() as cur:
        await cur.execute(
            "SELECT id, created_at, input_tokens, output_tokens, total_tokens, model, cost "
            "FROM token_usage WHERE thread_id = %s ORDER BY id",
            (thread_id,),
        )
        rows = await cur.fetchall()
    return [
        {
            "seq": i + 1,
            "created_at": str(created_at),
            "input_tokens": int(inp or 0),
            "output_tokens": int(out or 0),
            "total_tokens": int(total or 0),
            "model": model or "",
            "cost": float(cost or 0),
            "ratio": round((inp or 0) / window, 3) if window else None,
        }
        for i, (tid, created_at, inp, out, total, model, cost) in enumerate(rows)
    ]


async def _query_recent(conn, days: int, min_ratio: float, window: int) -> list[dict]:
    """最近 N 天所有线程的峰值汇总（聚合）。"""
    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=days)
    ratio_clause = ""
    params: list = [cutoff]
    if min_ratio > 0 and window > 0:
        ratio_clause = "HAVING MAX(input_tokens)::numeric / %s > %s"
        params.extend([window, min_ratio])

    async with conn.cursor() as cur:
        query = (
            "SELECT thread_id, model, count(*) AS rounds, "
            "avg(input_tokens) AS avg_input, max(input_tokens) AS max_input, "
            "sum(cost) AS total_cost "
            "FROM token_usage WHERE created_at > %s "
            "GROUP BY thread_id, model"
        )
        if ratio_clause:
            query += " " + ratio_clause
        query += " ORDER BY max_input DESC LIMIT 200"
        await cur.execute(query, tuple(params))
        rows = await cur.fetchall()

    return [
        {
            "thread_id": tid or "",
            "model": model or "",
            "rounds": int(rounds or 0),
            "avg_input": round(float(avg or 0), 1),
            "max_input": int(mx or 0),
            "peak_ratio": round((mx or 0) / window, 3) if window else None,
            "total_cost": round(float(cost or 0), 4),
        }
        for (tid, model, rounds, avg, mx, cost) in rows
    ]


def _print_thread_curve(rows: list[dict], thread_id: str, window: int) -> None:
    if not rows:
        print(f"⚠️ token_usage 里找不到 thread_id={thread_id} 的记录。")
        return
    print(f"\n📊 线程曲线  thread={thread_id}  rounds={len(rows)}  window={window}")
    print(f"{'#':>3}  {'input':>8}  {'output':>7}  {'ratio':>6}  {'model':<24}  {'cost':>8}")
    print("-" * 62)
    peak = 0
    for r in rows:
        ratio_str = f"{r['ratio'] or 0:>6.3f}" if r["ratio"] is not None else "     -"
        peak = max(peak, r["input_tokens"])
        print(f"{r['seq']:>3}  {r['input_tokens']:>8d}  {r['output_tokens']:>7d}  {ratio_str}  "
              f"{r['model'][:24]:<24}  {r['cost']:>8.4f}")
    print("-" * 62)
    print(f"峰值 input_token = {peak}  峰值占比 = {peak / window:.3f}" if window else f"峰值 = {peak}")


def _print_recent_summary(rows: list[dict], days: int, window: int) -> None:
    if not rows:
        print(f"⚠️ 最近 {days} 天没有符合条件的线程。")
        return
    print(f"\n📈 最近 {days} 天线程峰值汇总  window={window}  top={len(rows)}")
    print(f"{'thread_id':<42}  {'rounds':>6}  {'avg_in':>8}  {'max_in':>8}  {'ratio':>6}  {'model':<18}")
    print("-" * 100)
    for r in rows:
        ratio_str = f"{r['peak_ratio'] or 0:>6.3f}" if r["peak_ratio"] is not None else "     -"
        print(f"{r['thread_id'][:42]:<42}  {r['rounds']:>6d}  {r['avg_input']:>8.1f}  "
              f"{r['max_input']:>8d}  {ratio_str}  {r['model'][:18]:<18}")


def _export_csv(path: str, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\n📄 已导出 {len(rows)} 行 → {path}")


async def main_async(args: argparse.Namespace) -> None:
    from dotenv import load_dotenv

    # 和 main.py 完全一致的加载顺序：根 .env → python_agent/.env → python_agent/.env.local（override）
    REPO_ROOT = _AGENT_DIR.parent
    load_dotenv(REPO_ROOT / ".env")
    load_dotenv(_AGENT_DIR / ".env")
    load_dotenv(_AGENT_DIR / ".env.local", override=True)

    import psycopg  # psycopg 3 的 AsyncConnection / AsyncCursor 在主模块下

    # Agent 实际用 POSTGRES_DSN（config.py L17），兼容常见别名
    dsn = (
        os.getenv("POSTGRES_DSN")
        or os.getenv("PSYCOPG_DSN")
        or os.getenv("DATABASE_URL")
    )
    if not dsn:
        print(
            "❌ 未找到 POSTGRES_DSN（也没 DATABASE_URL / PSYCOPG_DSN）。\n"
            "   Agent 启动时是怎么拿到 DSN 的？请确认 python_agent/.env.local 里有 POSTGRES_DSN，\n"
            "   或 shell 导出一下：export POSTGRES_DSN=postgresql://user:pass@host:5432/dbname"
        )
        sys.exit(1)

    window = args.window or int(os.getenv("R13_WINDOW", "131072"))
    min_ratio = args.min_ratio or 0.0

    async with await psycopg.AsyncConnection.connect(dsn) as conn:
        if args.thread_id:
            rows = await _query_by_thread(conn, args.thread_id, window)
            _print_thread_curve(rows, args.thread_id, window)
            if args.export_csv:
                _export_csv(
                    args.export_csv, rows,
                    ["seq", "created_at", "input_tokens", "output_tokens",
                     "total_tokens", "model", "cost", "ratio"],
                )
        else:
            days = args.days or 7
            rows = await _query_recent(conn, days, min_ratio, window)
            _print_recent_summary(rows, days, window)
            if args.export_csv:
                _export_csv(
                    args.export_csv, rows,
                    ["thread_id", "model", "rounds", "avg_input", "max_input",
                     "peak_ratio", "total_cost"],
                )


def main() -> None:
    p = argparse.ArgumentParser(description="R13 Step 1：token_usage 基线查询")
    p.add_argument("--thread-id", default=None, help="单线程模式：按 thread_id 查每轮曲线")
    p.add_argument("--days", type=int, default=7, help="聚合模式：查最近多少天（默认 7）")
    p.add_argument("--min-ratio", type=float, default=0.0,
                   help="聚合模式：只显示 max_input/window 超过这个比例的线程（默认 0，不过滤）")
    p.add_argument("--window", type=int, default=None,
                   help="上下文窗口（算 ratio 用；默认读环境变量 R13_WINDOW 或 131072=glm-4.5-flash 128K）")
    p.add_argument("--export-csv", default=None, help="把查询结果导出成 CSV")
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
