#!/usr/bin/env python3
"""R11 离线评测脚本。

两种模式：
  1. retrieval（默认）：调 /api/admin/kb/test-query，只评检索层，不花 LLM 钱。
     指标：Recall@k、MRR、Evidence hit@k（chunk 级）、Top-1 分数分布、阈值扫描。
  2. e2e：调 /api/user/chat，评生成层（答案命中率 / 拒答率 / 误拒率）。
     需要用户账号（见环境变量 EVAL_USERNAME / EVAL_PASSWORD）。

用法：
    # 检索评测（默认 k=5，无需用户账号，不调 LLM）
    python3 python_agent/R11/run_eval.py
    # 指定 k 与阈值
    python3 python_agent/R11/run_eval.py --k 5 --threshold 0.6
    # 只跑指定 id
    python3 python_agent/R11/run_eval.py --ids E001,E002,E086
    # 端到端生成评测（需真实用户账号，会调 LLM）
    EVAL_USERNAME=真实用户名 EVAL_PASSWORD=真实密码 python3 python_agent/R11/run_eval.py --mode e2e

输出：
    docs/my1/eval_report_<mode>.md（默认文件名带检索模式，避免不同模式互相覆盖）
    可用 --report 指定其他路径（此时不自动加后缀）

环境变量：
    KB_IMPORT_BASE_URL   默认 http://localhost:8080
    KB_IMPORT_APP_KEY    默认 demo_key
    KB_IMPORT_APP_SECRET 默认 demo_secret
    EVAL_USERNAME / EVAL_PASSWORD  仅 e2e 模式需要
"""
import argparse
import json
import os
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
EVAL_FILE = EVAL_DIR / "kb_eval.jsonl"
# 报告默认输出到操作手册所在目录（docs/my1/）；可用 --report 覆盖
DEFAULT_REPORT = Path(__file__).resolve().parents[2] / "docs" / "my1" / "eval_report.md"

BASE_URL = os.getenv("KB_IMPORT_BASE_URL", "http://localhost:8080").rstrip("/")
APP_KEY = os.getenv("KB_IMPORT_APP_KEY", "demo_key")
APP_SECRET = os.getenv("KB_IMPORT_APP_SECRET", "demo_secret")


# ----------------------------- HTTP 工具 -----------------------------

class ApiError(RuntimeError):
    """接口错误：携带 HTTP 状态码与解析后的 body（若为 JSON）。"""
    def __init__(self, status: int, body, raw: str):
        self.status = status
        self.body = body  # dict 或 None
        self.raw = raw
        msg = body.get("msg") if isinstance(body, dict) else None
        super().__init__(f"HTTP {status}: {msg or raw[:200]}")


def request(method: str, path: str, token: str = "", body: dict = None, timeout: int = 60):
    url = BASE_URL + path
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = "Bearer " + token
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "ignore")
        try:
            parsed = json.loads(raw)
        except Exception:
            parsed = None
        raise ApiError(e.code, parsed, raw) from None


def login_admin() -> str:
    r = request("POST", "/api/auth/token",
                body={"app_key": APP_KEY, "app_secret": APP_SECRET})
    if r.get("code") != 0:
        raise RuntimeError("app_key 登录失败：" + json.dumps(r, ensure_ascii=False)[:300])
    token = (r.get("data") or {}).get("token")
    if not token:
        raise RuntimeError("登录响应缺少 data.token")
    return token


def login_user(tenant_id: str, username: str, password: str) -> str:
    try:
        r = request("POST", "/api/auth/user-token",
                    body={"tenant_id": tenant_id, "username": username, "password": password})
    except ApiError as e:
        if e.status == 401:
            msg = (e.body or {}).get("msg") if e.body else str(e)
            raise RuntimeError(
                f"用户登录失败：{msg}。请检查 EVAL_USERNAME/EVAL_PASSWORD 是否为真实用户账号；"
                f"若只是评测检索效果，去掉 --mode e2e 用默认 retrieval 模式即可，不需要用户账号。"
            ) from None
        raise
    if r.get("code") != 0:
        raise RuntimeError("用户登录失败：" + (r.get("msg") or json.dumps(r, ensure_ascii=False)[:200]))
    token = (r.get("data") or {}).get("token")
    if not token:
        raise RuntimeError("用户登录响应缺少 data.token")
    return token


def list_docs(token: str) -> list:
    r = request("GET", "/api/admin/kb/documents?page=1&page_size=100", token)
    if r.get("code") != 0:
        raise RuntimeError("查询文档失败：" + json.dumps(r, ensure_ascii=False)[:300])
    return (r.get("data") or {}).get("list") or []


def test_query(token: str, query: str, k: int, mode: str = "hybrid") -> list:
    # rerank 模式首次需加载模型 + CPU 推理，超时设大一些
    t = 180 if mode == "rerank" else 60
    r = request("POST", "/api/admin/kb/test-query", token, {"query": query, "k": k, "mode": mode}, timeout=t)
    if r.get("code") != 0:
        raise RuntimeError(f"test-query 失败 [{query[:20]}]: " + json.dumps(r, ensure_ascii=False)[:200])
    return (r.get("data") or {}).get("items") or []


def user_chat(token: str, user_id: str, content: str) -> str:
    """调用户聊天接口，返回 AI 回复文本。非流式，需用户 token。

    接口契约（go_service/internal/handler/user.go 的 UserChatBody / UserChat）：
      - 请求体：{"user_id": "t_demo:xxx", "msg": "问题"}
        user_id 必须是 {tenant_id}:{uid} 格式，否则中间件返回 403「user_id 不属于当前租户」
      - 响应体：{"code": 0, "msg": "<AI 回复文本>"}（回复在 msg，不是 data）
      - thread 与 user 同值：handler 里 threadID = body.UserID
    """
    r = None
    for attempt in range(3):
        try:
            r = request("POST", "/api/user/chat", token, {"user_id": user_id, "msg": content}, timeout=180)
            break
        except (ApiError, urllib.error.URLError) as e:
            # LLM 偶发超时（agent 侧 OpenAITimeoutError / 网关 5xx）：重试，避免单条用例打断整轮评测
            transient = (isinstance(e, urllib.error.URLError)
                         or getattr(e, "status", 0) >= 500
                         or "timed out" in str(e).lower())
            if not transient or attempt == 2:
                raise
            print(f"    重试 {attempt + 1}/2（{str(e)[:60]}）", flush=True)
            time.sleep(3)
    if r.get("code") != 0:
        msg = r.get("msg") or ""
        raise RuntimeError(f"chat 失败 [{content[:20]}]: {msg[:200]}")
    return (r.get("msg") or "").strip()


# ----------------------------- 指标计算 -----------------------------

_NORM_RE = re.compile(r"[\s*#>|`\-]+")


def norm(s: str) -> str:
    """归一化文本：去空格与 markdown 符号，避免"7个自然日"与"7 个自然日"判不等。"""
    return _NORM_RE.sub("", s or "")


def hit_evidence(content: str, terms: list) -> bool:
    """chunk 内容是否包含任一 evidence_term（归一化后子串匹配）。"""
    if not terms:
        return False
    nc = norm(content)
    return any(norm(t) in nc for t in terms)


def first_hit_rank(titles: list, expected: set) -> int:
    """第一个命中文档的位置（1-based），未命中返回 0。"""
    for i, t in enumerate(titles):
        if t in expected:
            return i + 1
    return 0


def compute_retrieval(rows: list, k: int, items_by_id: dict) -> dict:
    """计算检索层指标。items_by_id: id -> [{title, score, content}]。"""
    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]

    recall_at = {kk: 0 for kk in (1, 3, 5)}
    mrr_sum = 0.0
    evi_at = {3: 0, 5: 0}
    top1_misses = []  # 未命中的用例
    top1_scores_ans = []  # 有答案问题的 Top-1 分数
    top1_scores_noans = []  # 无答案问题的 Top-1 分数

    for r in answerable:
        items = items_by_id.get(r["id"], [])
        titles = [x["title"] for x in items]
        exp = set(r.get("expected_doc_titles") or [])
        # Recall@k：前 k 个结果里只要出现过期望文档就算命中（手册 3.3.2）
        for kk in recall_at:
            if any(t in exp for t in titles[:kk]):
                recall_at[kk] += 1
        # MRR：期望文档第一次出现的位置 rank，取 1/rank；没命中则贡献 0（手册 3.3.3）
        rank = first_hit_rank(titles, exp)
        if rank:
            mrr_sum += 1.0 / rank
        # Top-1 未命中：第一个结果不是期望文档（含"期望文档在 rank 2+"和"完全没命中"两种）
        if rank != 1:
            top1_misses.append(r)
        # Evidence hit@k：前 k 个 chunk 的 content（归一化后）是否包含任一 evidence_terms（手册 3.3.4）
        for kk in evi_at:
            if any(hit_evidence(x["content"], r.get("evidence_terms") or []) for x in items[:kk]):
                evi_at[kk] += 1
        # Top-1 分数：取 rank1 的相似度分数，用于有/无答案分布对比与阈值扫描（手册 3.3.5）
        if items:
            top1_scores_ans.append(items[0]["score"])

    for r in unanswerable:
        items = items_by_id.get(r["id"], [])
        if items:
            top1_scores_noans.append(items[0]["score"])

    n_ans = len(answerable)
    return {
        "n_answerable": n_ans,
        "n_unanswerable": len(unanswerable),
        "recall": {kk: (recall_at[kk] / n_ans if n_ans else 0.0) for kk in recall_at},
        "mrr": (mrr_sum / n_ans if n_ans else 0.0),
        "evidence_hit": {kk: (evi_at[kk] / n_ans if n_ans else 0.0) for kk in evi_at},
        "top1_misses": top1_misses,
        "top1_scores_ans": top1_scores_ans,
        "top1_scores_noans": top1_scores_noans,
    }


def threshold_scan(scores_ans: list, scores_noans: list, steps=21) -> list:
    """扫描阈值 0.0~1.0，返回每档的准确率/误拒率/泄漏率。
    判定规则：Top-1 分数 >= 阈值 → 认为"有答案"，否则"无答案"。
    """
    rows = []
    for i in range(steps):
        th = round(i / (steps - 1), 2)
        # 有答案问题中被误拒（分数 < 阈值）
        false_reject = sum(1 for s in scores_ans if s < th)
        # 无答案问题中泄漏（分数 >= 阈值）
        leak = sum(1 for s in scores_noans if s >= th)
        total = len(scores_ans) + len(scores_noans)
        correct = (len(scores_ans) - false_reject) + (len(scores_noans) - leak)
        acc = correct / total if total else 0.0
        fr_rate = false_reject / len(scores_ans) if scores_ans else 0.0
        leak_rate = leak / len(scores_noans) if scores_noans else 0.0
        rows.append((th, acc, fr_rate, leak_rate))
    return rows


def score_stats(scores: list) -> dict:
    if not scores:
        return {"n": 0, "median": 0, "min": 0, "max": 0}
    return {
        "n": len(scores),
        "median": round(statistics.median(scores), 3),
        "min": round(min(scores), 3),
        "max": round(max(scores), 3),
    }


# ----------------------------- 报告生成 -----------------------------

def fmt_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def gen_report(rows: list, k: int, metrics: dict, threshold: float, retrieval_mode: str = "hybrid") -> str:
    lines = []
    lines.append("# R11 评测报告\n")
    lines.append(f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- 检索模式：**{retrieval_mode}**（hybrid=dense+sparse RRF / dense=仅稠密 / sparse=仅 BM25 / rerank=hybrid+cross-encoder 重排）")
    lines.append(f"- 评测用例：{len(rows)} 条（有答案 {metrics['n_answerable']} / 无答案 {metrics['n_unanswerable']}）")
    lines.append(f"- 检索 k = {k}")
    lines.append("")

    # 1. 检索指标
    lines.append("## 一、检索指标\n")
    r = metrics["recall"]
    lines.append("| 指标 | 结果 |")
    lines.append("|---|---|")
    lines.append(f"| Recall@1 | **{r[1]:.3f}** |")
    lines.append(f"| Recall@3 | **{r[3]:.3f}** |")
    lines.append(f"| Recall@5 | **{r[5]:.3f}** |")
    lines.append(f"| MRR | **{metrics['mrr']:.3f}** |")
    ev = metrics["evidence_hit"]
    lines.append(f"| Evidence hit@3（chunk 级） | **{ev[3]:.3f}** |")
    lines.append(f"| Evidence hit@5（chunk 级） | **{ev[5]:.3f}** |")
    lines.append("")

    # 2. 分类别 Recall@1
    by_cat = defaultdict(list)
    for rr in rows:
        if rr["answerable"]:
            by_cat[rr["category"]].append(rr)
    lines.append("## 二、分类别 Recall@1\n")
    lines.append("| 类别 | 用例数 | Recall@1 |")
    lines.append("|---|---|---|")
    for cat, cat_rows in by_cat.items():
        hit = 0
        for rr in cat_rows:
            items = metrics["_items_by_id"].get(rr["id"], [])
            exp = set(rr.get("expected_doc_titles") or [])
            if items and items[0]["title"] in exp:
                hit += 1
        lines.append(f"| {cat} | {len(cat_rows)} | {hit/len(cat_rows):.3f} |")
    lines.append("")

    # 3. Top-1 未命中清单
    misses = metrics["top1_misses"]
    lines.append(f"## 三、Top-1 未命中（{len(misses)} 条）\n")
    if misses:
        lines.append("| ID | 问题 | 期望文档 | 实际 Top-1 |")
        lines.append("|---|---|---|---|")
        for rr in misses:
            items = metrics["_items_by_id"].get(rr["id"], [])
            actual = items[0]["title"] if items else "(无结果)"
            exp = "、".join(rr.get("expected_doc_titles") or [])
            lines.append(f"| {rr['id']} | {rr['q']} | {exp} | {actual} |")
    else:
        lines.append("无。")
    lines.append("")

    # 4. Top-1 分数分布
    lines.append("## 四、Top-1 分数分布\n")
    sa = score_stats(metrics["top1_scores_ans"])
    sn = score_stats(metrics["top1_scores_noans"])
    lines.append("| 分布 | n | 中位数 | 最小 | 最大 |")
    lines.append("|---|---|---|---|---|")
    lines.append(f"| 有答案 | {sa['n']} | {sa['median']} | {sa['min']} | {sa['max']} |")
    lines.append(f"| 无答案 | {sn['n']} | {sn['median']} | {sn['min']} | {sn['max']} |")
    lines.append("")

    # 5. 阈值扫描
    lines.append("## 五、阈值扫描（Top-1 分数 >= 阈值 视为有答案）\n")
    scan = threshold_scan(metrics["top1_scores_ans"], metrics["top1_scores_noans"])
    lines.append("| 阈值 | 准确率 | 误拒率（有答案被拒） | 泄漏率（无答案放过） |")
    lines.append("|---|---|---|---|")
    for th, acc, fr, leak in scan:
        mark = " ←" if abs(th - threshold) < 0.001 else ""
        lines.append(f"| {th:.2f} | {acc:.3f} | {fmt_pct(fr)} | {fmt_pct(leak)} |{mark}")
    lines.append("")
    lines.append("> 结论：若最优阈值的泄漏率仍偏高，说明单靠向量相似度无法可靠拒答，需叠加 R12 的模型判断 + 引用校验。")
    lines.append("")

    # 6. 逐条检索详情（汇总表格，每行一条用例，方便横向对比）
    items_by_id = metrics["_items_by_id"]
    lines.append(f"## 六、逐条检索详情（共 {len(rows)} 条）\n")
    lines.append("> 状态：✅ Top-1 命中期望 ｜ 🔸 期望在前 k 但非 Top-1 ｜ ❌ 前 k 未命中 ｜ — 无答案用例")
    lines.append("> 命中标记：`*` = 该 rank 的文档属于期望文档；`E` = 该 chunk 内容包含 evidence 关键词")
    lines.append("")
    lines.append("| ID | 状态 | 类别 | 问题 | 期望文档 | Top-1（分数） | Top-2（分数） | Top-3（分数） |")
    lines.append("|---|---|---|---|---|---|---|---|")

    def _cell(it, exp_set, answerable, evi_terms):
        """单个检索结果单元格：文档名(分数) + 命中标记"""
        if not it:
            return "—"
        t = it["title"]
        if len(t) > 12:
            t = t[:10] + "…"
        s = f"{t}({it['score']:.3f})"
        marks = []
        if it["title"] in exp_set:
            marks.append("*")
        if answerable and hit_evidence(it.get("content", ""), evi_terms):
            marks.append("E")
        if marks:
            s += " " + "".join(marks)
        return s

    for rr in rows:
        rid = rr["id"]
        q = rr["q"]
        cat = rr["category"]
        answerable = rr["answerable"]
        exp = set(rr.get("expected_doc_titles") or [])
        evi_terms = rr.get("evidence_terms") or []
        items = items_by_id.get(rid, [])
        top1 = items[0] if items else None
        top1_title = top1["title"] if top1 else "(无)"

        # 命中状态
        if not answerable:
            status = "—"
        elif top1 and top1_title in exp:
            status = "✅"
        elif any(x["title"] in exp for x in items):
            status = "🔸"
        else:
            status = "❌"

        exp_str = "、".join(sorted(exp)) if exp else "—"
        # 问题截断，避免表格过宽
        q_short = q if len(q) <= 25 else q[:23] + "…"
        c1 = _cell(items[0] if len(items) > 0 else None, exp, answerable, evi_terms)
        c2 = _cell(items[1] if len(items) > 1 else None, exp, answerable, evi_terms)
        c3 = _cell(items[2] if len(items) > 2 else None, exp, answerable, evi_terms)
        lines.append(f"| {rid} | {status} | {cat} | {q_short} | {exp_str} | {c1} | {c2} | {c3} |")

    lines.append("")
    return "\n".join(lines)


# ----------------------------- 主流程 -----------------------------

def load_eval(ids: set | None) -> list:
    rows = [json.loads(l) for l in EVAL_FILE.read_text(encoding="utf-8").splitlines() if l.strip()]
    if ids:
        rows = [r for r in rows if r["id"] in ids]
    return rows


def _fmt_dur(sec: float) -> str:
    """秒数转易读格式：65 -> 1m5s。"""
    sec = int(max(0, sec))
    return f"{sec // 60}m{sec % 60:02d}s" if sec >= 60 else f"{sec}s"


def run_retrieval(token: str, rows: list, k: int, retrieval_mode: str = "hybrid") -> dict:
    title_to_doc = {d["title"]: d["doc_id"] for d in list_docs(token)}
    # rerank 模式预热：首次调用会加载 cross-encoder 模型，提前触发避免首条查询超时
    if retrieval_mode == "rerank":
        t0 = time.time()
        print("  预热 reranker 模型（首次加载较慢）...", flush=True)
        test_query(token, "预热", k, mode=retrieval_mode)
        print(f"  预热完成，耗时 {_fmt_dur(time.time() - t0)}", flush=True)
    items_by_id = {}
    t_start = time.time()
    total = len(rows)
    print(f"  共 {total} 条用例，开始检索 ...", flush=True)
    for i, r in enumerate(rows):
        t0 = time.time()
        items = test_query(token, r["q"], k, mode=retrieval_mode)
        items_by_id[r["id"]] = items
        cost = time.time() - t0
        done = i + 1
        elapsed = time.time() - t_start
        avg = elapsed / done
        eta = avg * (total - done)
        top1 = f"{items[0]['title']}({items[0]['score']:.3f})" if items else "无结果"
        print(f"  [{done}/{total}] {r['id']} {cost:5.1f}s 累计{_fmt_dur(elapsed)} "
              f"剩余~{_fmt_dur(eta)} | {r['q'][:18]} -> {top1}", flush=True)
    metrics = compute_retrieval(rows, k, items_by_id)
    metrics["_items_by_id"] = items_by_id  # 报告里分类别/未命中清单要用
    metrics["_title_to_doc"] = title_to_doc
    return metrics


def run_e2e(user_token: str, tenant_id: str, rows: list) -> dict:
    """端到端生成评测：答案命中率 / 拒答率 / 误拒率。
    每条用例开独立 thread_id，避免上下文串扰。
    """
    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    answer_hit = 0
    false_reject = 0  # 有答案却拒答
    rejected = 0  # 无答案正确拒答
    leaked = 0  # 无答案却硬答
    # 拒答信号：明确不知道 / 查不到，或"有所不同 / 建议您查看"这类软拒答（对冲表述）
    # 说明：关键词判定较粗且对措辞敏感（同一问题每次生成的措辞可能不同），
    #       它只能粗分"答了"和"没答"，精确判定留给 R12 的 LLM-as-judge
    REJECT_PATTERNS = (
        "不知道", "查不到", "没有找到", "没有相关", "暂时没有", "无法回答",
        "无法确定", "无法准确", "有所不同", "建议您查看", "已提交人工客服",
    )

    for i, r in enumerate(rows):
        # user_id 必须是 {tenant}:{uid} 格式；每条用例独立 uid，避免上下文串扰
        uid = f"{tenant_id}:eval-{r['id']}-{int(time.time())}"
        ans = user_chat(user_token, uid, r["q"])
        nans = norm(ans)
        is_reject = any(p in ans for p in REJECT_PATTERNS)
        if r["answerable"]:
            if is_reject:
                false_reject += 1
            else:
                mcs = r.get("must_contain") or []
                if mcs and all(norm(m) in nans for m in mcs):
                    answer_hit += 1
        else:
            if is_reject:
                rejected += 1
            else:
                leaked += 1
        if (i + 1) % 10 == 0 or i == len(rows) - 1:
            print(f"  e2e 进度 {i+1}/{len(rows)}", flush=True)

    return {
        "answer_hit_rate": answer_hit / len(answerable) if answerable else 0,
        "false_reject_rate": false_reject / len(answerable) if answerable else 0,
        "reject_rate": rejected / len(unanswerable) if unanswerable else 0,
        "leak_rate": leaked / len(unanswerable) if unanswerable else 0,
        "n_answerable": len(answerable),
        "n_unanswerable": len(unanswerable),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["retrieval", "e2e"], default="retrieval",
                    help="retrieval=只评检索层（默认，无需用户账号）；e2e=端到端生成评测（需用户账号）")
    ap.add_argument("--k", type=int, default=5, help="检索 top_k（retrieval 模式）")
    ap.add_argument("--retrieval-mode", choices=["hybrid", "dense", "sparse", "rerank"], default="hybrid",
                    help="检索模式：hybrid=dense+sparse RRF 融合（默认）；dense=仅稠密向量；"
                         "sparse=仅 BM25；rerank=hybrid 取 top-20 后用 cross-encoder 重排")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="报告中标注的参考阈值（仅用于高亮，不影响指标）")
    ap.add_argument("--ids", default="", help="只跑指定 id，逗号分隔，如 E001,E002")
    ap.add_argument("--report", default=str(DEFAULT_REPORT),
                    help="报告输出路径（默认 docs/my1/eval_report.md）")
    ap.add_argument("--show-cases", type=int, default=0,
                    help="打印前 N 条用例的检索详情（rank/score/title），便于对照手册 3.3 理解指标")
    args = ap.parse_args()

    report_file = Path(args.report)
    # 默认报告文件名自动带检索模式后缀，避免 hybrid/dense/sparse 互相覆盖
    if str(report_file) == str(DEFAULT_REPORT):
        report_file = report_file.with_name(f"eval_report_{args.retrieval_mode}.md")

    ids = set(args.ids.split(",")) if args.ids else None
    rows = load_eval(ids)
    if not rows:
        print("没有匹配的评测用例")
        return 1
    print(f"评测用例 {len(rows)} 条，模式 {args.mode}，检索模式 {args.retrieval_mode}")

    if args.mode == "retrieval":
        token = login_admin()
        print("登录成功，开始检索评测 ...")
        metrics = run_retrieval(token, rows, args.k, args.retrieval_mode)
        if args.show_cases > 0:
            print(f"\n===== 前 {args.show_cases} 条用例的检索详情 =====")
            items_by_id = metrics.get("_items_by_id", {})
            for r in rows[:args.show_cases]:
                items = items_by_id.get(r["id"], [])
                exp = set(r.get("expected_doc_titles") or [])
                print(f"\n[{r['id']}] {r['q']}  期望={list(exp) or '(无答案)'}")
                for i, it in enumerate(items, 1):
                    hit = " ←期望" if it["title"] in exp else ""
                    print(f"  rank{i} score={it['score']:.3f}  title={it['title']}{hit}")
        report = gen_report(rows, args.k, metrics, args.threshold, args.retrieval_mode)
        report_file.parent.mkdir(parents=True, exist_ok=True)
        report_file.write_text(report, encoding="utf-8")
        print(f"\n检索模式={args.retrieval_mode}  Recall@1={metrics['recall'][1]:.3f}  MRR={metrics['mrr']:.3f}  "
              f"Evidence hit@3={metrics['evidence_hit'][3]:.3f}")
        print(f"报告已写入：{report_file}")

    else:  # e2e
        username = os.getenv("EVAL_USERNAME")
        password = os.getenv("EVAL_PASSWORD")
        tenant_id = os.getenv("EVAL_TENANT_ID", "t_demo")
        if not username or not password:
            print("e2e 模式需要真实用户账号：设置环境变量 EVAL_USERNAME / EVAL_PASSWORD 后再运行。")
            print("日常评测检索效果请用默认 retrieval 模式（不加 --mode），无需用户账号、不调 LLM。")
            return 2
        user_token = login_user(tenant_id, username, password)
        print("用户登录成功，开始端到端评测（将调用 LLM，较慢）...")
        m = run_e2e(user_token, tenant_id, rows)
        print(f"\n答案命中率={m['answer_hit_rate']:.3f}  误拒率={m['false_reject_rate']:.3f}  "
              f"拒答率={m['reject_rate']:.3f}  泄漏率={m['leak_rate']:.3f}")
        print("（e2e 指标为控制台输出，如需落报告可自行扩展 gen_e2e_report）")

    return 0


if __name__ == "__main__":
    sys.exit(main())
