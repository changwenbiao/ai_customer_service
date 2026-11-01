#!/usr/bin/env python3
"""R11 知识库一键重置 + 导入脚本（只依赖标准库）。

用途：
    把 kb_docs/ 下的 markdown 作为知识库源文件，批量导入指定租户，
    并可反复执行（默认先清空该租户全部文档，再导入 + 触发索引）。

用法：
    python3 python_agent/R11/kb_import.py              # 清空 + 重新导入 + 等待索引完成
    python3 python_agent/R11/kb_import.py --keep       # 保留已有文档，只增量导入
    python3 python_agent/R11/kb_import.py --dry-run    # 只打印将要做的事，不发请求
    python3 python_agent/R11/kb_import.py --only 01    # 只导入文件名以 01 开头的文档

环境变量：
    KB_IMPORT_BASE_URL   默认 http://localhost:8080
    KB_IMPORT_APP_KEY    默认 demo_key
    KB_IMPORT_APP_SECRET 默认 demo_secret
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
DOCS_DIR = EVAL_DIR / "kb_docs"
MANIFEST = EVAL_DIR / "kb_manifest.json"

BASE_URL = os.getenv("KB_IMPORT_BASE_URL", "http://localhost:8080").rstrip("/")
APP_KEY = os.getenv("KB_IMPORT_APP_KEY", "demo_key")
APP_SECRET = os.getenv("KB_IMPORT_APP_SECRET", "demo_secret")


def request(method: str, path: str, token: str = "", body: dict = None):
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
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "ignore")
        raise RuntimeError(f"{method} {path} -> HTTP {e.code}: {raw[:300]}") from None


def login() -> str:
    r = request("POST", "/api/auth/token", body={"app_key": APP_KEY, "app_secret": APP_SECRET})
    if r.get("code") != 0:
        raise RuntimeError("登录失败：" + json.dumps(r, ensure_ascii=False))
    token = (r.get("data") or {}).get("token")
    if not token:
        raise RuntimeError("登录响应缺少 data.token：" + json.dumps(r, ensure_ascii=False)[:300])
    return token


def list_docs(token: str) -> list:
    r = request("GET", "/api/admin/kb/documents?page=1&page_size=100", token)
    if r.get("code") != 0:
        raise RuntimeError("查询文档失败：" + json.dumps(r, ensure_ascii=False)[:300])
    return (r.get("data") or {}).get("list") or []


def delete_doc(token: str, doc_id: str, title: str) -> None:
    r = request("DELETE", "/api/admin/kb/documents/" + doc_id, token)
    if r.get("code") != 0:
        raise RuntimeError(f"删除 {title} 失败：" + json.dumps(r, ensure_ascii=False)[:200])


def upload_doc(token: str, title: str, category: str, content: str) -> str:
    r = request("POST", "/api/admin/kb/upload", token,
                {"title": title, "category": category, "content": content})
    if r.get("code") != 0:
        raise RuntimeError(f"上传 {title} 失败：" + json.dumps(r, ensure_ascii=False)[:300])
    return (r.get("data") or {}).get("doc_id", "")


def reindex_doc(token: str, doc_id: str, title: str) -> None:
    r = request("POST", f"/api/admin/kb/documents/{doc_id}/reindex", token, {})
    if r.get("code") != 0:
        raise RuntimeError(f"提交索引 {title} 失败：" + json.dumps(r, ensure_ascii=False)[:300])


def wait_indexed(token: str, expected: int, timeout: int = 180) -> bool:
    deadline = time.time() + timeout
    last = ""
    while time.time() < deadline:
        docs = list_docs(token)
        done = [d for d in docs if d.get("status") == 1]
        failed = [d for d in docs if d.get("status") == 2]
        msg = f"  索引进度 {len(done)}/{expected}，失败 {len(failed)}"
        if msg != last:
            print(msg, flush=True)
            last = msg
        if failed:
            for d in failed:
                print(f"  ✗ {d.get('title')} 索引失败：{d.get('error_msg')}")
            return False
        if len(done) >= expected:
            return True
        time.sleep(3)
    print("  ! 等待索引超时，请到管理端知识库页面查看 status 与 error_msg")
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true", help="不清空已有文档，只增量导入")
    ap.add_argument("--dry-run", action="store_true", help="只打印计划，不实际请求")
    ap.add_argument("--only", default="", help="只导入文件名以该前缀开头的文档")
    args = ap.parse_args()

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    docs = manifest["docs"]
    if args.only:
        docs = [d for d in docs if d["file"].startswith(args.only)]
    if not docs:
        print("没有匹配的文档")
        return 1

    print(f"目标服务 {BASE_URL}，租户 {manifest.get('tenant')}，文档 {len(docs)} 篇")
    if args.dry_run:
        for d in docs:
            p = DOCS_DIR / d["file"]
            print(f"  - {d['title']} [{d['category']}] {p.stat().st_size} 字节")
        return 0

    token = login()
    print("登录成功")

    if not args.keep:
        old = list_docs(token)
        print(f"清空已有文档 {len(old)} 篇")
        for d in old:
            delete_doc(token, d["doc_id"], d.get("title", ""))
            print(f"  - 已删除 {d.get('title')}")

    for d in docs:
        content = (DOCS_DIR / d["file"]).read_text(encoding="utf-8")
        doc_id = upload_doc(token, d["title"], d["category"], content)
        print(f"  + 已上传 {d['title']} ({len(content)} 字符) doc_id={doc_id}")
        reindex_doc(token, doc_id, d["title"])

    print("等待异步索引完成 ...")
    ok = wait_indexed(token, len(docs))
    if not ok:
        return 2

    docs_now = list_docs(token)
    total_chunks = sum(x.get("chunk_count") or 0 for x in docs_now)
    print(f"完成：{len(docs_now)} 篇文档，共 {total_chunks} 个 chunk")
    for x in docs_now:
        print(f"  ✓ {x.get('title')} | {x.get('category')} | chunks={x.get('chunk_count')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
