"""
数据飞轮 Step 2 脚本

从 data/evaluation/qa_batch_N.json 读取 Claude 验证后的 QA 对，
逐条调用生产系统 /chat_eval，收集 answer + contexts，
输出 data/evaluation/ragas_input_batch_N.json，
拉回本地交给 Claude 做 RAGAS 评估与诊断。

用法：
  python flywheel_run.py --batch 1
"""

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path

import requests

sys.stdout.reconfigure(encoding="utf-8")

# ── 配置 ──────────────────────────────────────────────────────────
_ROOT        = Path(__file__).parent
DATA_DIR     = _ROOT / "data"
CHAT_API_URL = "http://localhost:8000"
CHAT_TIMEOUT = 240
CHAT_DELAY   = 1.5


# /chat_eval 仅限 ops / admin：从环境变量 WOW_OPS_TOKEN 读取 ops 账号令牌（服务关闭认证时可不设）
OPS_HEADERS = {"Authorization": f"Bearer {os.environ['WOW_OPS_TOKEN']}"} if os.getenv("WOW_OPS_TOKEN") else {}


# ── 路径工具 ──────────────────────────────────────────────────────

def qa_path(batch: int) -> Path:
    return DATA_DIR / "evaluation" / f"qa_batch_{batch}.json"

def ragas_input_path(batch: int) -> Path:
    return DATA_DIR / "evaluation" / f"ragas_input_batch_{batch}.json"


# ── 核心逻辑 ──────────────────────────────────────────────────────

def call_chat_eval(question: str, idx: int, max_retries: int = 2) -> dict:
    payload = {
        "user_input": question,
        "session_id": f"flywheel_{idx}_{uuid.uuid4().hex[:8]}",
        "user_id":    "flywheel_eval",
    }
    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(
                f"{CHAT_API_URL}/chat_eval",
                json=payload,
                headers=OPS_HEADERS,
                timeout=CHAT_TIMEOUT,
            )
            resp.raise_for_status()
            data     = resp.json()
            answer   = data.get("answer", "")
            contexts = data.get("contexts", [])
            intent   = data.get("intent", "")
            # direct 意图不调用检索工具，contexts 为空属预期，无需重试
            if answer and (contexts or intent == "direct"):
                return {"answer": answer, "contexts": contexts, "intent": intent}
            reason = "answer 为空" if not answer else "contexts 为空"
            print(f"  ⚠️  {reason}（attempt {attempt + 1}/{max_retries + 1}）")
        except requests.exceptions.Timeout:
            print(f"  ⚠️  超时 {CHAT_TIMEOUT}s（attempt {attempt + 1}/{max_retries + 1}）")
        except Exception as e:
            print(f"  ⚠️  请求异常: {e}（attempt {attempt + 1}/{max_retries + 1}）")
        if attempt < max_retries:
            time.sleep(3)
    return {"answer": "", "contexts": [], "intent": ""}


def is_done(r: dict) -> bool:
    return bool(r.get("answer")) and (bool(r.get("contexts")) or r.get("intent") == "direct")


def run(batch: int):
    print(f"\n{'='*60}")
    print(f"  飞轮 Step 2 — 批量 RAG 推理  (batch={batch})")
    print(f"{'='*60}")

    qa_file = qa_path(batch)
    if not qa_file.exists():
        print(f"❌ 找不到 {qa_file}")
        sys.exit(1)

    with open(qa_file, encoding="utf-8") as f:
        qa_pairs = json.load(f)

    verified = [q for q in qa_pairs if q.get("verified", False)]
    print(f"QA 对总数: {len(qa_pairs)}，通过验证: {len(verified)}")
    if not verified:
        print("❌ 没有通过验证的 QA 对，退出")
        sys.exit(1)

    # 服务健康检查
    try:
        requests.get(f"{CHAT_API_URL}/health", timeout=10).raise_for_status()
        print(f"✅ 服务健康检查通过：{CHAT_API_URL}")
    except Exception as e:
        print(f"❌ 服务不可用: {e}")
        sys.exit(1)

    # 断点续跑：只保留成功记录
    ri_path = ragas_input_path(batch)
    ri_path.parent.mkdir(parents=True, exist_ok=True)

    ragas_input: list[dict] = []
    done_ids: set[str] = set()
    if ri_path.exists():
        with open(ri_path, encoding="utf-8") as f:
            existing = json.load(f)
        ragas_input = [r for r in existing if is_done(r)]
        done_ids = {r["chunk_id"] for r in ragas_input}
        dropped = len(existing) - len(ragas_input)
        if done_ids:
            print(f"断点续跑：已完成 {len(done_ids)} 条"
                  + (f"，丢弃 {dropped} 条失败记录重新处理" if dropped else ""))

    pending = [qa for qa in verified if qa.get("chunk_id") not in done_ids]
    print(f"\n向 {CHAT_API_URL} 发送 {len(pending)} 个问题（已跳过 {len(done_ids)} 条）...\n")

    for i, qa in enumerate(pending):
        question   = qa["question"]
        global_idx = len(done_ids) + i
        print(f"  [{global_idx + 1:03d}/{len(verified)}] {qa.get('topic', '')[:30]}")
        print(f"    Q: {question[:80]}{'...' if len(question) > 80 else ''}")

        t0  = time.time()
        res = call_chat_eval(question, global_idx)
        dt  = time.time() - t0

        a_preview = (res['answer'][:80] + '...') if len(res.get('answer','')) > 80 else (res.get('answer') or '（空，已重试）')
        print(f"    A: {a_preview}")
        print(f"    intent: {res['intent'] or '?'}  contexts: {len(res['contexts'])} 块  耗时: {dt:.1f}s")

        record = {
            "id":            global_idx,
            "chunk_id":      qa.get("chunk_id", ""),
            "topic":         qa.get("topic", ""),
            "question_type": qa.get("question_type", ""),
            "question":      question,
            "ground_truth":  qa["ground_truth"],
            "answer":        res["answer"],
            "contexts":      res["contexts"],
            "intent":        res["intent"],
        }
        # 透传分析用元数据（合成数据的配对/分组字段）
        for key in ("eval_set", "source_id", "adv_style", "question_original"):
            if key in qa:
                record[key] = qa[key]
        ragas_input.append(record)

        with open(ri_path, "w", encoding="utf-8") as f:
            json.dump(ragas_input, f, ensure_ascii=False, indent=2)

        if i < len(pending) - 1:
            time.sleep(CHAT_DELAY)

    empty = sum(1 for r in ragas_input if not is_done(r))
    intents: dict[str, int] = {}
    for r in ragas_input:
        intents[r.get("intent") or "?"] = intents.get(r.get("intent") or "?", 0) + 1
    print(f"\nStep 2 完成 ✅  共 {len(ragas_input)} 条，失败: {empty} 条，intent 分布: {intents}")
    print(f"结果文件：{ri_path}")
    print(f"[下一步] 将 {ri_path.name} 拉回本地，交给 Claude 做评估与诊断")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    run(parser.parse_args().batch)


if __name__ == "__main__":
    main()
