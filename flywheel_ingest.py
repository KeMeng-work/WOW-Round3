"""
知识库增量写入脚本（Human-in-the-Loop 人工触发）

从 flywheel_run.py Step 3 输出的 candidates_batch_N.json 中读取候选文档，
展示人工确认界面，对批准的文档直接写入 Milvus（嵌入 + 插入）。

用法：
  python flywheel_ingest.py --batch 1              # 交互确认后写入
  python flywheel_ingest.py --batch 1 --approve-all # 批量批准全部（谨慎）
  python flywheel_ingest.py --batch 1 --dry-run    # 仅预览，不写入
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from pymilvus import MilvusClient

sys.stdout.reconfigure(encoding="utf-8")

from config import (
    DATA_DIR,
    MILVUS_URI, MILVUS_COLLECTION,
    EMBED_CONFIG,
)
from utils.onnx_embedder import ONNXEmbedder

NOMIC_DOC_PREFIX = "search_document: "


def report_path(batch: int) -> Path:
    return DATA_DIR / "reports" / f"candidates_batch_{batch}.json"


def load_candidates(batch: int) -> list[dict]:
    path = report_path(batch)
    if not path.exists():
        print(f"❌ 找不到候选文档报告：{path}")
        print("   请先运行 flywheel_run.py --step 3 --batch {batch}")
        sys.exit(1)

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    items = data.get("items", [])
    pending = [i for i in items if i.get("status") == "pending_human_review"]
    print(f"候选文档：{len(items)} 条，待确认：{len(pending)} 条")
    return pending


def human_review(candidates: list[dict]) -> list[dict]:
    """逐条展示，人工输入批准/拒绝/编辑。"""
    approved = []

    print()
    print("─" * 60)
    print("  人工审核界面")
    print("  操作：y=批准  n=拒绝  e=编辑内容  q=退出审核")
    print("─" * 60)

    for i, c in enumerate(candidates):
        print(f"\n[{i+1}/{len(candidates)}] Topic: {c['topic']}")
        print(f"  问题   : {c['question']}")
        print(f"  缺口描述: {c['gap_description'][:200]}")
        print(f"  建议内容: {c['ground_truth'][:300]}")

        while True:
            choice = input("  操作 [y/n/e/q]: ").strip().lower()
            if choice == "y":
                approved.append(c)
                print("  ✅ 已批准")
                break
            elif choice == "n":
                print("  ❌ 已拒绝")
                break
            elif choice == "e":
                print("  当前内容（编辑后输入，直接回车保留原文）：")
                print(f"  {c['ground_truth']}")
                new_content = input("  新内容: ").strip()
                if new_content:
                    c["ground_truth"] = new_content
                    print("  内容已更新")
                approved.append(c)
                print("  ✅ 已批准（含编辑）")
                break
            elif choice == "q":
                print(f"\n  提前退出，已批准 {len(approved)} 条")
                return approved
            else:
                print("  请输入 y / n / e / q")

    print(f"\n审核完成：批准 {len(approved)} / {len(candidates)} 条")
    return approved


def build_chunk_id(topic: str, content: str) -> str:
    """为新文档生成唯一 chunk_id。"""
    h = hashlib.md5(f"{topic}:{content}".encode()).hexdigest()[:12]
    return f"flywheel_{h}"


def ingest_to_milvus(approved: list[dict], embedder: ONNXEmbedder, dry_run: bool):
    """将批准的文档嵌入并写入 Milvus。"""
    if not approved:
        print("没有批准的文档，跳过写入")
        return

    print(f"\n准备写入 {len(approved)} 条文档到 Milvus ({MILVUS_COLLECTION})...")

    if dry_run:
        print("[DRY RUN] 以下文档将被写入（实际未执行）：")
        for c in approved:
            print(f"  - {c['topic']}: {c['ground_truth'][:80]}...")
        return

    client = MilvusClient(uri=MILVUS_URI)

    rows = []
    for c in approved:
        content  = c["ground_truth"]
        topic    = c.get("topic", "flywheel_generated")
        chunk_id = build_chunk_id(topic, content)

        # nomic 文档前缀（与生产入库路径一致）
        embedding = embedder.embed_query(NOMIC_DOC_PREFIX + content)

        rows.append({
            "chunk_id":                 chunk_id,
            "vector":                   embedding,
            "content":                  content,
            "source":                   "flywheel",
            "document_id":              f"flywheel_{int(time.time())}",
            "chunk_seq":                0,
            "content_length":           len(content),
            "content_hash":             hashlib.sha256(content.encode()).hexdigest()[:64],
            "dialog_id":                "",
            "dialog_index":             0,
            "topic":                    topic,
            "turn_count":               0,
            "speaker_role":             "flywheel",
            "wizard_eval_score":        0,
            "has_checked_evidence":     True,
            "total_retrieved_passages": 0,
            "total_retrieved_topics":   0,
            "evidence_density":         0.0,
            "prev_chunk_id":            "",
            "next_chunk_id":            "",
            "split_type":               "flywheel",
            "keywords_text":            [],
            "ner_entities_text":        [],
            "keywords_json":            {},
            "ner_entities_json":        {},
            "relations_json":           {},
        })

    try:
        result = client.insert(collection_name=MILVUS_COLLECTION, data=rows)
        print(f"✅ 写入成功：{result.get('insert_count', len(rows))} 条")
    except Exception as e:
        print(f"❌ Milvus 写入失败：{e}")
        raise
    finally:
        client.close()


def update_report_status(batch: int, approved_ids: set[str]):
    """将报告中已批准的文档状态更新为 approved。"""
    path = report_path(batch)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    for item in data.get("items", []):
        cid = build_chunk_id(item.get("topic", ""), item.get("ground_truth", ""))
        if cid in approved_ids:
            item["status"] = "ingested"

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="知识库增量写入脚本")
    parser.add_argument("--batch",       type=int, required=True, help="批次号")
    parser.add_argument("--approve-all", action="store_true",
                        help="跳过人工确认，批量批准全部候选文档（谨慎使用）")
    parser.add_argument("--dry-run",     action="store_true",
                        help="预览模式，不实际写入 Milvus")
    args = parser.parse_args()

    candidates = load_candidates(args.batch)
    if not candidates:
        print("没有待确认的候选文档，退出")
        sys.exit(0)

    # 人工审核
    if args.approve_all:
        print(f"--approve-all 模式：直接批准全部 {len(candidates)} 条")
        approved = candidates
    else:
        approved = human_review(candidates)

    if not approved:
        print("没有批准任何文档，退出")
        sys.exit(0)

    # 初始化嵌入器（仅在有文档需要写入时才加载）
    if not args.dry_run:
        print(f"\n初始化 ONNX 嵌入器（{EMBED_CONFIG['model_path']}）...")
        embedder = ONNXEmbedder(
            model_path=EMBED_CONFIG["model_path"],
            onnx_file=EMBED_CONFIG["onnx_file"],
            providers=EMBED_CONFIG["providers"],
            intra_op_num_threads=EMBED_CONFIG["intra_op_num_threads"],
            inter_op_num_threads=EMBED_CONFIG["inter_op_num_threads"],
            max_length=EMBED_CONFIG["max_length"],
            batch_size=EMBED_CONFIG["batch_size"],
        )
    else:
        embedder = None

    ingest_to_milvus(approved, embedder, dry_run=args.dry_run)

    if not args.dry_run:
        approved_ids = {
            build_chunk_id(c.get("topic", ""), c.get("ground_truth", ""))
            for c in approved
        }
        update_report_status(args.batch, approved_ids)
        print(f"报告状态已更新：{report_path(args.batch)}")

    print("\n飞轮写入完成 ✅")


if __name__ == "__main__":
    main()
