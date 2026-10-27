"""
Milvus 知识库独立导出脚本
可单独拷贝到服务器任意目录运行，无需整个飞轮项目。

只拉文本/元数据字段，不拉 vector（768维浮点数组，对QA生成无用）。

依赖：pip install pymilvus
用法：python export_chunks_standalone.py
      python export_chunks_standalone.py --out /tmp/chunks_export.json
"""

import argparse
import json
import sys

# ── 配置（与生产项目 config_path.py 保持一致）─────────────────────
MILVUS_URI        = "http://localhost:19530"
MILVUS_COLLECTION = "wow_knowledge_base_v2"
OUTPUT_FILE       = "chunks_export.json"

# 只拉 QA 生成所需字段，跳过：
#   vector (768维float，无用)、sparse (BM25稀疏向量，无用)
#   content_length/content_hash/evidence_density 等统计字段（无用）
EXPORT_FIELDS = [
    "chunk_id",
    "content",
    "topic",
    "split_type",
    "dialog_id",
    "dialog_index",
    "chunk_seq",
    "prev_chunk_id",
    "next_chunk_id",
    "wizard_eval_score",
    "has_checked_evidence",
    "speaker_role",
    "ner_entities_json",
    "relations_json",
    "keywords_json",
]
# ──────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Milvus知识库独立导出脚本")
    parser.add_argument("--uri",        default=MILVUS_URI,
                        help=f"Milvus URI（默认 {MILVUS_URI}）")
    parser.add_argument("--collection", default=MILVUS_COLLECTION,
                        help=f"集合名（默认 {MILVUS_COLLECTION}）")
    parser.add_argument("--out",        default=OUTPUT_FILE,
                        help=f"输出文件路径（默认 {OUTPUT_FILE}）")
    parser.add_argument("--all",        action="store_true",
                        help="导出全部 chunk（默认只导出 has_checked_evidence=true 的块）")
    args = parser.parse_args()

    try:
        from pymilvus import MilvusClient
    except ImportError:
        print("缺少依赖：pip install pymilvus")
        sys.exit(1)

    print(f"连接 Milvus: {args.uri}")
    print(f"集合: {args.collection}")
    client = MilvusClient(uri=args.uri)

    if not client.has_collection(args.collection):
        print(f"集合 {args.collection} 不存在，请检查名称")
        client.close()
        sys.exit(1)

    filt  = "" if args.all else "has_checked_evidence == true"
    label = "全部" if args.all else "has_checked_evidence=true"

    print(f"拉取字段: {', '.join(EXPORT_FIELDS)}")
    print(f"过滤条件: {label}")

    results = client.query(
        collection_name=args.collection,
        filter=filt,
        output_fields=EXPORT_FIELDS,
        limit=10000,
    )
    client.close()

    if not results:
        print("未查询到数据，请检查过滤条件或集合名称")
        sys.exit(1)

    results.sort(key=lambda x: x.get("chunk_id", ""))
    print(f"共 {len(results)} 条 chunk")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    size_kb = len(json.dumps(results, ensure_ascii=False).encode()) / 1024
    print(f"已保存: {args.out}  ({size_kb:.1f} KB)")
    print()
    print("拷贝到本地后运行 Step 1：")
    print(f"  python flywheel_run.py --step 1 --batch 1 --chunks-file {args.out}")


if __name__ == "__main__":
    main()
