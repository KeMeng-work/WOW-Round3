"""
Milvus 知识库导出工具（在服务器上运行一次）

将全量 chunks 导出为本地 JSON，供本地机器通过 --chunks-file 参数离线使用。

用法（服务器上）：
  python utils/export_chunks.py
  python utils/export_chunks.py --out data/chunks_export.json
  python utils/export_chunks.py --uri http://localhost:19530 --collection wow_knowledge_base_v2

然后将 chunks_export.json 拷贝到本地：
  scp user@server:/path/to/rag-data-flywheel/data/chunks_export.json ./data/
"""

import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).parent.parent

try:
    from pymilvus import MilvusClient
except ImportError:
    print("请先安装 pymilvus：pip install pymilvus")
    sys.exit(1)

FIELDS = [
    "chunk_id", "content", "topic", "split_type",
    "dialog_id", "chunk_seq",
    "prev_chunk_id", "next_chunk_id",
    "wizard_eval_score",
    "ner_entities_json",
    "relations_json",
    "keywords_json",
]


def main():
    parser = argparse.ArgumentParser(description="Milvus 知识库导出工具")
    parser.add_argument("--uri",        default="http://localhost:19530")
    parser.add_argument("--collection", default="wow_knowledge_base_v2")
    parser.add_argument("--out",        default=str(ROOT / "data" / "chunks_export.json"))
    parser.add_argument("--all",        action="store_true",
                        help="导出全部 chunk（默认只导出 has_checked_evidence=true）")
    args = parser.parse_args()

    print(f"连接 Milvus: {args.uri} / {args.collection}")
    client = MilvusClient(uri=args.uri)

    filt = "" if args.all else "has_checked_evidence == true"
    label = "全部" if args.all else "has_checked_evidence=true"

    results = client.query(
        collection_name=args.collection,
        filter=filt,
        output_fields=FIELDS,
        limit=10000,
    )
    client.close()

    results.sort(key=lambda x: x.get("chunk_id", ""))
    print(f"导出 {len(results)} 条 chunk（{label}）")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"已保存：{out_path}")
    print()
    print("下一步（在本地机器上）：")
    print(f"  scp user@server:{out_path.resolve()} ./data/chunks_export.json")
    print(f"  python flywheel_run.py --step 1 --batch 1 --chunks-file data/chunks_export.json")


if __name__ == "__main__":
    main()
