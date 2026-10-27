"""
数据飞轮项目配置
从 .env 加载，默认值对应本机开发环境（服务在 localhost 运行）。
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── 项目路径 ──────────────────────────────────────────────────────
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"

# ── 生产系统 HTTP 接口 ────────────────────────────────────────────
CHAT_API_URL = os.getenv("CHAT_API_URL", "http://localhost:8000")

# ── Milvus ───────────────────────────────────────────────────────
MILVUS_URI        = os.getenv("MILVUS_URI", "http://localhost:19530")
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "wow_knowledge_base_v2")

# ── Redis（flywheel_ingest.py 用） ────────────────────────────────
REDIS_CONFIG = {
    "host":             os.getenv("REDIS_HOST", "localhost"),
    "port":             int(os.getenv("REDIS_PORT", 6380)),
    "password":         os.getenv("REDIS_PASSWORD"),
    "decode_responses": True,
}
REDIS_STREAM = "knowledge_pipeline"

# ── RAGAS Judge LLM（阿里百炼 qwen-max，与基准评估保持一致）───────
RAGAS_CONFIG = {
    "judge_model": os.getenv("RAGAS_JUDGE_MODEL", "qwen-max"),
    "api_key":     os.getenv("OPENAI_API_KEY"),
    "base_url":    "https://dashscope.aliyuncs.com/compatible-mode/v1",
}

# ── QA 生成 LLM（Qwen3.7-Flash，阿里百炼）────────────────────────────
QA_GEN_CONFIG = {
    "model":       os.getenv("QA_GEN_MODEL", "Qwen3.7-Flash"),
    "api_key":     os.getenv("QA_GEN_API_KEY"),
    "base_url":    "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "temperature": float(os.getenv("QA_GEN_TEMPERATURE", 0.7)),
    "max_tokens":  int(os.getenv("QA_GEN_MAX_TOKENS", 1024)),
}

# ── 本地知识库快照（Agent 1 使用，替代 Milvus 直连）──────────────────
_default_kb = str(ROOT.parent / "pythonProject" / "data" / "evaluation" / "kb_snapshot.json")
KB_SNAPSHOT_PATH = os.getenv("KB_SNAPSHOT_PATH", _default_kb)

# ── ONNX 嵌入模型（nomic-embed-text，用于 RAGAS AnswerRelevancy）──
# 默认指向同级 pythonProject 的模型目录，也可通过 .env 覆盖
_default_embed = str(ROOT.parent / "pythonProject" / "models" / "nomic-embed-text")
EMBED_CONFIG = {
    "model_path":           os.getenv("EMBED_MODEL_PATH", _default_embed),
    "onnx_file":            "onnx/model_int8.onnx",
    "providers":            ["CPUExecutionProvider"],
    "intra_op_num_threads": int(os.getenv("EMBED_THREADS", 4)),
    "inter_op_num_threads": 1,
    "max_length":           512,
    "batch_size":           32,
}

# ── 飞轮批次参数 ──────────────────────────────────────────────────
BATCH_SIZE = int(os.getenv("FLYWHEEL_BATCH_SIZE", 20))   # 每次送给 Claude 的 chunk 数（20 条：质量/长度最优平衡）

# 合成问题类型分布（适合 WoW 单域知识库）
QA_TYPE_DIST = {
    "simple":    0.50,   # 直接事实性问题——基础召回覆盖
    "reasoning": 0.30,   # 需要推理的问题
    "multi_hop": 0.20,   # 跨信息点的综合问题（WoW 跨块较难，保守比例）
}

# ── 四层门控阈值 ──────────────────────────────────────────────────
# 这些是诊断分类阈值，不是上线门槛：
#   低于阈值 → 归入对应失败报告 → 触发相应优化动作
#   F=0.90 对齐"接地约束改完后验收目标"
GATE_THRESHOLDS = {
    "faithfulness":       0.90,   # 对齐接地约束验收目标（基准 0.864，改完应 ≥0.90）
    "answer_relevancy":   0.80,   # 答案切题，适中
    "context_precision":  0.75,   # 检索精准度，适中
    "context_recall":     0.70,   # 知识覆盖度，适当放宽（过严会误报大量缺口）
}

# ── 请求参数 ──────────────────────────────────────────────────────
CHAT_TIMEOUT   = int(os.getenv("CHAT_TIMEOUT", 240))   # /chat/stream 单条超时（秒），对齐 run_eval_baseline v2
CHAT_DELAY     = float(os.getenv("CHAT_DELAY", 1.5))   # 批量推理请求间隔（秒），防止速率限制
RAGAS_WORKERS  = int(os.getenv("RAGAS_WORKERS", 4))    # RAGAS 并发 worker
RAGAS_TIMEOUT  = int(os.getenv("RAGAS_TIMEOUT", 300))  # RAGAS 单次 LLM 超时（秒）
