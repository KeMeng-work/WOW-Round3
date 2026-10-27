"""
数据飞轮主运行脚本

链路：
  Step 1 (Agent 1)  — 从本地 kb_snapshot.json 分层采样，用 Qwen3.7-Flash 自动生成中文 QA 对
                       输出 qa_raw_batch_N.json → 由 Claude 验证后保存为 qa_batch_N.json
  Step 2 (Agent 2)  — 加载 Claude 验证后的 QA 对，批量调用 /chat_eval，运行 RAGAS 评估
  Step 3 (Agent 3)  — 四层门控诊断，输出四类报告

用法：
  python flywheel_run.py --step 1               # 仅运行 Step 1（Qwen 生成原始 QA）
  python flywheel_run.py --step 2 --batch 1     # 仅运行 Step 2（需先完成 Claude 验证）
  python flywheel_run.py --step 3 --batch 1     # 仅运行 Step 3
  python flywheel_run.py --total 200            # Step 1 目标 QA 总数（默认 200）
"""

import sys
from types import ModuleType

# RAGAS 0.4.x 硬编码 import langchain_community.chat_models.vertexai，该模块已在新版删除
# 提前注入空壳让 ragas 顺利加载
class _AnyAttr(ModuleType):
    def __getattr__(self, name):
        return None

for _mod in ("langchain_community.chat_models.vertexai", "langchain_community.chat_models"):
    if _mod not in sys.modules:
        sys.modules[_mod] = _AnyAttr(_mod)

import argparse
import json
import math
import os
import random
import time
import uuid
from pathlib import Path

import requests
from dotenv import load_dotenv
from openai import OpenAI

sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

# ── 路径 ──────────────────────────────────────────────────────────
_ROOT    = Path(__file__).parent
DATA_DIR = _ROOT / "data"

# ── 生产系统接口 ──────────────────────────────────────────────────
CHAT_API_URL = os.getenv("CHAT_API_URL", "http://localhost:8000")

# ── RAGAS Judge LLM ───────────────────────────────────────────────
RAGAS_CONFIG = {
    "judge_model": os.getenv("RAGAS_JUDGE_MODEL", "qwen-max"),
    "api_key":     os.getenv("OPENAI_API_KEY"),
    "base_url":    "https://dashscope.aliyuncs.com/compatible-mode/v1",
}

# ── QA 生成 LLM ───────────────────────────────────────────────────
QA_GEN_CONFIG = {
    "model":       os.getenv("QA_GEN_MODEL", "qwen-turbo"),
    "api_key":     os.getenv("QA_GEN_API_KEY"),
    "base_url":    "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "temperature": float(os.getenv("QA_GEN_TEMPERATURE", "0.7")),
    "max_tokens":  int(os.getenv("QA_GEN_MAX_TOKENS", "1024")),
}

# ── 知识库快照路径 ────────────────────────────────────────────────
KB_SNAPSHOT_PATH = os.getenv(
    "KB_SNAPSHOT_PATH",
    str(_ROOT.parent / "pythonProject" / "data" / "evaluation" / "kb_snapshot.json"),
)

# ── ONNX 嵌入模型（RAGAS AnswerRelevancy 使用）───────────────────
EMBED_CONFIG = {
    "model_path":           os.getenv(
        "EMBED_MODEL_PATH",
        str(_ROOT.parent / "pythonProject" / "models" / "nomic-embed-text"),
    ),
    "onnx_file":            "onnx/model_int8.onnx",
    "providers":            ["CPUExecutionProvider"],
    "intra_op_num_threads": int(os.getenv("EMBED_THREADS", "4")),
    "inter_op_num_threads": 1,
    "max_length":           512,
    "batch_size":           32,
}

# ── 问题类型分布 ──────────────────────────────────────────────────
QA_TYPE_DIST = {
    "simple":    0.50,
    "reasoning": 0.30,
    "multi_hop": 0.20,
}

# ── 四层门控阈值 ──────────────────────────────────────────────────
GATE_THRESHOLDS = {
    "faithfulness":       0.90,
    "answer_relevancy":   0.80,
    "context_precision":  0.75,
    "context_recall":     0.70,
}

# ── 请求参数 ──────────────────────────────────────────────────────
CHAT_TIMEOUT  = int(os.getenv("CHAT_TIMEOUT",  "240"))
CHAT_DELAY    = float(os.getenv("CHAT_DELAY",  "1.5"))
RAGAS_WORKERS = int(os.getenv("RAGAS_WORKERS", "4"))
RAGAS_TIMEOUT = int(os.getenv("RAGAS_TIMEOUT", "300"))


# ─────────────────────────────────────────────────────────────────
# 路径工具
# ─────────────────────────────────────────────────────────────────

def chunks_path(batch: int) -> Path:
    return DATA_DIR / "qa" / f"chunks_batch_{batch}.json"

def qa_raw_path(batch: int) -> Path:
    return DATA_DIR / "qa" / f"qa_raw_batch_{batch}.json"

def qa_path(batch: int) -> Path:
    return DATA_DIR / "qa" / f"qa_batch_{batch}.json"

def ragas_input_path(batch: int) -> Path:
    return DATA_DIR / "ragas_input" / f"ragas_input_batch_{batch}.json"

def ragas_results_path(batch: int) -> Path:
    return DATA_DIR / "ragas_results" / f"ragas_results_batch_{batch}.json"

def report_path(name: str, batch: int) -> Path:
    return DATA_DIR / "reports" / f"{name}_batch_{batch}.json"


# ─────────────────────────────────────────────────────────────────
# Agent 1：从 kb_snapshot.json 分层采样，用 Qwen3.7-Flash 生成中文 QA
# ─────────────────────────────────────────────────────────────────

QWEN_GENERATION_PROMPT = """你是一名专业的知识库问答对生成专家。

任务：基于提供的英文知识块，生成一条高质量的**中文** QA 对。

输入字段：
- content / chunks  英文原文
- topic             所属话题
- question_type     问题类型（见下方）

问题类型说明：
- simple    事实性问题，答案可直接从原文找到（什么 / 谁 / 何时 / 哪里）
- reasoning 推断性问题，需要理解原文语义（为什么 / 怎么 / 会怎样）
- multi_hop 综合性问题，需要结合 chunks 中的两段内容才能完整回答

要求：
1. question 和 ground_truth 均为**中文**
2. ground_truth 须完全基于原文，不得加入外部知识，长度 80-200 字
3. 问题具体清晰，无歧义，不依赖原文以外的上下文
4. 完成后自我验证（三项全通过才 verified=true）：
   - 可回答性：问题的完整答案能从提供的文本中找到
   - GT 接地性：ground_truth 每处细节都有原文支撑
   - 表述清晰：问题无歧义、不引入外部前置知识

严格输出 JSON 对象，不含 markdown 代码块或任何额外说明：
{
  "question": "<中文问题>",
  "ground_truth": "<中文标准答案，80-200字>",
  "verified": true或false,
  "rejection_reason": null或"<失败原因>"
}"""


def load_kb_snapshot(path: str, min_score: int = 4) -> list[dict]:
    """加载并过滤 kb_snapshot.json，保留 wizard_eval_score >= min_score 的记录。"""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    filtered = [r for r in data if r.get("wizard_eval_score", 0) >= min_score]
    print(f"知识库快照: 共 {len(data)} 条 → score≥{min_score} 过滤后 {len(filtered)} 条")
    return filtered


def stratified_sample(records: list[dict], total: int = 200, seed: int = 42) -> list[dict]:
    """分层采样：70% 来自 score=5，30% 来自 score=4。"""
    random.seed(seed)
    score5 = [r for r in records if r.get("wizard_eval_score") == 5]
    score4 = [r for r in records if r.get("wizard_eval_score") == 4]
    n5 = min(round(total * 0.70), len(score5))
    n4 = min(total - n5, len(score4))
    sampled = random.sample(score5, n5) + random.sample(score4, n4)
    random.shuffle(sampled)
    print(f"分层采样: score=5 取 {n5} 条 + score=4 取 {n4} 条 = {len(sampled)} 条")
    return sampled


def build_topic_index(chunks: list[dict]) -> dict[str, list[dict]]:
    """按 topic 分组，供 multi_hop 配对使用。"""
    index: dict[str, list[dict]] = {}
    for c in chunks:
        t = c.get("topic") or "unknown"
        index.setdefault(t, []).append(c)
    return index


def assign_question_types(chunks: list[dict]) -> list[dict]:
    """按 QA_TYPE_DIST 比例给 chunk 分配问题类型。"""
    n = len(chunks)
    types = []
    for qtype, ratio in QA_TYPE_DIST.items():
        types.extend([qtype] * round(n * ratio))
    while len(types) < n:
        types.append("simple")
    types = types[:n]
    random.shuffle(types)
    for chunk, qtype in zip(chunks, types):
        chunk["question_type"] = qtype
    return chunks


def _build_gen_item(chunk: dict, topic_index: dict[str, list[dict]]) -> dict:
    """为单个 chunk 构建发给 Qwen 的输入（multi_hop 自动配对同 topic 块）。"""
    qtype = chunk["question_type"]
    topic = chunk.get("topic", "unknown")

    if qtype == "multi_hop":
        siblings = [x for x in topic_index.get(topic, [])
                    if x["chunk_id"] != chunk["chunk_id"]]
        peer = random.choice(siblings) if siblings else None

        if peer is not None:
            return {
                "question_type": "multi_hop",
                "topic":         topic,
                "chunks": [
                    {"content": chunk["content"]},
                    {"content": peer["content"]},
                ],
            }
        else:
            # 该 topic 只有一个块，降级为 reasoning
            chunk["question_type"] = "reasoning"
            return {
                "question_type": "reasoning",
                "topic":         topic,
                "content":       chunk["content"],
            }

    return {
        "question_type": qtype,
        "topic":         topic,
        "content":       chunk["content"],
    }


def generate_qa_with_qwen(item: dict, client: OpenAI) -> dict | None:
    """调用 Qwen3.7-Flash 为单个条目生成中文 QA 对，返回解析后的字典。"""
    user_content = json.dumps(item, ensure_ascii=False)
    try:
        resp = client.chat.completions.create(
            model=QA_GEN_CONFIG["model"],
            messages=[
                {"role": "system", "content": QWEN_GENERATION_PROMPT},
                {"role": "user",   "content": user_content},
            ],
            temperature=QA_GEN_CONFIG["temperature"],
            max_tokens=QA_GEN_CONFIG["max_tokens"],
        )
        text = resp.choices[0].message.content.strip()
        # 去掉可能包裹的 markdown 代码块
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        return json.loads(text.strip())
    except json.JSONDecodeError as e:
        print(f"    ⚠️  JSON 解析失败: {e} | 原始: {text[:120]}")
        return None
    except Exception as e:
        print(f"    ⚠️  Qwen 调用失败: {e}")
        return None


def run_step1(batch: int, total: int):
    print(f"\n{'='*60}")
    print(f"  Step 1 / Agent 1 — Qwen3.7-Flash 生成中文 QA  (batch={batch})")
    print(f"{'='*60}")

    # 1. 加载知识库快照
    all_chunks = load_kb_snapshot(KB_SNAPSHOT_PATH, min_score=4)

    # 2. 分层采样
    sampled = stratified_sample(all_chunks, total)

    # 3. 分配问题类型
    sampled = assign_question_types(sampled)

    # 4. topic 索引（multi_hop 配对用）
    topic_index = build_topic_index(all_chunks)

    # 5. 初始化 Qwen 客户端
    if not QA_GEN_CONFIG["api_key"]:
        print("❌ QA_GEN_API_KEY 未配置，请在 .env 中添加")
        sys.exit(1)
    client = OpenAI(
        api_key=QA_GEN_CONFIG["api_key"],
        base_url=QA_GEN_CONFIG["base_url"],
    )

    # 6. 逐条生成并保存（支持断点续跑）
    raw_out = qa_raw_path(batch)
    raw_out.parent.mkdir(parents=True, exist_ok=True)

    qa_results = []
    for i, chunk in enumerate(sampled):
        topic = chunk.get("topic", "unknown")
        qtype = chunk["question_type"]
        print(f"  [{i+1:03d}/{len(sampled)}] {topic[:40]:<40} | {qtype}")

        item = _build_gen_item(chunk, topic_index)
        qa   = generate_qa_with_qwen(item, client)

        if qa:
            qa_results.append({
                "chunk_id":          chunk["chunk_id"],
                "topic":             topic,
                "question_type":     chunk["question_type"],  # 可能被降级为 reasoning
                "wizard_eval_score": chunk.get("wizard_eval_score"),
                "question":          qa.get("question", ""),
                "ground_truth":      qa.get("ground_truth", ""),
                "verified":          qa.get("verified", False),
                "rejection_reason":  qa.get("rejection_reason"),
            })
            mark = "✅" if qa.get("verified") else "⚠️ "
            print(f"    {mark} {qa.get('question', '')[:65]}")
        else:
            print(f"    ❌ 生成失败，跳过")

        with open(raw_out, "w", encoding="utf-8") as f:
            json.dump(qa_results, f, ensure_ascii=False, indent=2)

        if i < len(sampled) - 1:
            time.sleep(0.5)

    verified_count = sum(1 for q in qa_results if q.get("verified"))
    print(f"\n{'='*60}")
    print(f"  Step 1 完成：生成 {len(qa_results)} 条，自验证通过 {verified_count} 条")
    print(f"  原始文件：{raw_out}")
    print(f"{'='*60}")
    print(f"\n[下一步] 在 Claude 对话窗口触发验证：")
    print(f"  '请验证 {raw_out.name} 中的 QA 对质量，输出 {qa_path(batch).name}'")


# ─────────────────────────────────────────────────────────────────
# Agent 2：批量推理 + RAGAS 评估
# ─────────────────────────────────────────────────────────────────

def call_chat_eval(question: str, idx: int, max_retries: int = 2) -> dict:
    """调用飞轮专用 /chat_eval，返回 answer + contexts。空结果自动重试。"""
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
                timeout=CHAT_TIMEOUT,
            )
            resp.raise_for_status()
            data     = resp.json()
            answer   = data.get("answer", "")
            contexts = data.get("contexts", [])
            if answer and contexts:
                return {"answer": answer, "contexts": contexts}
            reason = "answer 为空" if not answer else "contexts 为空"
            print(f"  ⚠️  {reason}（attempt {attempt + 1}/{max_retries + 1}）")
        except requests.exceptions.Timeout:
            print(f"  ⚠️  超时 {CHAT_TIMEOUT}s（attempt {attempt + 1}/{max_retries + 1}）")
        except Exception as e:
            print(f"  ⚠️  请求异常: {e}（attempt {attempt + 1}/{max_retries + 1}）")
        if attempt < max_retries:
            time.sleep(3)
    return {"answer": "", "contexts": []}


def build_ragas_components():
    """构建 RAGAS Judge LLM 和 ONNX 嵌入器（懒加载，Step 2 使用时才 import）。"""
    import logging as _logging
    import numpy as _np
    import onnxruntime as _ort
    from langchain_core.embeddings import Embeddings
    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.metrics import (Faithfulness, AnswerRelevancy,
                                ContextPrecision, ContextRecall, AnswerCorrectness)
    from transformers import AutoTokenizer
    from typing import List as _List

    _logger = _logging.getLogger(__name__)

    class ONNXEmbedder(Embeddings):
        def __init__(
            self,
            model_path: str,
            onnx_file: str = "onnx/model_int8.onnx",
            providers: list = None,
            intra_op_num_threads: int = 4,
            inter_op_num_threads: int = 1,
            max_length: int = 512,
            batch_size: int = 32,
        ):
            if providers is None:
                providers = ["CPUExecutionProvider"]
            self.max_length = max_length
            self.batch_size = batch_size
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True, local_files_only=True
            )
            onnx_path = os.path.join(model_path, onnx_file)
            sess_options = _ort.SessionOptions()
            sess_options.graph_optimization_level = _ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.intra_op_num_threads = intra_op_num_threads
            sess_options.inter_op_num_threads = inter_op_num_threads
            self.session = _ort.InferenceSession(onnx_path, sess_options, providers=providers)
            self._input_names = {inp.name for inp in self.session.get_inputs()}
            try:
                self.embed_query("warmup")
            except Exception as e:
                _logger.warning(f"ONNXEmbedder 预热失败: {e}")

        def _mean_pool(self, hidden: _np.ndarray, mask: _np.ndarray) -> _np.ndarray:
            m = mask[:, :, _np.newaxis].astype(_np.float32)
            return (hidden * m).sum(axis=1) / m.sum(axis=1).clip(min=1e-9)

        def _l2_norm(self, emb: _np.ndarray) -> _np.ndarray:
            return emb / _np.linalg.norm(emb, axis=-1, keepdims=True).clip(min=1e-9)

        def _embed_batch(self, texts: _List[str]) -> _np.ndarray:
            enc = self.tokenizer(
                texts, padding=True, truncation=True,
                max_length=self.max_length, return_tensors="np",
            )
            inputs = {
                "input_ids":      enc["input_ids"].astype(_np.int64),
                "attention_mask": enc["attention_mask"].astype(_np.int64),
            }
            if "token_type_ids" in self._input_names and "token_type_ids" in enc:
                inputs["token_type_ids"] = enc["token_type_ids"].astype(_np.int64)
            hidden = self.session.run(None, inputs)[0]
            return self._l2_norm(self._mean_pool(hidden, enc["attention_mask"]))

        def embed_query(self, text: str) -> _List[float]:
            return self._embed_batch([text])[0].tolist()

        def embed_documents(self, texts: _List[str]) -> _List[_List[float]]:
            out: _List[_List[float]] = []
            for i in range(0, len(texts), self.batch_size):
                out.extend(self._embed_batch(texts[i: i + self.batch_size]).tolist())
            return out

    judge_llm = ChatOpenAI(
        model=RAGAS_CONFIG["judge_model"],
        api_key=RAGAS_CONFIG["api_key"],
        base_url=RAGAS_CONFIG["base_url"],
        temperature=0,
        max_tokens=2048,
        timeout=RAGAS_TIMEOUT,
        max_retries=3,
    )
    embedder = ONNXEmbedder(
        model_path=EMBED_CONFIG["model_path"],
        onnx_file=EMBED_CONFIG["onnx_file"],
        providers=EMBED_CONFIG["providers"],
        intra_op_num_threads=EMBED_CONFIG["intra_op_num_threads"],
        inter_op_num_threads=EMBED_CONFIG["inter_op_num_threads"],
        max_length=EMBED_CONFIG["max_length"],
        batch_size=EMBED_CONFIG["batch_size"],
    )
    ragas_llm = LangchainLLMWrapper(judge_llm)
    ragas_emb = LangchainEmbeddingsWrapper(embedder)
    metrics = [
        Faithfulness(llm=ragas_llm),
        AnswerRelevancy(llm=ragas_llm, embeddings=ragas_emb),
        ContextPrecision(llm=ragas_llm),
        ContextRecall(llm=ragas_llm),
        AnswerCorrectness(llm=ragas_llm, embeddings=ragas_emb),
    ]
    return metrics


def classify_gates(scores: dict) -> dict:
    """四层门控分类，返回各指标 pass/fail 状态。"""
    gates = {}
    for metric, threshold in GATE_THRESHOLDS.items():
        val = scores.get(metric)
        if val is None:
            gates[metric] = "unknown"
        else:
            gates[metric] = "pass" if val >= threshold else "fail"
    return gates


def run_step2(batch: int):
    print(f"\n{'='*60}")
    print(f"  Step 2 / Agent 2 — RAG 推理 + RAGAS 评估  (batch={batch})")
    print(f"{'='*60}")

    qa_file = qa_path(batch)
    if not qa_file.exists():
        print(f"❌ 找不到 {qa_file}，请先完成 Step 1 并让 Claude 验证生成 QA 对")
        sys.exit(1)

    with open(qa_file, encoding="utf-8") as f:
        qa_pairs = json.load(f)

    # 只保留 verified=true 的样本
    verified = [q for q in qa_pairs if q.get("verified", False)]
    skipped  = len(qa_pairs) - len(verified)
    print(f"QA 对总数: {len(qa_pairs)}，通过验证: {len(verified)}，跳过: {skipped}")

    if not verified:
        print("❌ 没有通过验证的 QA 对，退出")
        sys.exit(1)

    # ── 0. 预检：服务健康检查 ─────────────────────────────────────
    try:
        hc = requests.get(f"{CHAT_API_URL}/health", timeout=10)
        hc.raise_for_status()
        print(f"✅ 服务健康检查通过：{CHAT_API_URL}")
    except Exception as e:
        print(f"❌ 服务不可用（{CHAT_API_URL}/health）: {e}")
        sys.exit(1)

    # ── 2a. 批量调用生产系统（支持断点续跑）──────────────────────
    ri_path = ragas_input_path(batch)
    ri_path.parent.mkdir(parents=True, exist_ok=True)

    # 加载已有结果：只保留成功记录（answer + contexts 均非空），失败记录丢弃并重处理
    ragas_input: list[dict] = []
    done_ids: set[str] = set()
    if ri_path.exists():
        with open(ri_path, encoding="utf-8") as f:
            existing = json.load(f)
        ragas_input = [r for r in existing if r.get("answer") and r.get("contexts")]
        done_ids = {r["chunk_id"] for r in ragas_input}
        dropped = len(existing) - len(ragas_input)
        if done_ids:
            print(f"断点续跑：已完成 {len(done_ids)} 条，跳过"
                  + (f"，丢弃 {dropped} 条失败记录重新处理" if dropped else ""))

    pending = [qa for qa in verified if qa.get("chunk_id") not in done_ids]
    print(f"\n推理阶段：向 {CHAT_API_URL} 发送 {len(pending)} 个问题"
          f"（已跳过 {len(done_ids)} 条）...\n")

    for i, qa in enumerate(pending):
        question   = qa["question"]
        global_idx = len(done_ids) + i
        print(f"  [{global_idx + 1:03d}/{len(verified)}] {qa.get('topic', '')[:30]}")
        print(f"    Q: {question[:80]}{'...' if len(question) > 80 else ''}")

        t0  = time.time()
        res = call_chat_eval(question, global_idx)
        dt  = time.time() - t0

        a_preview = (res['answer'][:80] + ('...' if len(res['answer']) > 80 else '')) if res['answer'] else '（空，已重试）'
        print(f"    A: {a_preview}")
        print(f"    contexts: {len(res['contexts'])} 块  耗时: {dt:.1f}s")

        ragas_input.append({
            "id":            global_idx,
            "chunk_id":      qa.get("chunk_id", ""),
            "topic":         qa.get("topic", ""),
            "question_type": qa.get("question_type", ""),
            "question":      question,
            "ground_truth":  qa["ground_truth"],
            "answer":        res["answer"],
            "contexts":      res["contexts"],
        })

        with open(ri_path, "w", encoding="utf-8") as f:
            json.dump(ragas_input, f, ensure_ascii=False, indent=2)

        if i < len(pending) - 1:
            time.sleep(CHAT_DELAY)

    # ── 2b. RAGAS 评估 ────────────────────────────────────────────
    valid_samples = [
        r for r in ragas_input
        if r.get("answer") and r.get("contexts") and r.get("ground_truth")
    ]
    invalid = len(ragas_input) - len(valid_samples)
    print(f"\nRAGAS 评估：{len(valid_samples)} 条有效（跳过空答案/空 contexts: {invalid} 条）")

    from ragas import evaluate, EvaluationDataset, SingleTurnSample
    from ragas.run_config import RunConfig

    print(f"初始化 Judge LLM ({RAGAS_CONFIG['judge_model']}) + ONNX 嵌入器...")
    metrics = build_ragas_components()

    ragas_samples = [
        SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=r["contexts"],
            reference=r["ground_truth"],
        )
        for r in valid_samples
    ]
    dataset = EvaluationDataset(samples=ragas_samples)

    print(f"开始 RAGAS 评估（{len(ragas_samples)} 条）...\n")
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        run_config=RunConfig(
            timeout=RAGAS_TIMEOUT,
            max_retries=3,
            max_wait=60,
            max_workers=RAGAS_WORKERS,
        ),
        batch_size=4,
        raise_exceptions=False,
    )
    df = result.to_pandas()

    # ── 2c. 组装结果 + 门控分类 ───────────────────────────────────
    metric_cols = ["faithfulness", "answer_relevancy", "context_precision",
                   "context_recall", "answer_correctness"]
    details = []
    for i, (_, row) in enumerate(df.iterrows()):
        src = valid_samples[i]
        scores = {}
        for col in metric_cols:
            v = row.get(col)
            scores[col] = None if (v is None or (isinstance(v, float) and math.isnan(v))) else round(float(v), 4)

        gates = classify_gates(scores)
        failure_types = [m for m, g in gates.items() if g == "fail"]

        details.append({
            **{k: src[k] for k in ["id", "chunk_id", "topic", "question_type",
                                    "question", "ground_truth", "answer"]},
            **scores,
            "gates":         gates,
            "failure_types": failure_types,
        })

    summary = {}
    for col in metric_cols:
        vals = [d[col] for d in details if d[col] is not None]
        summary[col] = round(sum(vals) / len(vals), 4) if vals else None

    report = {
        "batch":        batch,
        "sample_count": len(details),
        "summary":      summary,
        "details":      details,
    }

    rr_path = ragas_results_path(batch)
    rr_path.parent.mkdir(parents=True, exist_ok=True)
    with open(rr_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*52}")
    print(f"  RAGAS 评估结果  (batch {batch}，{len(details)} 条)")
    print(f"{'='*52}")
    for col in metric_cols:
        threshold = GATE_THRESHOLDS.get(col, 0)
        val = summary[col]
        marker = "✅" if val and val >= threshold else "⚠️ "
        print(f"  {marker} {col:<22}: {val}")
    print(f"{'='*52}")
    print(f"结果已保存：{rr_path}")


# ─────────────────────────────────────────────────────────────────
# Agent 3：缺口诊断 + 四类报告生成
# ─────────────────────────────────────────────────────────────────

def run_step3(batch: int):
    print(f"\n{'='*60}")
    print(f"  Step 3 / Agent 3 — 缺口诊断与报告生成  (batch={batch})")
    print(f"{'='*60}")

    rr_path = ragas_results_path(batch)
    if not rr_path.exists():
        print(f"❌ 找不到 {rr_path}，请先完成 Step 2")
        sys.exit(1)

    with open(rr_path, encoding="utf-8") as f:
        report = json.load(f)

    details = report["details"]
    summary = report["summary"]

    cr_fails      = [d for d in details if "context_recall" in d["failure_types"]]
    cp_only_fails = [d for d in details if "context_precision" in d["failure_types"]
                     and "context_recall" not in d["failure_types"]]
    f_fails       = [d for d in details if "faithfulness" in d["failure_types"]]
    ar_only_fails = [d for d in details if "answer_relevancy" in d["failure_types"]
                     and "faithfulness" not in d["failure_types"]
                     and "context_recall" not in d["failure_types"]
                     and "context_precision" not in d["failure_types"]]
    all_pass      = [d for d in details if not d["failure_types"]]

    print(f"\n门控统计 (batch {batch}):")
    print(f"  全部通过（高质量）   : {len(all_pass)}")
    print(f"  CR 失败（知识缺口）  : {len(cr_fails)}")
    print(f"  CP 失败（检索排序）  : {len(cp_only_fails)}")
    print(f"  F  失败（接地性）    : {len(f_fails)}")
    print(f"  AR 失败（问题歧义）  : {len(ar_only_fails)}")

    reports_dir = DATA_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    candidates = []
    for d in cr_fails:
        candidates.append({
            "chunk_id":        d["chunk_id"],
            "topic":           d["topic"],
            "question":        d["question"],
            "ground_truth":    d["ground_truth"],
            "context_recall":  d["context_recall"],
            "gap_description": (
                f"Question about '{d['topic']}' cannot be answered from retrieved contexts. "
                f"CR={d['context_recall']}. Ground truth: {d['ground_truth'][:200]}"
            ),
            "action": "review_and_add_to_kb",
            "status": "pending_human_review",
        })

    _write_report("candidates", batch, {
        "batch":       batch,
        "count":       len(candidates),
        "description": "CR 失败样本：知识库缺少对应内容，需人工确认后通过 flywheel_ingest.py 写入 Milvus",
        "items":       candidates,
    })

    cp_topics = {}
    for d in cp_only_fails:
        topic = d["topic"]
        if topic not in cp_topics:
            cp_topics[topic] = {"count": 0, "samples": []}
        cp_topics[topic]["count"] += 1
        cp_topics[topic]["samples"].append({
            "question":          d["question"],
            "context_precision": d["context_precision"],
        })

    retrieval_issues = [
        {"topic": t, "failure_count": v["count"], "samples": v["samples"][:3]}
        for t, v in sorted(cp_topics.items(), key=lambda x: -x[1]["count"])
    ]

    _write_report("retrieval_tuning", batch, {
        "batch":       batch,
        "count":       len(cp_only_fails),
        "description": "CP 失败样本：检索返回的文档精准度不足，建议调整 BM25/向量检索权重或 RRF 融合参数",
        "suggestions": [
            "检查 RRF 的 k 参数（当前检索 top_10，是否过多噪音文档）",
            "评估 BGE reranker 阈值，当前 -5.0 较宽松",
            "分析高频失败 topic，判断是否需要领域特化的 chunk 分割策略",
        ],
        "failure_by_topic": retrieval_issues,
        "raw_samples": [
            {k: d[k] for k in ["chunk_id", "topic", "question",
                                "context_precision", "answer_relevancy"]}
            for d in cp_only_fails
        ],
    })

    _write_report("faithfulness_monitor", batch, {
        "batch":                batch,
        "count":                len(f_fails),
        "description":          "F 失败样本：模型在接地约束下仍使用了参数化知识，需持续监控",
        "overall_faithfulness": summary.get("faithfulness"),
        "threshold":            GATE_THRESHOLDS["faithfulness"],
        "status":               "ok" if (summary.get("faithfulness") or 0) >= GATE_THRESHOLDS["faithfulness"] else "below_threshold",
        "raw_samples": [
            {k: d[k] for k in ["chunk_id", "topic", "question", "faithfulness", "answer"]}
            for d in f_fails
        ],
    })

    _write_report("qa_exclusions", batch, {
        "batch":       batch,
        "count":       len(ar_only_fails),
        "description": "AR 失败样本：答案偏离问题或问题歧义，这些 QA 对应从评估集中排除",
        "items": [
            {k: d[k] for k in ["chunk_id", "topic", "question", "answer_relevancy", "question_type"]}
            for d in ar_only_fails
        ],
    })

    print(f"\n四类报告已生成至 {reports_dir}:")
    print(f"  candidates_batch_{batch}.json         ({len(candidates)} 条待人工确认)")
    print(f"  retrieval_tuning_batch_{batch}.json   ({len(cp_only_fails)} 条检索优化建议)")
    print(f"  faithfulness_monitor_batch_{batch}.json ({len(f_fails)} 条接地性监控)")
    print(f"  qa_exclusions_batch_{batch}.json      ({len(ar_only_fails)} 条排除 QA)")

    if candidates:
        print(f"\n[下一步] 有 {len(candidates)} 条知识缺口候选文档等待人工确认")
        print(f"  确认后运行：python flywheel_ingest.py --batch {batch}")

    print(f"\nStep 3 完成 ✅")


def _write_report(name: str, batch: int, data: dict):
    path = report_path(name, batch)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ─────────────────────────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="数据飞轮主运行脚本")
    parser.add_argument("--step",  type=int, choices=[1, 2, 3], default=None,
                        help="只运行指定步骤（不指定则顺序运行全部步骤）")
    parser.add_argument("--batch", type=int, default=1,
                        help="批次号（Step 2/3 必须与 Step 1 对应）")
    parser.add_argument("--total", type=int, default=200,
                        help="Step 1 目标 QA 总数（默认 200）")
    args = parser.parse_args()

    batch = args.batch

    if args.step == 1:
        run_step1(batch, args.total)

    elif args.step == 2:
        run_step2(batch)

    elif args.step == 3:
        run_step3(batch)

    else:
        # 完整链路：Step 1 → Claude 验证（人工）→ Step 2 → Step 3
        run_step1(batch, args.total)
        print()
        print("─" * 60)
        input(f"[暂停] 请在 Claude 对话窗口完成验证，将结果保存为\n"
              f"  {qa_path(batch)}\n然后按 Enter 继续 Step 2...")
        print("─" * 60)
        run_step2(batch)
        run_step3(batch)


if __name__ == "__main__":
    main()
