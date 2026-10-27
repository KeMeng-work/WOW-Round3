"""
ONNX Runtime nomic-embed-text 嵌入器
从生产项目复制，实现 LangChain Embeddings 接口供 RAGAS AnswerRelevancy 使用。
推理链：tokenizer → session.run() → mean_pool → l2_normalize → 768 维向量
"""

import os
import logging
from typing import List

import numpy as np
import onnxruntime as ort
from langchain_core.embeddings import Embeddings
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


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
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = intra_op_num_threads
        sess_options.inter_op_num_threads = inter_op_num_threads
        self.session = ort.InferenceSession(onnx_path, sess_options, providers=providers)
        self._input_names = {inp.name for inp in self.session.get_inputs()}
        self._warmup()

    def _warmup(self):
        try:
            self.embed_query("warmup")
        except Exception as e:
            logger.warning(f"ONNXEmbedder 预热失败: {e}")

    def _mean_pool(self, hidden_states: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        mask = attention_mask[:, :, np.newaxis].astype(np.float32)
        sum_emb  = (hidden_states * mask).sum(axis=1)
        sum_mask = mask.sum(axis=1).clip(min=1e-9)
        return sum_emb / sum_mask

    def _l2_normalize(self, embeddings: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(embeddings, axis=-1, keepdims=True).clip(min=1e-9)
        return embeddings / norms

    def _embed_batch(self, texts: List[str]) -> np.ndarray:
        encoded = self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=self.max_length, return_tensors="np",
        )
        inputs = {
            "input_ids":      encoded["input_ids"].astype(np.int64),
            "attention_mask": encoded["attention_mask"].astype(np.int64),
        }
        if "token_type_ids" in self._input_names and "token_type_ids" in encoded:
            inputs["token_type_ids"] = encoded["token_type_ids"].astype(np.int64)
        hidden_states = self.session.run(None, inputs)[0]
        return self._l2_normalize(self._mean_pool(hidden_states, encoded["attention_mask"]))

    def embed_query(self, text: str) -> List[float]:
        return self._embed_batch([text])[0].tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        results: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            results.extend(self._embed_batch(texts[i: i + self.batch_size]).tolist())
        return results
