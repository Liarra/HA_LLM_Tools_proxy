"""Lazy E5 embedding support for tool selection."""

from __future__ import annotations

from collections.abc import Sequence
from threading import Lock

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


class SemanticEmbedder:
    """Encode user queries and tool definitions using the E5 conventions."""

    def __init__(self, model_name: str = "intfloat/e5-small-v2") -> None:
        self.model_name = model_name
        self._tokenizer = None
        self._model = None
        self._load_lock = Lock()

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name)
            self._model.eval()

    @staticmethod
    def _mean_pool(last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        return (last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        self._ensure_loaded()
        assert self._tokenizer is not None
        assert self._model is not None

        inputs = self._tokenizer(
            list(texts),
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        )
        with torch.no_grad():
            output = self._model(**inputs)
        vectors = self._mean_pool(output.last_hidden_state, inputs["attention_mask"])
        vectors = vectors.cpu().numpy().astype("float32")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norms, 1e-12)

    def encode_query(self, text: str) -> np.ndarray:
        """Encode one retrieval query."""
        return self._encode([f"query: {text}"])[0]

    def encode_passages(self, texts: Sequence[str]) -> np.ndarray:
        """Encode tool definitions as retrieval passages."""
        if not texts:
            return np.empty((0, 0), dtype="float32")
        return self._encode([f"passage: {text}" for text in texts])
