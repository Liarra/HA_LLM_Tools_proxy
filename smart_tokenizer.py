"""Lazy tokenizer selection for debug-only token accounting."""

from __future__ import annotations

from collections.abc import Mapping
import logging
import math
from threading import Lock

from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


class SmartTokenizer:
    """Load a suitable tokenizer only when debug diagnostics request one."""

    def __init__(self, tokenizer_model: str = "") -> None:
        self.tokenizer_model = tokenizer_model.strip()
        self._tokenizers: dict[str, object] = {}
        self._failed: set[str] = set()
        self._load_lock = Lock()

    def resolve_model(self, request_model: str) -> str:
        """Resolve a served model alias to a usable tokenizer repository."""
        if self.tokenizer_model:
            return self.tokenizer_model
        if "/" in request_model:
            return request_model

        model = request_model.lower()
        if "qwen" in model or "deepseek" in model:
            return "Qwen/Qwen3-8B"
        if "phi" in model:
            return "microsoft/Phi-3.5-mini-instruct"
        return "openai-community/gpt2"

    def _get_tokenizer(self, model_name: str):
        if model_name in self._tokenizers:
            return self._tokenizers[model_name]
        if model_name in self._failed:
            return None

        with self._load_lock:
            if model_name in self._tokenizers:
                return self._tokenizers[model_name]
            if model_name in self._failed:
                return None
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
            except (OSError, ValueError):
                logger.exception(
                    "Could not load debug tokenizer %s; using a character estimate",
                    model_name,
                )
                self._failed.add(model_name)
                return None
            self._tokenizers[model_name] = tokenizer
            return tokenizer

    def count_many(
        self, texts: Mapping[str, str], request_model: str
    ) -> tuple[dict[str, int], str, bool]:
        """Count several strings with one tokenizer load.

        Returns counts, the resolved tokenizer name, and whether the fallback
        character estimate had to be used.
        """
        tokenizer_name = self.resolve_model(request_model)
        tokenizer = self._get_tokenizer(tokenizer_name)
        if tokenizer is None:
            return (
                {name: math.ceil(len(text) / 4) for name, text in texts.items()},
                tokenizer_name,
                True,
            )
        return (
            {
                name: len(tokenizer.encode(text, add_special_tokens=False))
                for name, text in texts.items()
            },
            tokenizer_name,
            False,
        )
