"""Smart tokenizer that auto-detects the best tokenizer based on model name."""
import logging
from typing import Optional
from functools import lru_cache
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


# Cache tokenizers to avoid reloading them repeatedly
@lru_cache(maxsize=2)
def _get_cached_tokenizer(tokenizer_name: str):
    """Get a cached tokenizer instance."""
    try:
        return AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    except Exception as e:
        logger.warning(f"Failed to load tokenizer {tokenizer_name}: {e}")
        logger.warning("Falling back to default tokenizer - GPT-2")
        # Fallback to GPT-2 tokenizer
        return AutoTokenizer.from_pretrained("gpt2")


def detect_tokenizer_from_model(model_name: str) -> str:
    """
    Detect the appropriate tokenizer based on the model name.
    Returns the HuggingFace model ID for the tokenizer.
    """
    model_lower = model_name.lower()

    # OpenAI models
    if any(x in model_lower for x in ['gpt-4', 'gpt-3.5', 'chatgpt']):
        return "Xenova/gpt-4"  # or "gpt2" as fallback

    # Anthropic Claude
    elif 'claude' in model_lower:
        return "gpt2"  # Claude uses similar tokenization

    # Meta Llama models
    elif any(x in model_lower for x in ['llama', 'code-llama', 'llama2', 'llama3']):
        if 'llama-3' in model_lower or 'llama3' in model_lower:
            return "meta-llama/Meta-Llama-3-8B"
        elif 'llama-2' in model_lower or 'llama2' in model_lower:
            return "meta-llama/Llama-2-7b-hf"
        else:
            return "meta-llama/Llama-2-7b-hf"

    # Mistral models
    elif any(x in model_lower for x in ['mistral', 'mixtral']):
        return "mistralai/Mistral-7B-v0.1"

    # Qwen models (you mentioned these in README)
    elif 'qwen' in model_lower:
        return "Qwen/Qwen-7B"

    # DeepSeek models (also mentioned in README)
    elif 'deepseek' in model_lower:
        return "deepseek-ai/deepseek-coder-6.7b-base"

    # Cohere Command models
    elif 'command' in model_lower or 'cohere' in model_lower:
        return "gpt2"  # Fallback

    # Google models (Gemini, PaLM, etc.)
    elif any(x in model_lower for x in ['gemini', 'palm', 'bard']):
        return "gpt2"  # Fallback

    # Default fallback - GPT-2 is widely compatible
    else:
        logger.info(f"Unknown model {model_name}, falling back to GPT-2 tokenizer")
        return "gpt2"


class SmartTokenizer:
    """
    A smart tokenizer that automatically selects the appropriate tokenizer
    based on the model name from the OpenAI API request.
    """

    def __init__(self):
        self.current_model = None
        self.current_tokenizer = None
        # Pre-load GPT-2 as default fallback
        self.fallback_tokenizer = _get_cached_tokenizer("gpt2")

    def update_model(self, model_name: str) -> None:
        """Update the tokenizer if the model has changed."""
        if self.current_model != model_name:
            self.current_model = model_name
            tokenizer_name = detect_tokenizer_from_model(model_name)
            self.current_tokenizer = _get_cached_tokenizer(tokenizer_name)
            logger.info(f"Switched to tokenizer for model: {model_name} -> {tokenizer_name}")

    def count_tokens(self, text: str, model_name: Optional[str] = None) -> int:
        """
        Count tokens in text using the appropriate tokenizer.

        Args:
            text: Text to tokenize
            model_name: Optional model name to auto-select tokenizer

        Returns:
            Number of tokens
        """
        if model_name:
            self.update_model(model_name)

        tokenizer = self.current_tokenizer or self.fallback_tokenizer

        try:
            return len(tokenizer.encode(text, add_special_tokens=True))
        except Exception as e:
            logger.warning(f"Tokenization failed: {e}, using fallback heuristic")
            # Simple fallback estimation
            return len(text) // 4 + text.count(' ') + 1
