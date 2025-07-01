from typing import List, Tuple

from transformers import AutoModel, AutoTokenizer
import faiss  # type: ignore
import numpy as _np
import torch

#----------------------------------------------------------------------------
# Embedding → FAISS helper
# ----------------------------------------------------------------------------
_tokenizer = AutoTokenizer.from_pretrained("intfloat/e5-small-v2")
_model = AutoModel.from_pretrained("intfloat/e5-small-v2")
_model.eval()

_DIM = _model.config.hidden_size  # e5-small‑v2 has 384 dims
# Use inner‑product index on L2‑normalised vectors = cosine similarity
_faiss_index = faiss.IndexFlatIP(_DIM)

# Keep the raw texts so we can map vectors back to strings
_text_store: List[str] = []

def _mean_pooling(model_output, attention_mask):
    """Perform mean pooling as recommended for e5-* models."""
    token_embeddings = model_output.last_hidden_state  # (bs, seq_len, hid)
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = (token_embeddings * input_mask_expanded).sum(1)
    sum_mask = input_mask_expanded.sum(1)
    return sum_embeddings / sum_mask.clamp(min=1e-9)


def encode_text(text:str):
    """Encode *text* with e5-small-v2 and return the (1, D) numpy vector."""
    inputs = _tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        model_output = _model(**inputs)
    vec = _mean_pooling(model_output, inputs["attention_mask"])[0].cpu().numpy().astype("float32")
    # Normalise so inner product ≈ cosine similarity
    vec /= _np.linalg.norm(vec) + 1e-12
    return vec

def store_text_into_faiss(text: str, index: faiss.Index = _faiss_index) -> _np.ndarray:
    """Encode *text* with e5-small-v2 and add the embedding to *index*.

    Returns the (1, D) numpy vector that was inserted. You can provide your own
    mutable *index* (must match `e5-small-v2`'s dimensionality, 384) if you
    need multiple separate collections.
    """
    text_vec = encode_text(text)
    index.add(text_vec.reshape(1, -1))  # FAISS needs shape (n, d)
    _text_store.append(text)
    return text_vec

def retrieve_similar(text: str, k: int = 5) -> List[Tuple[int, float, str]]:
    """Return the *k* most similar items to *text* from *index*.

    Each result is a tuple *(rank_id, score, original_text)* where *score* is
    the inner‑product similarity (1.0 is identical, −1.0 opposite). If fewer
    than *k* vectors exist, returns whatever is available.
    """
    if _faiss_index.ntotal == 0:
        return []

    qvec = encode_text(text)
    scores, idxs = _faiss_index.search(qvec.reshape(1, -1), min(k, _faiss_index.ntotal))
    results = []
    for rank, (score, idx) in enumerate(zip(scores[0], idxs[0])):
        original = _text_store[idx] if 0 <= idx < len(_text_store) else "<unknown>"
        results.append((idx, float(score), original))
    return results

def save_to_file(file_path: str):
    """Save the FAISS index to a file."""
    print(f"Saving FAISS index to {file_path}...")
    print(f"ntotal: {_faiss_index.ntotal}, dim: {_faiss_index.d}")
    faiss.write_index(_faiss_index, file_path)

def load_from_file(file_path: str):
    """Load the FAISS index from a file."""
    global _faiss_index, _text_store
    print(f"Loading FAISS index from {file_path}...")
    _faiss_index = faiss.read_index(file_path)
    print(f"ntotal: {_faiss_index.ntotal}, dim: {_faiss_index.d}")

def get_labels() -> List[str]:
    """Return the list of texts currently stored in the FAISS index."""
    return _text_store

def load_labels(labels: List[str]):
    """Load a list of texts into the FAISS index.

    This will clear the existing index and store the new labels.
    """
    global _text_store
    _text_store = labels