from __future__ import annotations

import numpy as np

from embedding import SemanticEmbedder


def test_e5_query_and_passage_prefixes(monkeypatch) -> None:
    embedder = SemanticEmbedder()
    seen: list[list[str]] = []

    def fake_encode(texts):
        seen.append(list(texts))
        return np.ones((len(texts), 1), dtype="float32")

    monkeypatch.setattr(embedder, "_encode", fake_encode)

    embedder.encode_query("turn on the light")
    embedder.encode_passages(["TurnOnLight", "PlayMusic"])

    assert seen == [
        ["query: turn on the light"],
        ["passage: TurnOnLight", "passage: PlayMusic"],
    ]
