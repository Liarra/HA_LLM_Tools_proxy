from __future__ import annotations

import json

from diagnostics import DebugDiagnostics


class FakeCounter:
    def __init__(self) -> None:
        self.calls = 0

    def count_many(self, texts, request_model):
        self.calls += 1
        return (
            {name: len(text) for name, text in texts.items()},
            "fake-tokenizer",
            False,
        )


def test_debug_record_contains_bodies_and_before_after_counts(tmp_path) -> None:
    counter = FakeCounter()
    diagnostics = DebugDiagnostics(enabled=True, log_dir=tmp_path, counter=counter)
    received = {
        "model": "local/qwen3",
        "messages": [{"role": "user", "content": "turn on the light"}],
        "tools": [
            {"function": {"name": "TurnOnLight"}},
            {"function": {"name": "Other"}},
        ],
    }
    forwarded = {**received, "tools": [received["tools"][0]]}

    path = diagnostics.record(received, forwarded)

    assert path is not None
    assert path.parent == tmp_path
    assert path.name.startswith("local_qwen3_")
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["received"] == received
    assert record["forwarded"] == forwarded
    assert record["token_counts"]["received"]["tokenizer_model"] == "fake-tokenizer"
    assert record["token_counts"]["request_tokens_saved"] > 0
    assert counter.calls == 2


def test_disabled_diagnostics_do_no_work(tmp_path) -> None:
    counter = FakeCounter()
    diagnostics = DebugDiagnostics(enabled=False, log_dir=tmp_path, counter=counter)

    assert diagnostics.record({}, {}) is None
    assert counter.calls == 0
    assert list(tmp_path.iterdir()) == []
