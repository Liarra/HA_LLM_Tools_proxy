from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy

import numpy as np
import pytest

from tools_storage import ToolSelector, _retrieval_text, _tool_text, tool_name


@pytest.mark.parametrize(
    "identifier,domain,action",
    [
        ("vacuum__HassVacuumStart", "vacuum", "vacuum start"),
        ("media_player__HassSetVolumeRelative", "media player", "set volume relative"),
        ("homeassistant__GetLiveContext", "", "get live context"),
        ("HomeAssistant__HomeAssistantGetHTTPStatus", "", "get http status"),
        ("HassTurnOn", "", "turn on"),
        ("script__clean_chassis", "script", "clean chassis"),
    ],
)
def test_identifier_embedding(identifier, domain, action):
    original = tool(identifier, "Operate through HomeAssistant and Hass tools")
    before = deepcopy(original)
    text = _tool_text(original)
    assert text.splitlines()[:2] == [f"Domain: {domain}", f"Action: {action}"]
    assert "HomeAssistant" not in text
    assert "Hass" not in text
    assert original == before


def test_retrieval_query_normalizes_ids_without_changing_ordinary_words():
    assert _retrieval_text("Use media_player__HassSetVolumeRelative") == (
        "Use media player set volume relative"
    )
    assert _retrieval_text("Clean the chassis") == "Clean the chassis"


def tool(name: str, description: str = "", *, kind: str = "string") -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description or name,
            "parameters": {
                "type": "object",
                "properties": {"target": {"type": kind}},
            },
        },
    }


class FakeEmbedder:
    model_name = "fake-e5"

    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores
        self.passage_batches: list[list[str]] = []
        self.queries: list[str] = []

    def encode_query(self, text: str) -> np.ndarray:
        self.queries.append(text)
        return np.array([1.0], dtype="float32")

    def encode_passages(self, texts: Sequence[str]) -> np.ndarray:
        self.passage_batches.append(list(texts))
        return np.array(
            [
                [next(score for name, score in self.scores.items() if name in text)]
                for text in texts
            ],
            dtype="float32",
        )


def names(tools: list[dict]) -> list[str]:
    return [tool_name(item) for item in tools]


def test_normalization_is_retrieval_only(monkeypatch):
    original = tool("vacuum__HassVacuumStart", "Start vacuum with HomeAssistant")
    original["function"]["parameters"]["properties"]["target"]["description"] = (
        "HomeAssistant target"
    )
    before = deepcopy(original)
    embedder = FakeEmbedder({"Start vacuum": 0.9})
    selector = ToolSelector(embedder=embedder, cache_path=None, whitelisted_names=())
    result = selector.select([original], "Use vacuum__HassVacuumStart")
    assert embedder.queries == ["Use vacuum vacuum start"]
    assert "HomeAssistant" not in embedder.passage_batches[0][0]
    assert result.tools == [before]
    assert original == before
    key = selector._cache_key(original)
    monkeypatch.setattr("tools_storage._EMBEDDING_TEXT_VERSION", "2")
    assert selector._cache_key(original) != key


def test_tool_text_omits_common_target_parameters() -> None:
    light_tool = {
        "type": "function",
        "function": {
            "name": "HassLightSet",
            "description": "Sets the brightness of a light",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "area": {"type": "string"},
                    "floor": {"type": "string"},
                    "brightness": {
                        "type": "integer",
                        "description": "Brightness percentage",
                    },
                },
                "required": ["name", "brightness"],
            },
        },
    }

    text = _tool_text(light_tool)

    assert '"name"' not in text
    assert '"area"' not in text
    assert '"floor"' not in text
    assert '"brightness"' in text
    assert '"required":["brightness"]' in text


def test_threshold_can_return_fewer_than_maximum() -> None:
    selector = ToolSelector(
        max_tools=3,
        min_similarity=0.75,
        whitelisted_names=("GetLiveContext",),
        blacklisted_names=(),
        embedder=FakeEmbedder({"TurnOnLight": 0.91, "PlayMusic": 0.62}),
        cache_path=None,
    )

    result = selector.select(
        [tool("GetLiveContext"), tool("TurnOnLight"), tool("PlayMusic")],
        "turn on the kitchen light",
    )

    assert names(result.tools) == ["GetLiveContext", "TurnOnLight"]
    assert len(result.tools) < selector.max_tools


def test_whitelist_counts_toward_hard_maximum() -> None:
    selector = ToolSelector(
        max_tools=2,
        min_similarity=-1,
        whitelisted_names=("AlwaysOne", "AlwaysTwo", "AlwaysThree"),
        blacklisted_names=(),
        embedder=FakeEmbedder({"Other": 1.0}),
        cache_path=None,
    )

    result = selector.select(
        [tool("AlwaysOne"), tool("AlwaysTwo"), tool("AlwaysThree"), tool("Other")],
        "anything",
    )

    assert names(result.tools) == ["AlwaysOne", "AlwaysTwo"]


def test_named_tool_choice_beats_whitelist_and_blacklist() -> None:
    selector = ToolSelector(
        max_tools=1,
        min_similarity=1,
        whitelisted_names=("GetLiveContext",),
        blacklisted_names=("UnlockDoor",),
        embedder=FakeEmbedder({"Other": 0.1}),
        cache_path=None,
    )

    result = selector.select(
        [tool("GetLiveContext"), tool("UnlockDoor"), tool("Other")],
        "unlock the door",
        required_names=("UnlockDoor",),
    )

    assert names(result.tools) == ["UnlockDoor"]


def test_required_choice_keeps_one_tool_below_threshold() -> None:
    selector = ToolSelector(
        max_tools=3,
        min_similarity=0.99,
        whitelisted_names=(),
        blacklisted_names=(),
        embedder=FakeEmbedder({"TurnOnLight": 0.4, "PlayMusic": 0.2}),
        cache_path=None,
    )

    result = selector.select(
        [tool("TurnOnLight"), tool("PlayMusic")],
        "do something",
        require_at_least_one=True,
    )

    assert names(result.tools) == ["TurnOnLight"]


def test_required_choice_keeps_one_tool_when_query_is_empty() -> None:
    selector = ToolSelector(
        max_tools=3,
        min_similarity=0.99,
        whitelisted_names=(),
        blacklisted_names=(),
        embedder=FakeEmbedder({"TurnOnLight": 0.4}),
        cache_path=None,
    )

    result = selector.select([tool("TurnOnLight")], "", require_at_least_one=True)

    assert names(result.tools) == ["TurnOnLight"]


def test_cache_key_includes_full_schema(tmp_path) -> None:
    embedder = FakeEmbedder({"TurnOnLight": 0.9})
    selector = ToolSelector(
        max_tools=1,
        min_similarity=0,
        whitelisted_names=(),
        blacklisted_names=(),
        embedder=embedder,
        cache_path=tmp_path / "embeddings.sqlite3",
    )

    selector.select([tool("TurnOnLight", kind="string")], "turn it on")
    selector.select([tool("TurnOnLight", kind="string")], "turn it on")
    selector.select([tool("TurnOnLight", kind="integer")], "turn it on")

    assert len(embedder.passage_batches) == 2


def test_cache_key_includes_parameters_omitted_from_embedding_text(tmp_path) -> None:
    embedder = FakeEmbedder({"TurnOnLight": 0.9})
    selector = ToolSelector(
        max_tools=1,
        min_similarity=0,
        whitelisted_names=(),
        blacklisted_names=(),
        embedder=embedder,
        cache_path=tmp_path / "embeddings.sqlite3",
    )
    string_name = tool("TurnOnLight")
    string_name["function"]["parameters"]["properties"] = {"name": {"type": "string"}}
    integer_name = tool("TurnOnLight")
    integer_name["function"]["parameters"]["properties"] = {"name": {"type": "integer"}}

    selector.select([string_name], "turn it on")
    selector.select([integer_name], "turn it on")

    assert embedder.passage_batches[0] == embedder.passage_batches[1]
    assert len(embedder.passage_batches) == 2


def test_only_tools_in_current_request_can_be_selected() -> None:
    selector = ToolSelector(
        max_tools=3,
        min_similarity=0,
        whitelisted_names=(),
        blacklisted_names=(),
        embedder=FakeEmbedder({"CurrentTool": 0.8}),
        cache_path=None,
    )

    first = selector.select([tool("CurrentTool")], "first request")
    second = selector.select([], "second request")

    assert names(first.tools) == ["CurrentTool"]
    assert second.tools == []
