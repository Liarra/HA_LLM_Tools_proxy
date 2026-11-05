"""Select relevant tools from the tools present in the current request."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Protocol

import numpy as np

from embedding import SemanticEmbedder

logger = logging.getLogger(__name__)

_EMBEDDING_TEXT_VERSION = "3"
_COMMON_TARGET_PARAMETERS = frozenset({"name", "area", "floor"})


class Embedder(Protocol):
    """Small protocol that keeps the selector straightforward to test."""

    model_name: str

    def encode_query(self, text: str) -> np.ndarray: ...

    def encode_passages(self, texts: Sequence[str]) -> np.ndarray: ...


@dataclass(frozen=True)
class Selection:
    """Selected tool definitions and semantic scores by tool name."""

    tools: list[dict]
    scores: dict[str, float]
    trace: list[dict] = field(default_factory=list)


class EmbeddingCache:
    """Persist embeddings atomically in one SQLite database."""

    def __init__(self, path: Path | None) -> None:
        self.path = path
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(path) as connection:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS tool_embeddings (
                        cache_key TEXT PRIMARY KEY,
                        vector BLOB NOT NULL
                    )
                    """
                )

    def get(self, cache_key: str) -> np.ndarray | None:
        if self.path is None:
            return None
        with sqlite3.connect(self.path) as connection:
            row = connection.execute(
                "SELECT vector FROM tool_embeddings WHERE cache_key = ?", (cache_key,)
            ).fetchone()
        if row is None:
            return None
        return np.frombuffer(row[0], dtype="float32").copy()

    def put(self, cache_key: str, vector: np.ndarray) -> None:
        if self.path is None:
            return
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                "INSERT OR REPLACE INTO tool_embeddings(cache_key, vector) VALUES (?, ?)",
                (cache_key, vector.astype("float32").tobytes()),
            )


def tool_name(tool: dict) -> str:
    """Return an OpenAI function tool's name, or an empty string."""
    return str(tool.get("function", {}).get("name", ""))


def _canonical_tool(tool: dict) -> str:
    return json.dumps(tool, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _embedding_parameters(parameters: object) -> object:
    """Remove ubiquitous HA target fields that do not distinguish tools."""
    if not isinstance(parameters, dict):
        return parameters

    properties = parameters.get("properties")
    if not isinstance(properties, dict):
        return parameters

    filtered = dict(parameters)
    filtered["properties"] = {
        name: schema
        for name, schema in properties.items()
        if name not in _COMMON_TARGET_PARAMETERS
    }

    required = parameters.get("required")
    if isinstance(required, list):
        filtered_required = [
            name for name in required if name not in _COMMON_TARGET_PARAMETERS
        ]
        if filtered_required:
            filtered["required"] = filtered_required
        else:
            filtered.pop("required", None)

    return filtered


def _unbrand(text: str) -> str:
    """Remove HA branding without damaging words such as 'chassis'."""
    return re.sub(
        r"(?<![a-zA-Z])(?i:hass|home[ _-]*assistant)(?=$|[^a-zA-Z]|[A-Z])",
        "",
        text,
    )


def _identifier_words(identifier: str) -> str:
    text = _unbrand(identifier.replace("_", " "))
    text = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", text)
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    return " ".join(text.lower().split())


def _retrieval_text(text: str) -> str:
    """Normalise explicit tool IDs and branding in retrieval-only text."""
    text = re.sub(
        r"\b([A-Za-z][A-Za-z0-9_]*)__([A-Za-z][A-Za-z0-9_]*)\b",
        lambda match: " ".join(
            filter(None, (_identifier_words(part) for part in match.groups()))
        ),
        text,
    )
    return _unbrand(text)


def _tool_text(tool: dict) -> str:
    function = tool.get("function", {})
    domain, separator, action = function.get("name", "").partition("__")
    if not separator:
        domain, action = "", domain
    return "\n".join(
        (
            f"Domain: {_identifier_words(domain)}",
            f"Action: {_identifier_words(action)}",
            f"Description: {_retrieval_text(function.get('description', ''))}",
            "Parameters: "
            + _retrieval_text(
                json.dumps(
                    _embedding_parameters(function.get("parameters", {})),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            ),
        )
    )


class ToolSelector:
    """Apply mandatory names, a similarity floor, and a hard tool budget."""

    def __init__(
        self,
        *,
        max_tools: int = 3,
        min_similarity: float = 0.75,
        whitelisted_names: Sequence[str] = ("GetLiveContext",),
        blacklisted_names: Sequence[str] = (
            "HassHumidifierMode",
            "HassHumidifierSetPoint",
        ),
        embedder: Embedder | None = None,
        cache_path: Path | None = Path("data/tool_embeddings.sqlite3"),
    ) -> None:
        if max_tools < 0:
            raise ValueError("max_tools must be zero or greater")
        if not -1.0 <= min_similarity <= 1.0:
            raise ValueError("min_similarity must be between -1 and 1")
        self.max_tools = max_tools
        self.min_similarity = min_similarity
        self.whitelisted_names = tuple(whitelisted_names)
        self.blacklisted_names = frozenset(blacklisted_names)
        self.embedder = embedder or SemanticEmbedder()
        self.cache = EmbeddingCache(cache_path)
        self._selection_lock = Lock()

    def _cache_key(self, tool: dict) -> str:
        value = (
            f"{self.embedder.model_name}\0{_EMBEDDING_TEXT_VERSION}\0"
            f"{_canonical_tool(tool)}"
        )
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def _tool_vectors(self, tools: Sequence[dict]) -> np.ndarray:
        vectors: list[np.ndarray | None] = []
        missing_positions: list[int] = []
        missing_texts: list[str] = []

        for position, tool in enumerate(tools):
            cached = self.cache.get(self._cache_key(tool))
            vectors.append(cached)
            if cached is None:
                missing_positions.append(position)
                missing_texts.append(_tool_text(tool))

        if missing_texts:
            encoded = self.embedder.encode_passages(missing_texts)
            for position, vector in zip(missing_positions, encoded, strict=True):
                vectors[position] = vector
                self.cache.put(self._cache_key(tools[position]), vector)

        if not vectors:
            return np.empty((0, 0), dtype="float32")
        assert all(vector is not None for vector in vectors)
        return np.stack(vectors)  # type: ignore[arg-type]

    def select(
        self,
        tools: Sequence[dict],
        query: str,
        *,
        required_names: Sequence[str] = (),
        require_at_least_one: bool = False,
        messages: Sequence[dict] | None = None,
    ) -> Selection:
        """Select at most ``max_tools`` from only this request's tools.

        Explicitly required tools take priority, followed by configured whitelist
        entries. Both consume the same hard budget as semantic matches.
        """
        if messages is not None:
            return self._select_history(
                tools, messages, required_names, require_at_least_one
            )
        if self.max_tools == 0 or not tools:
            return Selection([], {})

        with self._selection_lock:
            indexed = [
                (position, tool, tool_name(tool)) for position, tool in enumerate(tools)
            ]
            mandatory_order = tuple(
                dict.fromkeys((*required_names, *self.whitelisted_names))
            )
            available_names = {name for _, _, name in indexed}
            available_mandatory = [
                name for name in mandatory_order if name in available_names
            ]
            mandatory: list[dict] = []
            used_names: set[str] = set()
            for name in mandatory_order:
                matching = next(
                    (tool for _, tool, item_name in indexed if item_name == name), None
                )
                if matching is not None and name not in used_names:
                    mandatory.append(matching)
                    used_names.add(name)
                if len(mandatory) == self.max_tools:
                    break

            if (
                len(available_mandatory) > len(mandatory)
                and len(mandatory) == self.max_tools
            ):
                logger.warning(
                    "Mandatory tools exceed TOOLS_TO_KEEP; enforcing the hard maximum"
                )

            remaining = self.max_tools - len(mandatory)
            if remaining == 0:
                return Selection(mandatory, {})

            candidates = [
                tool
                for _, tool, name in indexed
                if name not in used_names and name not in self.blacklisted_names
            ]
            if not candidates:
                return Selection(mandatory, {})

            if not query.strip():
                if require_at_least_one and not mandatory:
                    return Selection([candidates[0]], {})
                return Selection(mandatory, {})

            query_vector = self.embedder.encode_query(_retrieval_text(query))
            tool_vectors = self._tool_vectors(candidates)
            similarities = tool_vectors @ query_vector
            ranked = sorted(
                zip(candidates, similarities, strict=True),
                key=lambda item: float(item[1]),
                reverse=True,
            )

            selected = list(mandatory)
            scores: dict[str, float] = {}
            for tool, raw_score in ranked:
                score = float(raw_score)
                if score < self.min_similarity and not (
                    require_at_least_one and len(selected) == 0
                ):
                    continue
                selected.append(tool)
                scores[tool_name(tool)] = score
                if len(selected) == self.max_tools:
                    break

            return Selection(selected, scores)

    def _select_history(
        self,
        tools: Sequence[dict],
        messages: Sequence[dict],
        required_names: Sequence[str],
        require_at_least_one: bool,
    ) -> Selection:
        """Replay this request only; no mutable conversation state is shared.

        Mandatory tools are pinned and consume budget. Semantic matches refresh
        LRU order; ties favour the input catalog order. Successful tool results
        refresh their matching call, but failed/unknown calls do not.
        """
        pinned = self.select(tools, "", required_names=required_names).tools
        pinned_names = {tool_name(tool) for tool in pinned}
        capacity = self.max_tools - len(pinned)
        if capacity <= 0 or not tools:
            return Selection(pinned, {})
        available = {
            tool_name(tool): tool
            for tool in tools
            if tool_name(tool) not in pinned_names
            and tool_name(tool) not in self.blacklisted_names
        }
        active: OrderedDict[str, dict] = OrderedDict()
        scores: dict[str, float] = {}
        trace: list[dict] = []
        pending: dict[str, str] = {}
        queries: dict[str, Selection] = {}

        def touch(name: str, event: dict) -> None:
            if name not in available:
                return
            refreshed = name in active
            active[name] = available[name]
            active.move_to_end(name)
            evicted = None
            if len(active) > capacity:
                evicted, _ = active.popitem(last=False)
                scores.pop(evicted, None)
            trace.append(
                {**event, "tool": name, "refresh": refreshed, "evicted": evicted}
            )

        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            if role == "user":
                content = message.get("content", "")
                if isinstance(content, list):
                    content = " ".join(
                        part.get("text", "")
                        for part in content
                        if isinstance(part, dict) and isinstance(part.get("text"), str)
                    )
                if not isinstance(content, str) or not content.strip():
                    continue
                if content not in queries:
                    queries[content] = self.select(
                        tools, content, required_names=required_names
                    )
                matches = queries[content]
                trace.append(
                    {"message": index, "query": content, "scores": matches.scores}
                )
                # Best match is most recent; weaker matches must not evict it.
                for tool in reversed(matches.tools):
                    name = tool_name(tool)
                    touch(name, {"message": index, "source": "semantic"})
                    if name in active and name in matches.scores:
                        scores[name] = matches.scores[name]
            elif role == "assistant":
                for call in message.get("tool_calls") or []:
                    if not isinstance(call, dict):
                        continue
                    function = call.get("function")
                    if isinstance(function, dict) and isinstance(call.get("id"), str):
                        name = function.get("name")
                        if isinstance(name, str):
                            pending[call["id"]] = name
            elif role == "tool":
                name = pending.pop(message.get("tool_call_id"), None)
                if name is None:
                    continue
                result = message.get("content")
                if isinstance(result, str):
                    try:
                        result = json.loads(result)
                    except (ValueError, TypeError):
                        # Unknown result format is not evidence of success.
                        continue
                if (
                    not isinstance(result, dict)
                    or result.get("error")
                    or result.get("isError")
                ):
                    continue
                touch(name, {"message": index, "source": "tool_success"})

        selected = pinned + list(reversed(active.values()))
        if not selected and require_at_least_one:
            fallback = self.select(tools, "", require_at_least_one=True)
            return Selection(fallback.tools, {}, trace)
        return Selection(selected, scores, trace)
