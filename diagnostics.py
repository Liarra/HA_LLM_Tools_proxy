"""Opt-in request capture and token accounting for debug mode."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import re
from typing import Protocol
from uuid import uuid4

from smart_tokenizer import SmartTokenizer

logger = logging.getLogger(__name__)


class TokenCounter(Protocol):
    """Interface used by debug diagnostics and its tests."""

    def count_many(
        self, texts: Mapping[str, str], request_model: str
    ) -> tuple[dict[str, int], str, bool]: ...


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _token_sections(body: dict) -> dict[str, str]:
    messages = body.get("messages", [])
    if not isinstance(messages, list):
        messages = []

    system_messages = [
        message
        for message in messages
        if isinstance(message, dict) and message.get("role") in ("system", "developer")
    ]
    user_messages = [
        message
        for message in messages
        if isinstance(message, dict) and message.get("role") == "user"
    ]
    return {
        "system": _json(system_messages),
        "user": _json(user_messages),
        "messages": _json(messages),
        "tools": _json(body.get("tools", [])),
        "request_total": _json(body),
    }


class DebugDiagnostics:
    """Write received/forwarded bodies and token estimates only when enabled."""

    def __init__(
        self,
        *,
        enabled: bool,
        log_dir: Path = Path("logs"),
        counter: TokenCounter | None = None,
    ) -> None:
        self.enabled = enabled
        self.log_dir = log_dir
        self.counter = counter or SmartTokenizer()

    def _counts(self, body: dict) -> dict[str, object]:
        request_model = str(body.get("model", ""))
        counts, tokenizer_name, approximate = self.counter.count_many(
            _token_sections(body), request_model
        )
        return {
            "request_model": request_model,
            "tokenizer_model": tokenizer_name,
            "approximate": approximate,
            **counts,
        }

    def record(self, received: dict, forwarded: dict) -> Path | None:
        """Persist one structured diagnostic record and log its token delta."""
        if not self.enabled:
            return None

        received_counts = self._counts(received)
        forwarded_counts = self._counts(forwarded)
        saved = int(received_counts["request_total"]) - int(
            forwarded_counts["request_total"]
        )
        record = {
            "received": received,
            "forwarded": forwarded,
            "token_counts": {
                "received": received_counts,
                "forwarded": forwarded_counts,
                "request_tokens_saved": saved,
            },
        }

        self.log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        model = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(received.get("model", "model")))
        path = self.log_dir / f"{model}_{timestamp}_{uuid4().hex[:8]}.json"
        with path.open("x", encoding="utf-8") as handle:
            json.dump(record, handle, ensure_ascii=False, indent=2)

        logger.debug(
            "Token counts received=%d forwarded=%d saved=%d; tools=%d -> %d; log=%s",
            received_counts["request_total"],
            forwarded_counts["request_total"],
            saved,
            received_counts["tools"],
            forwarded_counts["tools"],
            path,
        )
        return path
