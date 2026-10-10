"""SSE encoding and incremental tool-call reconstruction."""
import json
from typing import Any

from fastapi.encoders import jsonable_encoder

from .serialization import JsonValueSerializer


class ServerSentEventEncoder:
    @staticmethod
    def encode(event: str, payload: Any) -> str:
        return f"event: {event}\ndata: {json.dumps(jsonable_encoder(payload))}\n\n"


class ToolCallAccumulator:
    """Reconstruct fragmented Chat Completions tool-call deltas."""

    def __init__(self, serializer: JsonValueSerializer | None = None):
        self._serializer = serializer or JsonValueSerializer()
        self._pending: dict[int, dict[str, Any]] = {}

    def add(self, delta_call: Any) -> None:
        entry = self._pending.setdefault(
            delta_call.index,
            {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
        )
        if delta_call.id:
            entry["id"] = delta_call.id
        function = delta_call.function
        if function and function.name:
            entry["function"]["name"] += function.name
        if function and function.arguments:
            entry["function"]["arguments"] += function.arguments
        extra_content = getattr(delta_call, "extra_content", None)
        if extra_content:
            entry["extra_content"] = self._serializer.encode(extra_content)

    @property
    def has_calls(self) -> bool:
        return bool(self._pending)

    def completed(self, round_index: int) -> list[dict[str, Any]]:
        calls = []
        for index, call in sorted(self._pending.items()):
            call["id"] = call["id"] or f"call_{round_index}_{index}"
            calls.append(call)
        return calls
