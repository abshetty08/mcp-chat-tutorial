"""JSON-safe serialization at SDK and transport boundaries."""
from typing import Any

from fastapi.encoders import jsonable_encoder


class JsonValueSerializer:
    """Normalize Pydantic, SDK, dataclass, and FastMCP result values."""

    def encode(self, value: Any) -> Any:
        data = getattr(value, "data", None)
        if data is not None:
            return self.encode(data)
        structured_content = getattr(value, "structured_content", None)
        if structured_content is not None:
            return self.encode(structured_content)
        if hasattr(value, "model_dump"):
            return value.model_dump(mode="json", by_alias=True)
        if hasattr(value, "dict"):
            return value.dict()
        return jsonable_encoder(value)
