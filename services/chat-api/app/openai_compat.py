"""OpenAI-compatible HTTP contracts used by the LibreChat frontend."""
import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from .chat_service import ChatCompletionService, ChatStreamEvent
from .models import ChatRequest, Message, ProviderId
from .providers import ProviderRegistry


class CompatibleMessage(BaseModel):
    """Subset of an OpenAI message that this text tutorial supports."""

    role: Literal["system", "developer", "user", "assistant"]
    content: str | list[dict[str, Any]]

    @field_validator("content")
    @classmethod
    def require_text(cls, content: str | list[dict[str, Any]]):
        if isinstance(content, str):
            text = content
        else:
            text = "\n".join(
                str(part["text"])
                for part in content
                if part.get("type") == "text" and str(part.get("text", "")).strip()
            )
        if not text.strip():
            raise ValueError("Message content must include text.")
        if len(text) > 12_000:
            raise ValueError("Message text must not exceed 12000 characters.")
        return content

    def text(self) -> str:
        if isinstance(self.content, str):
            return self.content
        return "\n".join(
            part["text"]
            for part in self.content
            if part.get("type") == "text" and isinstance(part.get("text"), str)
        )


class CompatibleChatRequest(BaseModel):
    """Accept common Chat Completions fields and ignore unsupported extras."""

    model: str = Field(min_length=1, max_length=100)
    messages: list[CompatibleMessage] = Field(min_length=1, max_length=40)
    stream: bool = False


class OpenAICompatibilityAdapter:
    """Translate between LibreChat's OpenAI protocol and the tutorial API."""

    def __init__(self, providers: ProviderRegistry, chat_service: ChatCompletionService):
        self._providers = providers
        self._chat_service = chat_service

    def models(self, provider: ProviderId) -> dict[str, Any]:
        data = [
            {
                "id": model,
                "object": "model",
                "created": 0,
                "owned_by": provider,
            }
            for model in self._providers.models_for(provider)
        ]
        return {"object": "list", "data": data}

    async def complete(
        self,
        provider: ProviderId,
        request: CompatibleChatRequest,
    ) -> dict[str, Any]:
        internal_request = self._to_internal_request(provider, request)
        response = await self._chat_service.complete(internal_request)
        return {
            "id": self._completion_id(),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": response.model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": response.answer},
                "finish_reason": "stop",
            }],
        }

    def stream(
        self,
        provider: ProviderId,
        request: CompatibleChatRequest,
    ) -> AsyncIterator[str]:
        internal_request = self._to_internal_request(provider, request)
        events = self._chat_service.events(internal_request)
        return self._openai_events(events, internal_request.model or request.model)

    def _to_internal_request(
        self,
        provider: ProviderId,
        request: CompatibleChatRequest,
    ) -> ChatRequest:
        messages = []
        for source in request.messages:
            content = source.text().strip()
            if content:
                role = "system" if source.role == "developer" else source.role
                messages.append(Message(role=role, content=content))
        return ChatRequest(
            provider=provider,
            model=request.model,
            messages=messages,
            stream=request.stream,
        )

    async def _openai_events(
        self,
        events: AsyncIterator[ChatStreamEvent],
        model: str,
    ) -> AsyncIterator[str]:
        completion_id = self._completion_id()
        created = int(time.time())
        first_delta = True
        async for event in events:
            if event.name == "delta":
                delta = {"content": event.payload["content"]}
                if first_delta:
                    delta["role"] = "assistant"
                    first_delta = False
                yield self._data({
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "delta": delta,
                        "finish_reason": None,
                    }],
                })
            elif event.name == "error":
                yield self._data({
                    "error": {
                        "message": event.payload["detail"],
                        "type": "upstream_error",
                    },
                })
                yield "data: [DONE]\n\n"
                return
            elif event.name == "done":
                yield self._data({
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }],
                })
                yield "data: [DONE]\n\n"
                return

    @staticmethod
    def _completion_id() -> str:
        return f"chatcmpl-{uuid.uuid4().hex}"

    @staticmethod
    def _data(payload: dict[str, Any]) -> str:
        return f"data: {json.dumps(payload)}\n\n"
