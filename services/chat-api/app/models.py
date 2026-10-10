"""API contracts shared by the HTTP layer and chat service."""
from typing import Any, Literal

from pydantic import BaseModel, Field


ProviderId = Literal["openai", "gemini"]


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=12_000)


class ChatRequest(BaseModel):
    messages: list[Message] = Field(min_length=1, max_length=40)
    provider: ProviderId = "openai"
    model: str | None = Field(default=None, max_length=100)
    stream: bool = False


class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any]
    result: Any


class ChatResponse(BaseModel):
    answer: str
    provider: ProviderId
    model: str
    tool_calls: list[ToolCall]


class ModelOption(BaseModel):
    id: str
    name: str


class ProviderOption(BaseModel):
    id: ProviderId
    name: str
    configured: bool
    models: list[ModelOption]


class ModelsResponse(BaseModel):
    providers: list[ProviderOption]
    default_provider: ProviderId
