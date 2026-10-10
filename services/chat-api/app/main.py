"""FastAPI composition root for the multi-provider MCP chat gateway."""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from .chat_service import ChatCompletionService
from .config import Settings
from .errors import ChatServiceError
from .mcp_gateway import MCPToolGateway
from .models import ChatRequest, ChatResponse, ModelsResponse
from .providers import ProviderRegistry


class ApplicationContainer:
    """Construct and own application-scoped services."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.providers = ProviderRegistry(settings)
        self.chat_service = ChatCompletionService(
            self.providers,
            lambda: MCPToolGateway(settings.mcp_urls),
        )


def create_app(container: ApplicationContainer) -> FastAPI:
    app = FastAPI(title="Multi-provider MCP Chat API", version="3.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(ChatServiceError)
    async def chat_service_error(_request: Request, exc: ChatServiceError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models", response_model=ModelsResponse)
    async def models() -> ModelsResponse:
        return container.providers.catalog()

    @app.post("/v1/chat/completions", response_model=None)
    async def chat(request: ChatRequest) -> ChatResponse | StreamingResponse:
        if request.stream:
            return StreamingResponse(
                container.chat_service.stream(request),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        return await container.chat_service.complete(request)

    return app


settings = Settings()
container = ApplicationContainer(settings)
app = create_app(container)
