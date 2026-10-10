"""Domain errors translated to HTTP responses by the application layer."""


class ChatServiceError(Exception):
    def __init__(self, detail: str, status_code: int = 502):
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


class ProviderConfigurationError(ChatServiceError):
    """The requested provider or model is unavailable or misconfigured."""


class ProviderRequestError(ChatServiceError):
    """A model vendor rejected or failed a completion request."""


class ToolServiceError(ChatServiceError):
    """An MCP connection or tool execution failed."""
