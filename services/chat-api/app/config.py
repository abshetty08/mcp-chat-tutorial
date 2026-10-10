"""Runtime configuration for the chat gateway."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-backed application settings."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str | None = None
    gemini_api_key: str | None = None
    openai_models: str = "gpt-6-luna,gpt-6.1-sol,gpt-6-astra"
    gemini_models: str = "gemini-3.8-flash,gemini-3.5-flash-lite,gemini-3.1-pro-preview"
    mcp_search_url: str = "http://localhost:8001/mcp"
    mcp_market_url: str = "http://localhost:8002/mcp"

    @property
    def mcp_urls(self) -> tuple[str, ...]:
        return self.mcp_search_url, self.mcp_market_url
