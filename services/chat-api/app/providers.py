"""Model-vendor configuration, validation, and client construction."""
from dataclasses import dataclass, field

from openai import AsyncOpenAI

from .config import Settings
from .errors import ProviderConfigurationError
from .models import ModelOption, ModelsResponse, ProviderId, ProviderOption


@dataclass(frozen=True)
class ProviderDefinition:
    id: ProviderId
    name: str
    models: tuple[str, ...]
    api_key: str | None = field(repr=False)
    base_url: str | None = None

    @property
    def configured(self) -> bool:
        return bool(self.api_key)


class ProviderRegistry:
    """Own the supported providers and enforce their model allowlists."""

    def __init__(self, settings: Settings):
        self._providers: dict[ProviderId, ProviderDefinition] = {
            "openai": ProviderDefinition(
                id="openai",
                name="OpenAI (ChatGPT)",
                models=self._parse_models(settings.openai_models),
                api_key=settings.openai_api_key,
            ),
            "gemini": ProviderDefinition(
                id="gemini",
                name="Google Gemini",
                models=self._parse_models(settings.gemini_models),
                api_key=settings.gemini_api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            ),
        }

    @staticmethod
    def _parse_models(raw_models: str) -> tuple[str, ...]:
        return tuple(model.strip() for model in raw_models.split(",") if model.strip())

    @staticmethod
    def _model_name(model: str) -> str:
        return " ".join("GPT" if part == "gpt" else part.capitalize() for part in model.split("-"))

    def catalog(self) -> ModelsResponse:
        providers = [self._providers["openai"], self._providers["gemini"]]
        return ModelsResponse(
            providers=[
                ProviderOption(
                    id=provider.id,
                    name=provider.name,
                    configured=provider.configured,
                    models=[ModelOption(id=model, name=self._model_name(model)) for model in provider.models],
                )
                for provider in providers
            ],
            default_provider="gemini" if providers[1].configured and not providers[0].configured else "openai",
        )

    def models_for(self, provider_id: ProviderId) -> tuple[str, ...]:
        """Return the server-controlled model allowlist for one provider."""

        return self._providers[provider_id].models

    def resolve_model(self, provider_id: ProviderId, requested_model: str | None) -> str:
        provider = self._providers[provider_id]
        if not provider.models:
            raise ProviderConfigurationError(f"No models are configured for {provider_id}.", 503)
        model = requested_model or provider.models[0]
        if model not in provider.models:
            raise ProviderConfigurationError(f"Model {model!r} is not enabled for {provider_id}.", 400)
        if not provider.api_key:
            env_name = "OPENAI_API_KEY" if provider_id == "openai" else "GEMINI_API_KEY"
            raise ProviderConfigurationError(f"{env_name} is not configured on the server.", 503)
        return model

    def client_for(self, provider_id: ProviderId) -> AsyncOpenAI:
        provider = self._providers[provider_id]
        options = {"api_key": provider.api_key}
        if provider.base_url:
            options["base_url"] = provider.base_url
        return AsyncOpenAI(**options)
