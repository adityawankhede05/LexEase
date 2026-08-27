from app.ai.base_provider import BaseAIProvider
from app.ai.cerebras_provider import CerebrasProvider
from app.ai.exceptions import AIProviderError
from app.ai.gemini_provider import GeminiProvider
from app.ai.groq_provider import GroqProvider
from app.ai.openrouter_provider import OpenRouterProvider
from app.core.config import settings


def get_ai_provider(provider_name: str | None = None) -> BaseAIProvider:
    """
    Factory function that instantiates and returns the configured AI provider.

    Args:
        provider_name: Optional provider name override. If None, reads settings.AI_PROVIDER.

    Returns:
        An instance of BaseAIProvider (GeminiProvider, GroqProvider, OpenRouterProvider,
        or CerebrasProvider).

    Raises:
        AIProviderError: If the specified or configured provider is unsupported.
    """
    active_provider = (provider_name or settings.AI_PROVIDER).strip().lower()

    if active_provider == "gemini":
        return GeminiProvider()
    elif active_provider == "groq":
        return GroqProvider()
    elif active_provider == "openrouter":
        return OpenRouterProvider()
    elif active_provider == "cerebras":
        return CerebrasProvider()
    else:
        raise AIProviderError(
            f"Unsupported AI provider: '{active_provider}'. "
            f"Supported providers are: 'gemini', 'groq', 'openrouter', 'cerebras'."
        )
