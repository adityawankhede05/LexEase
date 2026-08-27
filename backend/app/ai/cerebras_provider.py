import asyncio
import logging
from typing import Any
import httpx

from app.ai.base_provider import BaseAIProvider
from app.ai.exceptions import AIProviderError
from app.core.config import settings

logger = logging.getLogger(__name__)

CEREBRAS_API_URL = "https://api.cerebras.ai/v1/chat/completions"


class CerebrasProvider(BaseAIProvider):
    """
    Cerebras AI provider implementation conforming to BaseAIProvider.
    Uses async httpx to communicate with the Cerebras OpenAI-compatible
    chat completions endpoint. Implements retry with exponential backoff
    for transient errors (5xx, timeout, network) consistent with existing providers.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
    ):
        self.api_key = api_key or settings.CEREBRAS_API_KEY
        self.model = model or settings.CEREBRAS_MODEL
        self.timeout = timeout or settings.CEREBRAS_TIMEOUT

    async def generate(self, prompt: str, system_instruction: str | None = None) -> str:
        """
        Sends a prompt to the Cerebras chat completions API and returns raw string response.

        Args:
            prompt: The formatted user input prompt for the model.
            system_instruction: Optional system-level guidelines for the model.

        Returns:
            The raw text completion string returned by the model.

        Raises:
            AIProviderError: If generation fails, authentication fails, or retries are exhausted.
        """
        if not self.api_key:
            raise AIProviderError("CEREBRAS_API_KEY is not configured in settings.")

        messages: list[dict[str, str]] = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": settings.GEMINI_TEMPERATURE,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        max_retries = max(1, settings.MAX_RETRIES)
        last_exception: Exception | None = None

        for attempt in range(1, max_retries + 1):
            backoff: float = 2 ** (attempt - 1)
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        CEREBRAS_API_URL,
                        json=payload,
                        headers=headers,
                    )

                if response.status_code == 200:
                    data = response.json()
                    choices = data.get("choices")
                    if choices and isinstance(choices, list) and len(choices) > 0:
                        content = choices[0].get("message", {}).get("content")
                        if content is not None:
                            return str(content)
                    raise AIProviderError(
                        "Cerebras returned empty or malformed completion content."
                    )

                # Non-retryable client errors: 400, 401, 403
                if response.status_code in (400, 401, 403):
                    raise AIProviderError(
                        f"Cerebras API request failed with status {response.status_code}."
                    )

                # Retryable status codes: 429, 500, 502, 503, 504
                if response.status_code in (429, 500, 502, 503, 504):
                    last_exception = AIProviderError(
                        f"Cerebras transient error ({response.status_code})."
                    )
                    logger.warning(
                        f"Cerebras API transient error ({response.status_code}) "
                        f"on attempt {attempt}/{max_retries}."
                    )
                else:
                    raise AIProviderError(
                        f"Cerebras API error with status {response.status_code}."
                    )

            except httpx.TimeoutException as e:
                last_exception = e
                logger.warning(
                    f"Cerebras API request timed out (attempt {attempt}/{max_retries})."
                )
            except httpx.NetworkError as e:
                last_exception = e
                logger.warning(
                    f"Cerebras API network error (attempt {attempt}/{max_retries}): {e}"
                )
            except AIProviderError:
                raise
            except Exception as e:
                last_exception = e
                logger.warning(
                    f"Unexpected error calling Cerebras API (attempt {attempt}/{max_retries}): {e}"
                )

            if attempt < max_retries:
                logger.info(
                    f"Retrying Cerebras API call in {backoff:.2f}s "
                    f"(attempt {attempt}/{max_retries})..."
                )
                await asyncio.sleep(backoff)

        raise AIProviderError(
            f"CerebrasProvider failed after {max_retries} attempts: {str(last_exception)}"
        ) from last_exception
