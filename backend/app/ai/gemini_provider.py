import asyncio
import logging
from google import genai
from google.genai import types
from google.genai.errors import APIError

from app.ai.base_provider import BaseAIProvider
from app.ai.exceptions import AIProviderError
from app.core.config import settings

logger = logging.getLogger(__name__)

class GeminiProvider(BaseAIProvider):
    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key or settings.GEMINI_API_KEY
        self.model = model or settings.GEMINI_MODEL
        self._client: genai.Client | None = None

    def _get_client(self) -> genai.Client:
        if not self.api_key:
            raise AIProviderError("GEMINI_API_KEY is not configured in settings.")
        if self._client is None:
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    async def generate(self, prompt: str, system_instruction: str | None = None) -> str:
        """
        Sends a prompt to the Google Gemini model and returns the raw string response.
        Implements automatic retries with exponential backoff for transient and rate-limit errors.
        
        Args:
            prompt: The input prompt for the model.
            system_instruction: Optional system level guidelines for the model.
            
        Returns:
            The raw text completion returned by the model.
            
        Raises:
            AIProviderError: If the provider fails to generate a response or settings are missing.
        """
        client = self._get_client()
        
        config_args: dict = {
            "temperature": settings.GEMINI_TEMPERATURE,
            "response_mime_type": "application/json",
        }
        if system_instruction:
            config_args["system_instruction"] = system_instruction

        config = types.GenerateContentConfig(**config_args)

        max_retries = max(1, settings.MAX_RETRIES)
        last_exception: Exception | None = None

        for attempt in range(1, max_retries + 1):
            try:
                async with asyncio.timeout(settings.GEMINI_TIMEOUT):
                    response = await client.aio.models.generate_content(
                        model=self.model,
                        contents=prompt,
                        config=config,
                    )
                if response and response.text is not None:
                    return response.text
                raise AIProviderError("Gemini model returned empty response text.")

            except TimeoutError as e:
                last_exception = e
                logger.warning(f"Gemini API request timed out (attempt {attempt}/{max_retries}).")
            except APIError as e:
                last_exception = e
                status_code = getattr(e, "code", None) or getattr(e, "status_code", None)
                if status_code in (429, 500, 503, 504) or "RESOURCE_EXHAUSTED" in str(e):
                    logger.warning(f"Gemini API rate-limit/transient error ({e}) (attempt {attempt}/{max_retries}).")
                else:
                    raise AIProviderError(f"Gemini API Error: {str(e)}") from e
            except Exception as e:
                if isinstance(e, AIProviderError):
                    raise e
                last_exception = e
                logger.warning(f"Unexpected error calling Gemini API (attempt {attempt}/{max_retries}): {e}")

            if attempt < max_retries:
                backoff = 2 ** (attempt - 1)
                await asyncio.sleep(backoff)

        raise AIProviderError(f"GeminiProvider failed after {max_retries} attempts: {str(last_exception)}") from last_exception
