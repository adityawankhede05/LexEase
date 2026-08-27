import asyncio
import logging
import re
from typing import Any
import httpx

from app.ai.base_provider import BaseAIProvider
from app.ai.exceptions import AIProviderError
from app.core.config import settings

logger = logging.getLogger(__name__)

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


def _parse_duration_string(duration_str: str) -> float | None:
    """Parses duration strings like '17.45s', '500ms', '1m', '17' into seconds."""
    s = duration_str.strip().lower()
    if s.endswith("ms"):
        try:
            return float(s[:-2]) / 1000.0
        except ValueError:
            return None
    elif s.endswith("s"):
        try:
            return float(s[:-1])
        except ValueError:
            return None
    elif s.endswith("m") and not s.endswith("ms"):
        try:
            return float(s[:-1]) * 60.0
        except ValueError:
            return None
    else:
        try:
            return float(s)
        except ValueError:
            return None


def parse_retry_after(response: httpx.Response) -> float | None:
    """
    Extracts retry delay in seconds from HTTP headers or Groq rate limit error response body.
    Supports:
    - 'Retry-After' header (seconds)
    - 'x-ratelimit-reset-tokens' / 'x-ratelimit-reset-requests' header
    - Error message regex: 'Please try again in Xs'
    """
    # 1. Check 'Retry-After' header
    retry_after_hdr = response.headers.get("retry-after")
    if retry_after_hdr:
        try:
            val = float(retry_after_hdr)
            if val >= 0:
                return val
        except ValueError:
            pass

    # 2. Check ratelimit headers
    for hdr_name in ("x-ratelimit-reset-tokens", "x-ratelimit-reset-requests"):
        val_str = response.headers.get(hdr_name)
        if val_str:
            parsed = _parse_duration_string(val_str)
            if parsed is not None and parsed >= 0:
                return parsed

    # 3. Check error message in response body
    try:
        data = response.json()
        error_msg = data.get("error", {}).get("message", "")
    except Exception:
        error_msg = response.text or ""

    if error_msg:
        # Match e.g. "Please try again in 17.45s", "try again in 17s", "try again in 500ms"
        match = re.search(
            r"try again in (\d+(?:\.\d+)?)\s*(ms|s|m|seconds?|minutes?)?",
            error_msg,
            re.IGNORECASE,
        )
        if match:
            try:
                num = float(match.group(1))
                unit = (match.group(2) or "s").lower()
                if unit.startswith("ms"):
                    return num / 1000.0
                elif unit.startswith("m") and not unit.startswith("ms"):
                    return num * 60.0
                else:
                    return num
            except ValueError:
                pass

    return None


class GroqProvider(BaseAIProvider):
    """
    Groq AI provider implementation conforming to BaseAIProvider.
    Uses async httpx client to communicate with Groq OpenAI-compatible chat completions endpoint.
    Implements retry with exponential backoff and provider-specified delay parsing for rate limits (429).
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
    ):
        self.api_key = api_key or settings.GROQ_API_KEY
        self.model = model or settings.GROQ_MODEL
        self.timeout = timeout or settings.GROQ_TIMEOUT

    async def generate(self, prompt: str, system_instruction: str | None = None) -> str:
        """
        Sends a prompt to the Groq chat completions API and returns raw string response.
        
        Args:
            prompt: The formatted user input prompt for the model.
            system_instruction: Optional system level guidelines for the model.
            
        Returns:
            The raw text completion string returned by the model.
            
        Raises:
            AIProviderError: If generation fails, authentication fails, or retries are exhausted.
        """
        if not self.api_key:
            raise AIProviderError("GROQ_API_KEY is not configured in settings.")

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
            retry_delay: float = 2 ** (attempt - 1)
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        GROQ_API_URL,
                        json=payload,
                        headers=headers,
                    )

                # Check HTTP status codes
                if response.status_code == 200:
                    data = response.json()
                    choices = data.get("choices")
                    if choices and isinstance(choices, list) and len(choices) > 0:
                        content = choices[0].get("message", {}).get("content")
                        if content is not None:
                            return str(content)
                    raise AIProviderError("Groq returned empty or malformed completion content.")

                # Non-retryable client errors: 400, 401, 403
                if response.status_code in (400, 401, 403):
                    raise AIProviderError(
                        f"Groq API request failed with status {response.status_code}: {response.text}"
                    )

                # Retryable status codes: 429, 500, 502, 503, 504
                if response.status_code in (429, 500, 502, 503, 504):
                    last_exception = AIProviderError(
                        f"Groq transient error ({response.status_code}): {response.text}"
                    )
                    logger.warning(
                        f"Groq API transient error ({response.status_code}) on attempt {attempt}/{max_retries}."
                    )
                    if response.status_code == 429:
                        parsed_delay = parse_retry_after(response)
                        if parsed_delay is not None and parsed_delay > 0:
                            retry_delay = parsed_delay
                            logger.info(
                                f"Parsed Groq 429 retry delay: {retry_delay:.2f}s."
                            )
                else:
                    raise AIProviderError(
                        f"Groq API error with status {response.status_code}: {response.text}"
                    )

            except httpx.TimeoutException as e:
                last_exception = e
                logger.warning(f"Groq API request timed out (attempt {attempt}/{max_retries}).")
            except httpx.NetworkError as e:
                last_exception = e
                logger.warning(f"Groq API network error (attempt {attempt}/{max_retries}): {e}")
            except AIProviderError:
                raise
            except Exception as e:
                last_exception = e
                logger.warning(f"Unexpected error calling Groq API (attempt {attempt}/{max_retries}): {e}")

            if attempt < max_retries:
                logger.info(f"Retrying Groq API call in {retry_delay:.2f}s (attempt {attempt}/{max_retries})...")
                await asyncio.sleep(retry_delay)

        raise AIProviderError(
            f"GroqProvider failed after {max_retries} attempts: {str(last_exception)}"
        ) from last_exception
