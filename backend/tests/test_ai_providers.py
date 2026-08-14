import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import httpx
import pytest

from app.ai.exceptions import AIProviderError
from app.ai.gemini_provider import GeminiProvider
from app.ai.groq_provider import GroqProvider
from app.ai.openrouter_provider import OpenRouterProvider
from app.ai.provider_factory import get_ai_provider


# ===========================================================================
# 1. OPENROUTER PROVIDER TESTS
# ===========================================================================

@pytest.mark.anyio
async def test_openrouter_missing_api_key():
    """Verify OpenRouterProvider raises AIProviderError when API key is missing."""
    with patch("app.ai.openrouter_provider.settings.OPENROUTER_API_KEY", None):
        provider = OpenRouterProvider(api_key=None)
        with pytest.raises(AIProviderError) as exc_info:
            await provider.generate("Test prompt")
        assert "OPENROUTER_API_KEY is not configured" in str(exc_info.value)


@pytest.mark.anyio
async def test_openrouter_successful_response():
    """Verify OpenRouterProvider makes correct HTTP request and returns parsed content string."""
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": '{"summary": "Test plain English summary."}'
                }
            }
        ]
    }

    mock_client = AsyncMock()
    mock_client.post.return_value = mock_response

    provider = OpenRouterProvider(api_key="test-openrouter-key", model="test/model")

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client_cls.return_value.__aenter__.return_value = mock_client
        result = await provider.generate("Simplify this clause.")

        assert result == '{"summary": "Test plain English summary."}'
        mock_client.post.assert_called_once()
        call_kwargs = mock_client.post.call_args.kwargs
        assert call_kwargs["json"]["model"] == "test/model"
        assert call_kwargs["json"]["messages"] == [{"role": "user", "content": "Simplify this clause."}]
        assert call_kwargs["headers"]["Authorization"] == "Bearer test-openrouter-key"


@pytest.mark.anyio
async def test_openrouter_system_instruction_included():
    """Verify OpenRouterProvider includes system role message when system_instruction is provided."""
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "Response with system instruction"}}]
    }

    mock_client = AsyncMock()
    mock_client.post.return_value = mock_response

    provider = OpenRouterProvider(api_key="test-openrouter-key")

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client_cls.return_value.__aenter__.return_value = mock_client
        result = await provider.generate(
            prompt="User prompt",
            system_instruction="System guidelines"
        )

        assert result == "Response with system instruction"
        call_kwargs = mock_client.post.call_args.kwargs
        messages = call_kwargs["json"]["messages"]
        assert len(messages) == 2
        assert messages[0] == {"role": "system", "content": "System guidelines"}
        assert messages[1] == {"role": "user", "content": "User prompt"}


@pytest.mark.anyio
async def test_openrouter_401_403_handling():
    """Verify OpenRouterProvider raises AIProviderError immediately without retrying on 401/403."""
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 401
    mock_response.text = "Unauthorized - Invalid API Key"

    mock_client = AsyncMock()
    mock_client.post.return_value = mock_response

    provider = OpenRouterProvider(api_key="invalid-key")

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client_cls.return_value.__aenter__.return_value = mock_client
        with pytest.raises(AIProviderError) as exc_info:
            await provider.generate("Test prompt")
        assert "401" in str(exc_info.value)
        assert mock_client.post.call_count == 1  # No retries


@pytest.mark.anyio
async def test_openrouter_429_retry():
    """Verify OpenRouterProvider retries on 429 rate limit and succeeds if subsequent attempt succeeds."""
    mock_rate_limit = MagicMock(spec=httpx.Response)
    mock_rate_limit.status_code = 429
    mock_rate_limit.text = "Rate limit exceeded"

    mock_success = MagicMock(spec=httpx.Response)
    mock_success.status_code = 200
    mock_success.json.return_value = {
        "choices": [{"message": {"content": "Success after retry"}}]
    }

    mock_client = AsyncMock()
    mock_client.post.side_effect = [mock_rate_limit, mock_success]

    provider = OpenRouterProvider(api_key="test-key")

    with patch("httpx.AsyncClient") as mock_client_cls, \
         patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        mock_client_cls.return_value.__aenter__.return_value = mock_client
        result = await provider.generate("Test prompt")

        assert result == "Success after retry"
        assert mock_client.post.call_count == 2
        mock_sleep.assert_called_once_with(1)  # 2^(1-1) = 1


@pytest.mark.anyio
async def test_openrouter_503_retry():
    """Verify OpenRouterProvider retries on 503 service unavailable and succeeds."""
    mock_503 = MagicMock(spec=httpx.Response)
    mock_503.status_code = 503
    mock_503.text = "Service Unavailable"

    mock_success = MagicMock(spec=httpx.Response)
    mock_success.status_code = 200
    mock_success.json.return_value = {
        "choices": [{"message": {"content": "Recovered from 503"}}]
    }

    mock_client = AsyncMock()
    mock_client.post.side_effect = [mock_503, mock_success]

    provider = OpenRouterProvider(api_key="test-key")

    with patch("httpx.AsyncClient") as mock_client_cls, \
         patch("asyncio.sleep", new_callable=AsyncMock):
        mock_client_cls.return_value.__aenter__.return_value = mock_client
        result = await provider.generate("Test prompt")

        assert result == "Recovered from 503"
        assert mock_client.post.call_count == 2


@pytest.mark.anyio
async def test_openrouter_retry_exhaustion():
    """Verify OpenRouterProvider raises AIProviderError after exhausting MAX_RETRIES on 500 errors."""
    mock_500 = MagicMock(spec=httpx.Response)
    mock_500.status_code = 500
    mock_500.text = "Internal Server Error"

    mock_client = AsyncMock()
    mock_client.post.return_value = mock_500

    provider = OpenRouterProvider(api_key="test-key")

    with patch("httpx.AsyncClient") as mock_client_cls, \
         patch("app.ai.openrouter_provider.settings.MAX_RETRIES", 3), \
         patch("asyncio.sleep", new_callable=AsyncMock):
        mock_client_cls.return_value.__aenter__.return_value = mock_client
        with pytest.raises(AIProviderError) as exc_info:
            await provider.generate("Test prompt")

        assert "failed after 3 attempts" in str(exc_info.value)
        assert mock_client.post.call_count == 3


@pytest.mark.anyio
async def test_openrouter_empty_malformed_response():
    """Verify OpenRouterProvider raises AIProviderError on empty/malformed choices JSON."""
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {"choices": []}

    mock_client = AsyncMock()
    mock_client.post.return_value = mock_response

    provider = OpenRouterProvider(api_key="test-key")

    with patch("httpx.AsyncClient") as mock_client_cls, \
         patch("app.ai.openrouter_provider.settings.MAX_RETRIES", 1):
        mock_client_cls.return_value.__aenter__.return_value = mock_client
        with pytest.raises(AIProviderError) as exc_info:
            await provider.generate("Test prompt")

        assert "empty or malformed completion content" in str(exc_info.value)


@pytest.mark.anyio
async def test_openrouter_timeout_network_error():
    """Verify OpenRouterProvider retries on timeout/network error up to MAX_RETRIES."""
    mock_client = AsyncMock()
    mock_client.post.side_effect = httpx.ConnectTimeout("Connection timed out")

    provider = OpenRouterProvider(api_key="test-key")

    with patch("httpx.AsyncClient") as mock_client_cls, \
         patch("app.ai.openrouter_provider.settings.MAX_RETRIES", 2), \
         patch("asyncio.sleep", new_callable=AsyncMock):
        mock_client_cls.return_value.__aenter__.return_value = mock_client
        with pytest.raises(AIProviderError) as exc_info:
            await provider.generate("Test prompt")

        assert "failed after 2 attempts" in str(exc_info.value)
        assert mock_client.post.call_count == 2


# ===========================================================================
# 2. GROQ PROVIDER TESTS
# ===========================================================================

@pytest.mark.anyio
async def test_groq_missing_api_key():
    """Verify GroqProvider raises AIProviderError when API key is missing."""
    with patch("app.ai.groq_provider.settings.GROQ_API_KEY", None):
        provider = GroqProvider(api_key=None)
        with pytest.raises(AIProviderError) as exc_info:
            await provider.generate("Test prompt")
        assert "GROQ_API_KEY is not configured" in str(exc_info.value)


@pytest.mark.anyio
async def test_groq_successful_response():
    """Verify GroqProvider makes correct HTTP request and returns completion content string."""
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": '{"simplified_text": "Plain English version."}'
                }
            }
        ]
    }

    mock_client = AsyncMock()
    mock_client.post.return_value = mock_response

    provider = GroqProvider(api_key="test-groq-key", model="openai/gpt-oss-120b")

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client_cls.return_value.__aenter__.return_value = mock_client
        result = await provider.generate("Simplify this legal text.")

        assert result == '{"simplified_text": "Plain English version."}'
        mock_client.post.assert_called_once()
        call_kwargs = mock_client.post.call_args.kwargs
        assert call_kwargs["json"]["model"] == "openai/gpt-oss-120b"
        assert call_kwargs["json"]["messages"] == [{"role": "user", "content": "Simplify this legal text."}]
        assert call_kwargs["headers"]["Authorization"] == "Bearer test-groq-key"


@pytest.mark.anyio
async def test_groq_system_instruction_included():
    """Verify GroqProvider includes system role message when system_instruction is provided."""
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "Groq system instruction response"}}]
    }

    mock_client = AsyncMock()
    mock_client.post.return_value = mock_response

    provider = GroqProvider(api_key="test-groq-key")

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client_cls.return_value.__aenter__.return_value = mock_client
        result = await provider.generate(
            prompt="User prompt",
            system_instruction="You are a legal assistant."
        )

        assert result == "Groq system instruction response"
        call_kwargs = mock_client.post.call_args.kwargs
        messages = call_kwargs["json"]["messages"]
        assert len(messages) == 2
        assert messages[0] == {"role": "system", "content": "You are a legal assistant."}
        assert messages[1] == {"role": "user", "content": "User prompt"}


@pytest.mark.anyio
async def test_groq_401_403_handling():
    """Verify GroqProvider raises AIProviderError immediately without retrying on 401/403."""
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 401
    mock_response.text = "Invalid API Key"

    mock_client = AsyncMock()
    mock_client.post.return_value = mock_response

    provider = GroqProvider(api_key="invalid-groq-key")

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client_cls.return_value.__aenter__.return_value = mock_client
        with pytest.raises(AIProviderError) as exc_info:
            await provider.generate("Test prompt")
        assert "401" in str(exc_info.value)
        assert mock_client.post.call_count == 1  # No retries


@pytest.mark.anyio
async def test_groq_429_retry():
    """Verify GroqProvider retries on 429 rate limit and succeeds on retry."""
    mock_rate_limit = MagicMock(spec=httpx.Response)
    mock_rate_limit.status_code = 429
    mock_rate_limit.text = "Rate limit reached"

    mock_success = MagicMock(spec=httpx.Response)
    mock_success.status_code = 200
    mock_success.json.return_value = {
        "choices": [{"message": {"content": "Groq success after retry"}}]
    }

    mock_client = AsyncMock()
    mock_client.post.side_effect = [mock_rate_limit, mock_success]

    provider = GroqProvider(api_key="test-key")

    with patch("httpx.AsyncClient") as mock_client_cls, \
         patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        mock_client_cls.return_value.__aenter__.return_value = mock_client
        result = await provider.generate("Test prompt")

        assert result == "Groq success after retry"
        assert mock_client.post.call_count == 2
        mock_sleep.assert_called_once_with(1)


@pytest.mark.anyio
async def test_groq_503_retry():
    """Verify GroqProvider retries on 503 service unavailable and succeeds."""
    mock_503 = MagicMock(spec=httpx.Response)
    mock_503.status_code = 503
    mock_503.text = "Service Unavailable"

    mock_success = MagicMock(spec=httpx.Response)
    mock_success.status_code = 200
    mock_success.json.return_value = {
        "choices": [{"message": {"content": "Groq recovered from 503"}}]
    }

    mock_client = AsyncMock()
    mock_client.post.side_effect = [mock_503, mock_success]

    provider = GroqProvider(api_key="test-key")

    with patch("httpx.AsyncClient") as mock_client_cls, \
         patch("asyncio.sleep", new_callable=AsyncMock):
        mock_client_cls.return_value.__aenter__.return_value = mock_client
        result = await provider.generate("Test prompt")

        assert result == "Groq recovered from 503"
        assert mock_client.post.call_count == 2


@pytest.mark.anyio
async def test_groq_retry_exhaustion():
    """Verify GroqProvider raises AIProviderError after exhausting MAX_RETRIES on 500 errors."""
    mock_500 = MagicMock(spec=httpx.Response)
    mock_500.status_code = 500
    mock_500.text = "Internal Groq Error"

    mock_client = AsyncMock()
    mock_client.post.return_value = mock_500

    provider = GroqProvider(api_key="test-key")

    with patch("httpx.AsyncClient") as mock_client_cls, \
         patch("app.ai.groq_provider.settings.MAX_RETRIES", 3), \
         patch("asyncio.sleep", new_callable=AsyncMock):
        mock_client_cls.return_value.__aenter__.return_value = mock_client
        with pytest.raises(AIProviderError) as exc_info:
            await provider.generate("Test prompt")

        assert "failed after 3 attempts" in str(exc_info.value)
        assert mock_client.post.call_count == 3


@pytest.mark.anyio
async def test_groq_empty_malformed_response():
    """Verify GroqProvider raises AIProviderError on empty/malformed choices JSON."""
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {"choices": []}

    mock_client = AsyncMock()
    mock_client.post.return_value = mock_response

    provider = GroqProvider(api_key="test-key")

    with patch("httpx.AsyncClient") as mock_client_cls, \
         patch("app.ai.groq_provider.settings.MAX_RETRIES", 1):
        mock_client_cls.return_value.__aenter__.return_value = mock_client
        with pytest.raises(AIProviderError) as exc_info:
            await provider.generate("Test prompt")

        assert "empty or malformed completion content" in str(exc_info.value)


@pytest.mark.anyio
async def test_groq_timeout_network_error():
    """Verify GroqProvider retries on timeout/network error up to MAX_RETRIES."""
    mock_client = AsyncMock()
    mock_client.post.side_effect = httpx.ReadTimeout("Read timed out")

    provider = GroqProvider(api_key="test-key")

    with patch("httpx.AsyncClient") as mock_client_cls, \
         patch("app.ai.groq_provider.settings.MAX_RETRIES", 2), \
         patch("asyncio.sleep", new_callable=AsyncMock):
        mock_client_cls.return_value.__aenter__.return_value = mock_client
        with pytest.raises(AIProviderError) as exc_info:
            await provider.generate("Test prompt")

        assert "failed after 2 attempts" in str(exc_info.value)
        assert mock_client.post.call_count == 2


# ===========================================================================
# 3. PROVIDER SELECTION / FACTORY TESTS
# ===========================================================================

def test_provider_factory_selects_gemini():
    """Verify get_ai_provider() instantiates GeminiProvider when AI_PROVIDER=gemini."""
    with patch("app.ai.provider_factory.settings.AI_PROVIDER", "gemini"):
        provider = get_ai_provider()
        assert isinstance(provider, GeminiProvider)


def test_provider_factory_selects_groq():
    """Verify get_ai_provider() instantiates GroqProvider when AI_PROVIDER=groq."""
    with patch("app.ai.provider_factory.settings.AI_PROVIDER", "groq"):
        provider = get_ai_provider()
        assert isinstance(provider, GroqProvider)


def test_provider_factory_selects_openrouter():
    """Verify get_ai_provider() instantiates OpenRouterProvider when AI_PROVIDER=openrouter."""
    with patch("app.ai.provider_factory.settings.AI_PROVIDER", "openrouter"):
        provider = get_ai_provider()
        assert isinstance(provider, OpenRouterProvider)


def test_provider_factory_override_argument():
    """Verify get_ai_provider('groq') overrides settings.AI_PROVIDER."""
    with patch("app.ai.provider_factory.settings.AI_PROVIDER", "gemini"):
        provider = get_ai_provider("groq")
        assert isinstance(provider, GroqProvider)


def test_provider_factory_invalid_provider_raises_error():
    """Verify get_ai_provider() raises AIProviderError for unsupported provider string."""
    with patch("app.ai.provider_factory.settings.AI_PROVIDER", "unsupported_provider"):
        with pytest.raises(AIProviderError) as exc_info:
            get_ai_provider()
        assert "Unsupported AI provider: 'unsupported_provider'" in str(exc_info.value)
        assert "'gemini', 'groq', 'openrouter'" in str(exc_info.value)
