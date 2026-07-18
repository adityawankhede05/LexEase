import pytest
from pydantic import BaseModel
from app.schemas.ai import AITask, ClauseSimplificationResponse
from app.ai.base_provider import BaseAIProvider
from app.ai.gemini_provider import GeminiProvider
from app.ai.prompt_builder import PromptBuilder
from app.ai.response_parser import ResponseParser
from app.ai.exceptions import AIResponseValidationError
from app.services.ai import AIService

class MockAIProvider(BaseAIProvider):
    def __init__(self, response_text: str = ""):
        self.response_text = response_text
        self.last_prompt: str | None = None
        self.last_system_instruction: str | None = None

    async def generate(self, prompt: str, system_instruction: str | None = None) -> str:
        self.last_prompt = prompt
        self.last_system_instruction = system_instruction
        return self.response_text

@pytest.mark.anyio
async def test_gemini_provider_not_implemented():
    """Verify that the GeminiProvider skeleton raises NotImplementedError."""
    provider = GeminiProvider()
    with pytest.raises(NotImplementedError) as exc_info:
        await provider.generate("test prompt")
    assert "GeminiProvider is not yet implemented" in str(exc_info.value)

def test_prompt_builder_simplification():
    """Verify that PromptBuilder builds correct prompt for SIMPLIFICATION."""
    payload = {"clause_text": "The Client shall compensate the Provider."}
    prompt = PromptBuilder.build_prompt(AITask.SIMPLIFICATION, payload)
    assert "Simplify this legal clause:" in prompt
    assert "The Client shall compensate the Provider." in prompt

def test_prompt_builder_missing_keys():
    """Verify that PromptBuilder raises ValueError if required payload keys are missing."""
    with pytest.raises(ValueError) as exc_info:
        PromptBuilder.build_prompt(AITask.SIMPLIFICATION, {})
    assert "Missing required key 'clause_text'" in str(exc_info.value)

def test_prompt_builder_invalid_task():
    """Verify that PromptBuilder raises ValueError for unsupported or invalid task types."""
    with pytest.raises(ValueError) as exc_info:
        # Pass raw string instead of AITask enum
        PromptBuilder.build_prompt("unsupported_task", {})
    assert "task_type must be an instance of AITask" in str(exc_info.value)

def test_response_parser_success():
    """Verify that ResponseParser successfully validates raw JSON response."""
    raw_json = '{"simplified_text": "The customer must pay the supplier."}'
    parsed = ResponseParser.parse_json_response(raw_json, ClauseSimplificationResponse)
    assert isinstance(parsed, ClauseSimplificationResponse)
    assert parsed.simplified_text == "The customer must pay the supplier."

def test_response_parser_markdown_backticks():
    """Verify that ResponseParser handles markdown JSON wrapper backticks correctly."""
    raw_json = '```json\n{"simplified_text": "Simplified text here."}\n```'
    parsed = ResponseParser.parse_json_response(raw_json, ClauseSimplificationResponse)
    assert parsed.simplified_text == "Simplified text here."

def test_response_parser_invalid_json():
    """Verify that ResponseParser raises AIResponseValidationError on invalid JSON."""
    bad_json = "not-a-json-string"
    with pytest.raises(AIResponseValidationError) as exc_info:
        ResponseParser.parse_json_response(bad_json, ClauseSimplificationResponse)
    assert "Response text is not valid JSON" in str(exc_info.value)

def test_response_parser_failed_validation():
    """Verify that ResponseParser raises AIResponseValidationError on schema validation errors."""
    incorrect_json = '{"unrelated_field": "some data"}'
    with pytest.raises(AIResponseValidationError) as exc_info:
        ResponseParser.parse_json_response(incorrect_json, ClauseSimplificationResponse)
    assert "Response data failed validation" in str(exc_info.value)

@pytest.mark.anyio
async def test_ai_service_orchestration():
    """Verify AIService orchestrates PromptBuilder, provider generation, and ResponseParser validation."""
    mock_json_response = '{"simplified_text": "Generic simplified clause."}'
    mock_provider = MockAIProvider(response_text=mock_json_response)
    ai_service = AIService(provider=mock_provider)
    
    payload = {"clause_text": "Party of the first part agrees to..."}
    
    result = await ai_service.generate(
        task_type=AITask.SIMPLIFICATION,
        payload=payload,
        response_schema=ClauseSimplificationResponse
    )
    
    assert isinstance(result, ClauseSimplificationResponse)
    assert result.simplified_text == "Generic simplified clause."
    
    # Assert PromptBuilder was called correctly
    assert mock_provider.last_prompt is not None
    assert "Party of the first part agrees to..." in mock_provider.last_prompt
