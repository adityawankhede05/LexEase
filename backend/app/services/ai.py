from typing import Type, TypeVar
from pydantic import BaseModel
from app.schemas.ai import AITask
from app.ai.base_provider import BaseAIProvider
from app.ai.prompt_builder import PromptBuilder
from app.ai.response_parser import ResponseParser

T = TypeVar("T", bound=BaseModel)

class AIService:
    """
    AIService orchestrates prompt building, provider invocation, and response parsing
    for generic AI-driven tasks, remaining independent of specific provider logic.
    """
    def __init__(self, provider: BaseAIProvider):
        self.provider = provider

    async def generate(self, task_type: AITask, payload: dict, response_schema: Type[T]) -> T:
        """
        Orchestrates prompt formatting, provider invocation, and JSON-to-Pydantic parsing.
        
        Args:
            task_type: The enum value for the requested task.
            payload: A dictionary of key-value properties required for the task.
            response_schema: The Pydantic BaseModel subclass to validate the output against.
            
        Returns:
            An instance of the response_schema containing validated model outputs.
        """
        # 1. Construct prompt driven by task type
        prompt = PromptBuilder.build_prompt(task_type, payload)
        
        # 2. Call the injected abstract AI provider
        raw_response = await self.provider.generate(prompt)
        
        # 3. Parse and validate the JSON output
        parsed_response = ResponseParser.parse_json_response(raw_response, response_schema)
        
        return parsed_response
