import json
from typing import Type, TypeVar
from pydantic import BaseModel, ValidationError
from app.ai.exceptions import AIResponseValidationError

T = TypeVar("T", bound=BaseModel)

class ResponseParser:
    @classmethod
    def parse_json_response(cls, response_text: str, schema: Type[T]) -> T:
        """
        Parses JSON response text and validates it against the provided Pydantic model.
        
        Args:
            response_text: The raw string response from the AI model (expected to be JSON).
            schema: The Pydantic model subclass to parse/validate against.
            
        Returns:
            An instance of the schema populated with the validated data.
            
        Raises:
            AIResponseValidationError: If the response is not valid JSON or fails schema validation.
        """
        if not response_text:
            raise AIResponseValidationError("Received empty response from AI model.")

        # Strip potential markdown formatting if model output wraps JSON in ```json blocks
        cleaned_text = response_text.strip()
        if cleaned_text.startswith("```json"):
            cleaned_text = cleaned_text[7:]
        elif cleaned_text.startswith("```"):
            cleaned_text = cleaned_text[3:]
        if cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[:-3]
        cleaned_text = cleaned_text.strip()

        try:
            parsed_data = json.loads(cleaned_text)
        except json.JSONDecodeError as e:
            raise AIResponseValidationError(f"Response text is not valid JSON: {str(e)}") from e

        try:
            # Check if it is a subclass of BaseModel
            if not issubclass(schema, BaseModel):
                raise ValueError("Target validation schema must be a subclass of pydantic.BaseModel")
            return schema.model_validate(parsed_data)
        except ValidationError as e:
            raise AIResponseValidationError(f"Response data failed validation against schema {schema.__name__}: {str(e)}") from e
