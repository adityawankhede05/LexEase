from app.ai.base_provider import BaseAIProvider

class GeminiProvider(BaseAIProvider):
    async def generate(self, prompt: str, system_instruction: str | None = None) -> str:
        """
        Skeleton implementation for Google Gemini model generation.
        Raises NotImplementedError directly for Sprint 4A.
        """
        raise NotImplementedError("GeminiProvider is not yet implemented. Will be resolved in Sprint 4B.")
