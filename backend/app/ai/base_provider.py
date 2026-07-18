from abc import ABC, abstractmethod

class BaseAIProvider(ABC):
    @abstractmethod
    async def generate(self, prompt: str, system_instruction: str | None = None) -> str:
        """
        Sends a prompt to the AI provider and returns the raw string response.
        
        Args:
            prompt: The formatted input prompt for the model.
            system_instruction: Optional system level guidelines for the model.
            
        Returns:
            The raw text completion returned by the model.
            
        Raises:
            AIProviderError: If the provider fails to generate a response.
        """
        pass
