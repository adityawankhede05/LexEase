class AIError(Exception):
    """Base exception for all AI foundation layer errors."""
    pass

class AIProviderError(AIError):
    """Raised when the AI provider encounters issues or is unimplemented."""
    pass

class AIResponseValidationError(AIError):
    """Raised when the response validation or parsing fails."""
    pass
