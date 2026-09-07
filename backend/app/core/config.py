import os
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

class Settings(BaseModel):
    PROJECT_NAME: str = "LexEase API"
    DESCRIPTION: str = "Backend API for Legal Document Simplification and Clause Risk Analysis"
    VERSION: str = "0.1.0"
    API_PREFIX: str = "/api"
    
    ALLOWED_ORIGINS: list[str] = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",")
    
    # Active AI Provider: "gemini", "groq", "openrouter", "cerebras"
    AI_PROVIDER: str = os.getenv("AI_PROVIDER", "gemini").lower()
    
    # Gemini AI Provider Configuration
    GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    GEMINI_TEMPERATURE: float = float(os.getenv("GEMINI_TEMPERATURE", "0.2"))
    GEMINI_TIMEOUT: float = float(os.getenv("GEMINI_TIMEOUT", "30.0"))
    
    # Groq AI Provider Configuration
    GROQ_API_KEY: str | None = os.getenv("GROQ_API_KEY")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    GROQ_TIMEOUT: float = float(os.getenv("GROQ_TIMEOUT", "30.0"))
    GROQ_REQUEST_DELAY: float = float(os.getenv("GROQ_REQUEST_DELAY", "2.0"))
    
    # OpenRouter AI Provider Configuration
    OPENROUTER_API_KEY: str | None = os.getenv("OPENROUTER_API_KEY")
    OPENROUTER_MODEL: str = os.getenv("OPENROUTER_MODEL", "openrouter/free")
    OPENROUTER_TIMEOUT: float = float(os.getenv("OPENROUTER_TIMEOUT", "30.0"))
    
    # General AI Configuration
    MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "3"))
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./lexease.db")
    PORT: int = int(os.getenv("PORT", "8000"))

    # Cerebras AI Provider Configuration
    CEREBRAS_API_KEY: str | None = os.getenv("CEREBRAS_API_KEY")
    CEREBRAS_MODEL: str = os.getenv("CEREBRAS_MODEL", "llama-3.3-70b")
    CEREBRAS_TIMEOUT: float = float(os.getenv("CEREBRAS_TIMEOUT", "30.0"))

settings = Settings()
