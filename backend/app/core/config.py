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
    GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY")
    DATABASE_URL: str | None = os.getenv("DATABASE_URL")
    PORT: int = int(os.getenv("PORT", "8000"))

settings = Settings()
