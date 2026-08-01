import asyncio
import sys
from pathlib import Path

# Ensure backend directory is in sys.path for direct execution
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.core.config import settings
from app.ai.gemini_provider import GeminiProvider
from app.services.ai import AIService
from app.schemas.ai import AITask, ClauseSimplificationResponse

async def main():
    print("==========================================")
    print(" LexEase - Gemini Integration Verification ")
    print("==========================================")
    print(f"Project Name : {settings.PROJECT_NAME}")
    print(f"Gemini Model : {settings.GEMINI_MODEL}")
    
    if not settings.GEMINI_API_KEY:
        print("\n[WARNING] GEMINI_API_KEY is not set in environment or .env file.")
        print("Set GEMINI_API_KEY in backend/.env to execute live API generation.")
        return

    provider = GeminiProvider()
    ai_service = AIService(provider=provider)

    sample_clause = (
        "The Licensee shall indemnify, defend, and hold harmless the Licensor, "
        "its officers, directors, employees, and agents from and against any and all "
        "claims, liabilities, damages, losses, or expenses arising out of or in connection with "
        "Licensee's use of the Software."
    )

    print(f"\nOriginal Clause:\n{sample_clause}\n")
    print("Sending clause to Gemini API...")

    payload = {"clause_text": sample_clause}

    try:
        response: ClauseSimplificationResponse = await ai_service.generate(
            task_type=AITask.SIMPLIFICATION,
            payload=payload,
            response_schema=ClauseSimplificationResponse,
        )
        print("\n==========================================")
        print(" ClauseSimplificationResponse Result ")
        print("==========================================")
        print(f"Simplified Text:\n{response.simplified_text}\n")
        print("Model JSON Dump:")
        print(response.model_dump_json(indent=2))
    except Exception as e:
        import traceback
        print(f"\n[ERROR] Gemini API call failed: {e}\n")
        print("--- Exception Repr ---")
        print(repr(e))
        print("\n--- Exception Cause / Context ---")
        if e.__cause__:
            print(f"Cause: {repr(e.__cause__)}")
        if e.__context__:
            print(f"Context: {repr(e.__context__)}")
        print("\n--- Full Traceback ---")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
