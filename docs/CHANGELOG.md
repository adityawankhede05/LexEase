# Changelog

All notable changes to the LexEase project will be documented in this file.

## [Sprint 5] - Whole Document Summarization

### Added
- Created `POST /documents/summarize` endpoint in dedicated router `app/api/routers/summary.py`.
- Created `DocumentSummaryService` in `app/services/summary.py` for concatenating clauses and orchestrating AI summarization.
- Defined `DocumentContext` base schema in `app/schemas/document.py` and `DocumentSummaryRequest` & `DocumentSummaryResponse` in `app/schemas/summary.py`.
- Added prompt template `app/prompts/document_summary.txt` instructing Google Gemini to return raw JSON matching `DocumentSummaryResponse`.
- Added `AITask.DOCUMENT_SUMMARY` to `AITask` enum in `app/schemas/ai.py` and updated `PromptBuilder` in `app/ai/prompt_builder.py`.
- Added domain-specific exceptions `EmptyClauseListError` and `SummaryGenerationError` in `app/services/exceptions.py`.
- Created comprehensive test suite in `tests/test_summary.py` including unit tests, API integration tests, and 50+ clause large document tests.
- Updated project documentation across `README.md`, `API.md`, `ARCHITECTURE.md`, `CHANGELOG.md`, `TESTING.md`, `FRONTEND_INTEGRATION.md`, `DEPLOYMENT.md`, and `DATABASE.md`.

### Changed
- Updated default Gemini model setting to `gemini-2.0-flash`.
- Mounted `summary.router` in `app/main.py`.

---

## [Sprint 4B] - Gemini Integration
- Integrated `GeminiProvider` using Google GenAI SDK.
- Implemented exponential backoff retries and error handling.

## [Sprint 4A] - AI Foundation Layer
- Created `AIService`, `BaseAIProvider`, `PromptBuilder`, and `ResponseParser`.

## [Sprint 3] - Text Preprocessing
- Implemented text cleaning, clause segmentation, and Indian PII masking.

## [Sprint 2] - PDF Upload & Extraction
- Created `POST /documents/upload` endpoint using PyMuPDF text extraction.

## [Sprint 1] - Backend Architecture
- Initialized FastAPI project structure, configuration, and health check endpoint.
