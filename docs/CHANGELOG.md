# Changelog

All notable changes to the LexEase project will be documented in this file.

## [Sprint 8] - Multi-Provider AI Architecture (Groq + OpenRouter + Provider Factory)

### Added
- Created `GroqProvider` in `app/ai/groq_provider.py` conforming to `BaseAIProvider` using async `httpx` to call Groq's OpenAI-compatible completions endpoint (`https://api.groq.com/openai/v1/chat/completions`) with exponential backoff retries.
- Created `OpenRouterProvider` in `app/ai/openrouter_provider.py` conforming to `BaseAIProvider` using async `httpx` to call OpenRouter's completions endpoint (`https://openrouter.ai/api/v1/chat/completions`) with exponential backoff retries.
- Created centralized `ProviderFactory` (`get_ai_provider()`) in `app/ai/provider_factory.py` for dynamic provider resolution based on `settings.AI_PROVIDER` (`groq`, `openrouter`, `gemini`), with domain error handling for unsupported providers.
- Added configuration settings in `app/core/config.py`: `AI_PROVIDER`, `GROQ_API_KEY`, `GROQ_MODEL`, `GROQ_TIMEOUT`, `OPENROUTER_API_KEY`, `OPENROUTER_MODEL`, `OPENROUTER_TIMEOUT` while retaining full backward compatibility for Gemini.
- Created comprehensive 23-test suite in `tests/test_ai_providers.py` covering OpenRouter (missing key, success, system instruction, 401/403, 429/503 retry, retry exhaustion, empty/malformed, timeout/network errors), Groq (missing key, success, system instruction, 401/403, 429/503 retry, retry exhaustion, empty/malformed, timeout/network errors), and Provider Factory (`gemini`, `groq`, `openrouter`, override argument, unsupported provider error).
- Updated `DocumentSummaryService`, `ClauseAnalysisService`, and `DocumentQAService` to resolve default provider via `get_ai_provider()` instead of hardcoded `GeminiProvider()`.
- Updated `pyproject.toml` moving `httpx` to core dependencies.
- Updated `.env.example` with multi-provider environment variables.
- Enhanced `ClauseRetrievalService` in `app/services/retrieval.py` with suffix stemming normalization and query framing stop words (`due`, `amount`, `much`, `many`, `tell`, `explain`, `state`, `mention`, `give`, `find`), fixing zero-retrieval false negatives on compound inquiries (e.g. *"What is the monthly rent and when is it due?"*) while strictly preserving the `min_score = 0.6` unrelated question protection.
- Added dedicated regression tests in `tests/test_qa.py` covering monthly rent and due date retrieval across morphological variations and end-to-end service execution.
- Updated project documentation across `README.md`, `API.md`, `ARCHITECTURE.md`, `CHANGELOG.md`, `TESTING.md`, `FRONTEND_INTEGRATION.md`, `DEPLOYMENT.md`, and `DATABASE.md`.

---

## [Sprint 7] - Grounded Document Q&A

### Added
- Created `POST /documents/ask` endpoint in dedicated router `app/api/routers/qa.py`.
- Created `DocumentQAService` in `app/services/qa.py` for orchestrating context retrieval, relevance scoring, early exit for unanswerable context, and AI generation.
- Created `ClauseRetrievalService` in `app/services/retrieval.py` for lightweight lexical relevance retrieval (term overlap scoring, stop-word removal, clause ranking, top-N filtering, and minimum relevance threshold `min_score = 0.6`) without vector databases or embeddings.
- Fixed false positive clause retrieval on unrelated questions (e.g. *"What is the capital of France?"*) by enforcing minimum relevance score threshold `min_score = 0.6`, ensuring unrelated questions terminate early without calling `GeminiProvider` or `AIService`.
- Created modular in-memory `DocumentContextStore` in `app/database/document_context_store.py` for server-side storage of preprocessed, PII-masked `ClauseSegment` lists, indexed by `document_id`.
- Defined `DocumentQARequest` (with validated `document_id` and `question` max 2000 chars) and `DocumentQAResponse` (with `answer`, `source_clauses`, `confidence` 0.0–1.0, `cannot_answer`) in `app/schemas/qa.py`.
- Extended `DocumentUploadResponse` schema in `app/schemas/document.py` to include optional `document_id: str | None = None`.
- Updated `DocumentService.extract_text_from_pdf` to automatically store preprocessed clauses in `DocumentContextStore` and return `document_id`.
- Added prompt template `app/prompts/document_qa.txt` with `{{question}}` and `{{clauses_context}}` placeholders, instructing Gemini to answer strictly using provided context, return used `source_clauses` IDs, and signal `cannot_answer: true` if context is insufficient.
- Added `AITask.DOCUMENT_QA` to `AITask` enum in `app/schemas/ai.py`.
- Added `DOCUMENT_QA` branch to `PromptBuilder` in `app/ai/prompt_builder.py`.
- Added domain-specific exceptions `DocumentContextNotFoundError` and `QAGenerationError` in `app/services/exceptions.py`.
- Mounted `qa.router` in `app/main.py`.
- Created comprehensive 23-test suite in `tests/test_qa.py` covering schemas, retrieval unit tests (term matching, ranking, min_score threshold filtering, irrelevant question regression tests, zero AI calls on unrelated questions), service unit tests (success, missing doc context, grounded source clauses, cannot_answer early exit, invalid AI JSON, AI provider failure, confidence validation, large doc retrieval), and endpoint tests (200 OK, 422 validation error, 404 not found error).
- Updated project documentation across `README.md`, `API.md`, `ARCHITECTURE.md`, `CHANGELOG.md`, `TESTING.md`, `FRONTEND_INTEGRATION.md`, `DEPLOYMENT.md`, and `DATABASE.md`.


---

## [Sprint 6] - Clause Risk Analysis


### Added
- Created `POST /clauses/analyze` endpoint in dedicated router `app/api/routers/clause_analysis.py`.
- Created `ClauseAnalysisService` in `app/services/clause_analysis.py` for batching all clauses into a single AI request and orchestrating risk analysis.
- Defined `RiskLevel` enum (`low`, `medium`, `high`) and `ClauseRiskResult` (with mandatory `clause_id`, `clause_number`, `risk_level`, `explanation`, `recommendation`, `confidence`) in `app/schemas/clause_analysis.py`.
- Defined `ClauseAnalysisRequest` (inheriting `DocumentContext`) and `ClauseAnalysisResponse` (with `total_clauses`) in `app/schemas/clause_analysis.py`.
- Defined internal `ClauseAnalysisAIResponse` wrapper for ResponseParser validation.
- Added prompt template `app/prompts/clause_risk.txt` with `{{clauses_json}}` placeholder, per-risk-level guidelines, and per-level recommendation guidelines; instructs Gemini to return all six fields including `recommendation`.
- Added `AITask.CLAUSE_ANALYSIS` to `AITask` enum in `app/schemas/ai.py`.
- Added `CLAUSE_ANALYSIS` branch to `PromptBuilder` in `app/ai/prompt_builder.py` (reads `clause_risk.txt`, substitutes `{{clauses_json}}`).
- Added domain-specific exception `ClauseAnalysisError` in `app/services/exceptions.py`.
- Mounted `clause_analysis.router` in `app/main.py`.
- Created comprehensive 15-test suite in `tests/test_clause_analysis.py` covering PromptBuilder, service unit tests (success, empty input, AI errors, invalid JSON, schema validation failure, large documents, confidence range), all 3 endpoint scenarios, and 3 dedicated Pydantic schema validation tests (invalid `risk_level`, out-of-range `confidence`, missing `recommendation`).
- Updated project documentation across `README.md`, `API.md`, `ARCHITECTURE.md`, `CHANGELOG.md`, `TESTING.md`, `FRONTEND_INTEGRATION.md`, `DEPLOYMENT.md`, and `DATABASE.md`.

---

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
