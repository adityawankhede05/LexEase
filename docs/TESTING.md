# LexEase Testing Documentation

## Overview

LexEase uses `pytest` and `anyio` for asynchronous unit and integration testing of the FastAPI backend.
All external AI API calls are completely mocked during automated testing to ensure 100% offline, deterministic, and fast test execution.

---

## Test Suites

The backend test suite is located under `backend/tests/`:

1. **`test_ai_foundation.py`**:
   - Provider initialization and missing API key error handling.
   - Mocked Gemini provider generate calls.
   - Retries on transient rate-limit / timeout errors.
   - PromptBuilder formatting for simplification tasks.
   - ResponseParser JSON decoding and Pydantic validation.

2. **`test_ai_providers.py` (Multi-Provider AI Tests)**:
   - **OpenRouter Tests**: Missing API key validation, successful generation, system instruction formatting, immediate 401/403 error propagation, 429 rate limit retry & backoff, 503 transient error retry, retry exhaustion after MAX_RETRIES, empty/malformed choices JSON error handling, timeout & network connection error retries.
   - **Groq Tests**: Missing API key validation, successful generation, system instruction formatting, immediate 401/403 error propagation, 429 rate limit retry & backoff, 503 transient error retry, retry exhaustion after MAX_RETRIES, empty/malformed choices JSON error handling, timeout & network connection error retries.
   - **Provider Factory Tests**: Dynamic resolution of `gemini` -> `GeminiProvider`, `groq` -> `GroqProvider`, `openrouter` -> `OpenRouterProvider`, explicit function argument override, and validation of unsupported provider strings raising domain `AIProviderError`.

3. **`test_documents.py`**:
   - `/api/health` system health endpoint.
   - `/documents/upload` PDF upload, text extraction, clause segmentation, and Indian PII masking.
   - Error handling for invalid file extensions, corrupt PDFs, and non-extractable text.

4. **`test_summary.py`**:
   - `PromptBuilder` for `AITask.DOCUMENT_SUMMARY`.
   - `DocumentSummaryService` concatenation and orchestration using `MockAIProvider`.
   - Domain exception handling (`EmptyClauseListError`, `SummaryGenerationError`).
   - Large document summarization with 50+ clause segments.
   - Endpoint tests for `POST /documents/summarize` (200 OK, 400 Bad Request, 500 Internal Server Error).

5. **`test_clause_analysis.py`**:
   - `PromptBuilder` for `AITask.CLAUSE_ANALYSIS`: verifies prompt contains `clause_id`, `risk_level`, `explanation`, `recommendation`, and `confidence` keywords.
   - `ClauseAnalysisService` success path: validates all fields in returned `ClauseRiskResult`, including `recommendation` as a non-empty string.
   - Empty clause list raises `EmptyClauseListError`.
   - `AIProviderError` is wrapped and re-raised as `ClauseAnalysisError`.
   - Malformed JSON from AI raises `ClauseAnalysisError`.
   - Schema validation failure (missing `recommendation`) raises `ClauseAnalysisError`.
   - Large document (54 clauses) handled in a single batched request; all results returned.
   - All `confidence` values validated within `[0.0, 1.0]`.
   - Endpoint test for `POST /clauses/analyze` (200 OK): verifies `total_clauses`, `risk_level`, and `recommendation` in response body.
   - Endpoint test for empty clause list -> 400 Bad Request.
   - Endpoint test for AI failure -> 500 Internal Server Error.
   - Pydantic schema tests for `risk_level`, `confidence`, and `recommendation`.

6. **`test_qa.py`**:
   - `DocumentQARequest` schema validation.
   - `DocumentQAService` missing document handling.
   - `ClauseRetrievalService` unit tests: relevant term matching, stop-word filtering, term-overlap ranking order, zero-overlap empty list.
   - Service orchestration with `MockAIProvider`.
   - Early exit optimization: returns `cannot_answer: true` without making AI provider call when retrieval yields zero matches (`mock_provider.call_count == 0`).
   - Minimum relevance threshold regression tests for unrelated questions.
   - End-to-end regression test for unrelated questions.
   - Genuine multi-term match test for valid legal queries.
   - **Monthly rent due date regression tests**: Verifies that compound questions (*"What is the monthly rent and when is it due?"*) accurately match rental clauses with morphological variations and trigger successful end-to-end AI completion with `cannot_answer: false`.
   - Error handling and confidence range validation.
   - Large document retrieval from 50+ clauses.
   - `POST /documents/ask` endpoint tests: 200 OK success, 422 validation errors, 404 document-not-found error.

---

## Execution Commands

Navigate to `backend/` directory:

```bash
cd backend
```

Run all tests:
```bash
uv run pytest
```

Run all tests with verbose output:
```bash
uv run pytest -v
```

Run only AI provider tests:
```bash
uv run pytest tests/test_ai_providers.py
```

Run only Q&A tests:
```bash
uv run pytest tests/test_qa.py
```

Run only clause analysis tests:
```bash
uv run pytest tests/test_clause_analysis.py
```

Run only summary tests:
```bash
uv run pytest tests/test_summary.py
```

---

## Test Results

```
94 passed in ~7s
```
