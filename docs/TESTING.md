# LexEase Testing Documentation

## Overview

LexEase uses `pytest` and `anyio` for asynchronous unit and integration testing of the FastAPI backend.

---

## Test Suites

The backend test suite is located under `backend/tests/`:

1. **`test_ai_foundation.py`**:
   - Provider initialization and missing API key error handling.
   - Mocked Gemini provider generate calls.
   - Retries on transient rate-limit / timeout errors.
   - PromptBuilder formatting for simplification tasks.
   - ResponseParser JSON decoding and Pydantic validation.

2. **`test_documents.py`**:
   - `/api/health` system health endpoint.
   - `/documents/upload` PDF upload, text extraction, clause segmentation, and Indian PII masking.
   - Error handling for invalid file extensions, corrupt PDFs, and non-extractable text.

3. **`test_summary.py`**:
   - `PromptBuilder` for `AITask.DOCUMENT_SUMMARY`.
   - `DocumentSummaryService` concatenation and orchestration using `MockAIProvider`.
   - Domain exception handling (`EmptyClauseListError`, `SummaryGenerationError`).
   - Large document summarization with 50+ clause segments.
   - Endpoint tests for `POST /documents/summarize` (200 OK, 400 Bad Request, 500 Internal Server Error).

4. **`test_clause_analysis.py`**:
   - `PromptBuilder` for `AITask.CLAUSE_ANALYSIS`: verifies prompt contains `clause_id`, `risk_level`, `explanation`, `recommendation`, and `confidence` keywords.
   - `ClauseAnalysisService` success path: validates all fields in returned `ClauseRiskResult`, including `recommendation` as a non-empty string.
   - Empty clause list raises `EmptyClauseListError`.
   - `AIProviderError` is wrapped and re-raised as `ClauseAnalysisError`.
   - Malformed JSON from AI raises `ClauseAnalysisError`.
   - Schema validation failure (missing `recommendation`) raises `ClauseAnalysisError`.
   - Large document (54 clauses) handled in a single batched request; all results returned.
   - All `confidence` values validated within `[0.0, 1.0]`.
   - Endpoint test for `POST /clauses/analyze` (200 OK): verifies `total_clauses`, `risk_level`, and `recommendation` in response body.
   - Endpoint test for empty clause list → 400 Bad Request.
   - Endpoint test for AI failure → 500 Internal Server Error.
   - Pydantic schema test: invalid `risk_level` string raises `ValidationError`.
   - Pydantic schema test: `confidence` outside `[0.0, 1.0]` raises `ValidationError`.
   - Pydantic schema test: missing `recommendation` field raises `ValidationError`.

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

Run only clause analysis tests:
```bash
uv run pytest tests/test_clause_analysis.py
```

Run only summary tests:
```bash
uv run pytest tests/test_summary.py
```

Run tests with verbose output:
```bash
uv run pytest -v
```

---

## Test Results

```
46 passed in ~9s
```
