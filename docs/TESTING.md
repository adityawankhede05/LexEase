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
31 passed in 6.89s
```
