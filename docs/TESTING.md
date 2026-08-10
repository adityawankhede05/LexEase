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

5. **`test_qa.py`**:
   - `DocumentQARequest` schema validation: accepts valid IDs/questions, rejects empty question, rejects question > 2000 chars.
   - `DocumentQAService` missing document handling: raises `DocumentContextNotFoundError` when `document_id` is invalid.
   - `ClauseRetrievalService` unit tests: relevant term matching, stop-word filtering, term-overlap ranking order, zero-overlap empty list.
   - Service orchestration with `MockAIProvider`: verifies valid `DocumentQAResponse` structure and grounded `source_clauses` list.
   - Early exit optimization: returns `cannot_answer: true` without making AI provider call when retrieval yields zero matches (verified by `mock_provider.call_count == 0`).
   - **Minimum relevance threshold regression tests**: Verifies unrelated questions — `"What is the capital of France?"`, `"What is the recipe for making pizza?"`, `"Who won the World Cup?"` — return `[]` from `ClauseRetrievalService` even when legal documents contain partial term overlaps (e.g. `"capital"` in `"working capital"`).
   - **End-to-end regression test**: Verifies `DocumentQAService` returns `cannot_answer: true` with `confidence: 0.0` and **zero AI provider calls** for the question `"What is the capital of France?"` against a legal document.
   - **Genuine multi-term match test**: Verifies that valid legal questions with multi-term overlap (e.g. `"What is the security deposit amount?"`) still pass the `min_score = 0.6` threshold and return the correct clause.
   - Error handling: wraps malformed JSON and `AIProviderError` into `QAGenerationError`.
   - `DocumentQAResponse` Pydantic confidence range validation `[0.0, 1.0]`.
   - Large document retrieval: selects top relevant clauses from 50+ clause document.
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
69 passed in ~6s
```

