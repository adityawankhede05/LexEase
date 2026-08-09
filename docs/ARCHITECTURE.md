# LexEase System Architecture

## Overview

LexEase is designed around a modular, layered backend architecture built with FastAPI, Pydantic, and Google Gemini API.

```
+-----------------------------------------------------------------------------------------------------+
|                                          API Layer                                                  |
|  GET /api/health  |  POST /documents/upload  |  POST /documents/summarize  |  POST /clauses/analyze |
+-------------------+--------------------------+-----------------------------+------------------------+
                                              |
                                              v
+-----------------------------------------------------------------------------------------------------+
|                                       Services Layer                                                |
|   DocumentService   |   DocumentSummaryService   |   ClauseAnalysisService   |   AIService          |
+---------------------+----------------------------+---------------------------+----------------------+
                                              |
                                              v
+-----------------------------------------------------------------------------------------------------+
|                                    AI Foundation Layer                                              |
|            PromptBuilder  <-->  GeminiProvider  <-->  ResponseParser                                |
+-----------------------------------------------------------------------------------------------------+
```

---

## 1. Request Invocation Pipeline (`POST /documents/summarize`)

When a client submits a list of preprocessed clause segments for whole document summarization, execution proceeds strictly through the following steps:

1. **Router (`app/api/routers/summary.py`)**:
   - Accepts `DocumentSummaryRequest` (inheriting from `DocumentContext`).
   - Delegates request processing to `DocumentSummaryService`.
   - Catches domain-specific exceptions (`EmptyClauseListError` → 400 Bad Request, `SummaryGenerationError` → 500 Internal Server Error).

2. **Summary Service (`app/services/summary.py`)**:
   - Validates that `clauses` array is non-empty.
   - Formats each `ClauseSegment` (including clause numbers if present) and concatenates them into a unified full-document text string.
   - Formulates the payload `{"document_text": full_document_text}`.
   - Invokes `AIService.generate(AITask.DOCUMENT_SUMMARY, payload, DocumentSummaryResponse)`.

3. **AI Service (`app/services/ai.py`)**:
   - Orchestrates prompt creation via `PromptBuilder`.
   - Invokes the configured provider (`GeminiProvider` implementing `BaseAIProvider`).
   - Passes raw response string to `ResponseParser`.

4. **Prompt Builder (`app/ai/prompt_builder.py`)**:
   - Reads `app/prompts/document_summary.txt`.
   - Substitutes `{{document}}` placeholder with concatenated document text.
   - Enforces strict JSON prompt formatting instructions.

5. **AI Provider (`app/ai/gemini_provider.py`)**:
   - Communicates asynchronously with the Google Gemini API using structured JSON configuration.
   - Implements exponential backoff retries for transient errors.

6. **Response Parser (`app/ai/response_parser.py`)**:
   - Strips backticks/markdown if present.
   - Parses JSON string and validates against `DocumentSummaryResponse`.

---

## 2. Request Invocation Pipeline (`POST /clauses/analyze`)

When a client submits a list of preprocessed clause segments for clause-level risk analysis, execution proceeds strictly through the following steps:

1. **Router (`app/api/routers/clause_analysis.py`)**:
   - Accepts `ClauseAnalysisRequest` (inheriting from `DocumentContext`).
   - Delegates request processing to `ClauseAnalysisService`.
   - Catches domain-specific exceptions (`EmptyClauseListError` → 400 Bad Request, `ClauseAnalysisError` → 500 Internal Server Error).

2. **Clause Analysis Service (`app/services/clause_analysis.py`)**:
   - Validates that `clauses` array is non-empty.
   - Serializes all `ClauseSegment` objects (`clause_id`, `clause_number`, `text`) into a single JSON array string.
   - Formulates the payload `{"clauses_json": <json_string>}` — **a single batched request** covering all clauses.
   - Invokes `AIService.generate(AITask.CLAUSE_ANALYSIS, payload, ClauseAnalysisAIResponse)`.
   - Wraps `ClauseAnalysisAIResponse.results` into `ClauseAnalysisResponse` with `total_clauses`.

3. **AI Service (`app/services/ai.py`)**:
   - Orchestrates prompt creation via `PromptBuilder`.
   - Invokes the configured provider (`GeminiProvider`).
   - Passes raw response string to `ResponseParser`.

4. **Prompt Builder (`app/ai/prompt_builder.py`)**:
   - Reads `app/prompts/clause_risk.txt`.
   - Substitutes `{{clauses_json}}` placeholder with the serialized clause array.
   - Instructs the model to echo `clause_id` and `clause_number` verbatim for deterministic result mapping.

5. **AI Provider (`app/ai/gemini_provider.py`)**:
   - Communicates asynchronously with the Google Gemini API.
   - Implements exponential backoff retries for transient errors.

6. **Response Parser (`app/ai/response_parser.py`)**:
   - Strips backticks/markdown if present.
   - Parses JSON string and validates against `ClauseAnalysisAIResponse`.
   - Pydantic automatically rejects any result missing a mandatory field (e.g., `recommendation`).

---

## 3. Reusable Schemas

- **`DocumentContext`**: Reusable base schema containing `clauses: list[ClauseSegment]`.
- **`DocumentSummaryRequest`**: Inherits from `DocumentContext`.
- **`DocumentSummaryResponse`**: Contains `summary: str`, `key_points: list[str]`, and optional `document_type: str | None`.
- **`ClauseAnalysisRequest`**: Inherits from `DocumentContext` (same pattern as `DocumentSummaryRequest`).
- **`RiskLevel`**: `StrEnum` with values `low`, `medium`, `high`.
- **`ClauseRiskResult`**: Contains `clause_id`, `clause_number`, `risk_level` (`RiskLevel`), `explanation`, `recommendation`, and `confidence` (float, 0.0–1.0, validated via `Field(ge=0.0, le=1.0)`). All fields are mandatory.
- **`ClauseAnalysisAIResponse`**: Internal wrapper `{ results: list[ClauseRiskResult] }` used exclusively by `ResponseParser`.
- **`ClauseAnalysisResponse`**: Public HTTP response containing `total_clauses: int` and `results: list[ClauseRiskResult]`.
