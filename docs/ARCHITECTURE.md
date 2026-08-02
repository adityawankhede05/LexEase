# LexEase System Architecture

## Overview

LexEase is designed around a modular, layered backend architecture built with FastAPI, Pydantic, and Google Gemini API.

```
+-----------------------------------------------------------------------+
|                             API Layer                                 |
|   GET /api/health    |    POST /documents/upload   |   POST /documents/summarize |
+-----------------------+-----------------------------+-----------------+
                                      |
                                      v
+-----------------------------------------------------------------------+
|                           Services Layer                              |
|   DocumentService    |   DocumentSummaryService    |    AIService     |
+----------------------+-----------------------------+------------------+
                                      |
                                      v
+-----------------------------------------------------------------------+
|                        AI Foundation Layer                            |
| PromptBuilder  <-->  GeminiProvider  <-->  ResponseParser             |
+-----------------------------------------------------------------------+
```

---

## 1. Request Invocation Pipeline (`POST /documents/summarize`)

When a client submits a list of preprocessed clause segments for whole document summarization, execution proceeds strictly through the following steps:

1. **Router (`app/api/routers/summary.py`)**:
   - Accepts `DocumentSummaryRequest` (inheriting from `DocumentContext`).
   - Delegates request processing to `DocumentSummaryService`.
   - Catches domain-specific exceptions (`EmptyClauseListError` $\rightarrow$ 400 Bad Request, `SummaryGenerationError` $\rightarrow$ 500 Internal Server Error).

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

## 2. Reusable Schemas

- **`DocumentContext`**: Reusable base schema containing `clauses: list[ClauseSegment]`.
- **`DocumentSummaryRequest`**: Inherits from `DocumentContext`.
- **`DocumentSummaryResponse`**: Contains `summary: str`, `key_points: list[str]`, and optional `document_type: str | None`.
