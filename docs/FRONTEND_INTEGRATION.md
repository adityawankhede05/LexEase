# Frontend Integration Guide

## Overview

This document details how the React SPA frontend integrates with the LexEase backend API endpoints.

---

## Sprint 5 – Summarization Workflow

### Summarization Steps

1. **Step 1: Document Upload & Preprocessing**
   - User uploads a PDF document via `POST /documents/upload`.
   - Frontend receives `DocumentUploadResponse` containing `clauses: list[ClauseSegment]`.

2. **Step 2: Document Summarization**
   - Frontend posts the array of preprocessed clauses to `POST /documents/summarize`.
   - Backend returns `DocumentSummaryResponse` with executive summary, key points, and document classification.

### API Specification

#### Endpoint
`POST http://localhost:8000/documents/summarize`

#### Request Headers
`Content-Type: application/json`

#### Example Request Body
```json
{
  "clauses": [
    {
      "clause_id": "clause_1",
      "clause_number": "1.0",
      "text": "The Licensee shall pay monthly rent of INR 25,000 on or before the 5th day of every calendar month."
    },
    {
      "clause_id": "clause_2",
      "clause_number": "2.0",
      "text": "Either party may terminate this agreement by providing 30 days written notice."
    }
  ]
}
```

#### Example Response (200 OK)
```json
{
  "summary": "This is a license agreement requiring a monthly rent payment of INR 25,000 due by the 5th of each month, with a mutual 30-day notice termination clause.",
  "key_points": [
    "Monthly rent: INR 25,000 due by the 5th",
    "Termination notice period: 30 days written notice"
  ],
  "document_type": "License Agreement"
}
```

### React / TypeScript Integration Example

```typescript
import axios from 'axios';

interface ClauseSegment {
  clause_id: string;
  clause_number: string | null;
  text: string;
}

interface DocumentSummaryResponse {
  summary: string;
  key_points: string[];
  document_type: string | null;
}

export async function summarizeDocument(clauses: ClauseSegment[]): Promise<DocumentSummaryResponse> {
  const response = await axios.post<DocumentSummaryResponse>(
    'http://localhost:8000/documents/summarize',
    { clauses }
  );
  return response.data;
}
```

---

## Sprint 6 – Clause Risk Analysis Workflow

### Clause Analysis Steps

1. **Step 1: Document Upload & Preprocessing**
   - User uploads a PDF document via `POST /documents/upload`.
   - Frontend receives `DocumentUploadResponse` containing `clauses: list[ClauseSegment]`.

2. **Step 2: Clause Risk Analysis**
   - Frontend posts the same array of preprocessed clauses to `POST /clauses/analyze`.
   - Backend returns `ClauseAnalysisResponse` with a risk classification, explanation, actionable recommendation, and confidence score for every clause — all analyzed in a **single AI request**.

### API Specification

#### Endpoint
`POST http://localhost:8000/clauses/analyze`

#### Request Headers
`Content-Type: application/json`

#### Example Request Body
```json
{
  "clauses": [
    {
      "clause_id": "clause_1",
      "clause_number": "1.0",
      "text": "The tenant shall pay monthly rent of INR 25,000 on or before the 5th day of every calendar month."
    },
    {
      "clause_id": "clause_2",
      "clause_number": "2.0",
      "text": "The landlord may terminate this agreement without any notice at any time."
    }
  ]
}
```

#### Example Response (200 OK)
```json
{
  "total_clauses": 2,
  "results": [
    {
      "clause_id": "clause_1",
      "clause_number": "1.0",
      "risk_level": "low",
      "explanation": "This is a standard rent payment clause with a clear due date and no unusual penalty terms.",
      "recommendation": "Review for completeness; this clause is generally acceptable as written.",
      "confidence": 0.91
    },
    {
      "clause_id": "clause_2",
      "clause_number": "2.0",
      "risk_level": "high",
      "explanation": "This clause grants the landlord unchecked termination rights with no notice requirement, which is highly one-sided and offers the tenant no protection.",
      "recommendation": "Negotiate a minimum notice period (e.g., 30 days) and seek independent legal advice before signing.",
      "confidence": 0.95
    }
  ]
}
```

#### Error Responses

| HTTP Status | Condition | Detail |
|-------------|-----------|--------|
| 400 | Empty clause list | `"Clause list cannot be empty for clause risk analysis."` |
| 500 | AI generation failure | `"Clause risk analysis failed: ..."` |

### React / TypeScript Integration Example

```typescript
import axios from 'axios';

interface ClauseSegment {
  clause_id: string;
  clause_number: string | null;
  text: string;
}

type RiskLevel = 'low' | 'medium' | 'high';

interface ClauseRiskResult {
  clause_id: string;
  clause_number: string | null;
  risk_level: RiskLevel;
  explanation: string;
  recommendation: string;
  confidence: number;  // 0.0 – 1.0
}

interface ClauseAnalysisResponse {
  total_clauses: number;
  results: ClauseRiskResult[];
}

export async function analyzeClauses(
  clauses: ClauseSegment[]
): Promise<ClauseAnalysisResponse> {
  const response = await axios.post<ClauseAnalysisResponse>(
    'http://localhost:8000/clauses/analyze',
    { clauses }
  );
  return response.data;
}
```

### Recommended UI Display Pattern

For each `ClauseRiskResult`, the frontend can render:

- A **risk badge** styled by `risk_level`:  
  - `low` → green  
  - `medium` → amber  
  - `high` → red  
- The **explanation** paragraph below the clause text.
- An **actionable recommendation card** (e.g., a callout box) advising the user what to do before signing.
- A **confidence indicator** (e.g., percentage bar or label) showing the model's certainty.

---

## Sprint 7 – Grounded Document Q&A Workflow

### Q&A Steps

1. **Step 1: Document Upload & Preprocessing**
   - User uploads a PDF document via `POST /documents/upload`.
   - Frontend receives `DocumentUploadResponse` containing `clauses: list[ClauseSegment]` and `document_id: string`.

2. **Step 2: Interactive Document Q&A**
   - User asks questions about the uploaded document in an interactive chat interface.
   - Frontend posts `{ document_id, question }` to `POST /documents/ask`.
   - Backend performs lexical relevance retrieval with a minimum score threshold (`min_score = 0.6`). If the question is unrelated to the document (e.g. *"What is the capital of France?"*), the backend returns `cannot_answer: true` immediately **without calling the AI provider**. Otherwise it queries Gemini and returns `DocumentQAResponse` with `answer`, `source_clauses`, `confidence`, and `cannot_answer`.


### API Specification

#### Endpoint
`POST http://localhost:8000/documents/ask`

#### Request Headers
`Content-Type: application/json`

#### Example Request Body
```json
{
  "document_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "question": "What is the monthly rent amount and when is it due?"
}
```

#### Example Response (200 OK)
```json
{
  "answer": "The monthly rent is INR 25,000 and is due on or before the 5th day of every calendar month.",
  "source_clauses": ["clause_1"],
  "confidence": 0.95,
  "cannot_answer": false
}
```

#### Example Response (Unanswerable Context)
```json
{
  "answer": "The provided document does not contain relevant information to answer this question.",
  "source_clauses": [],
  "confidence": 0.0,
  "cannot_answer": true
}
```

#### Error Responses

| HTTP Status | Condition | Detail |
|-------------|-----------|--------|
| 422 | Missing/invalid fields or question > 2000 chars | Validation error payload |
| 404 | Unknown `document_id` | `"No document context found for document_id: ..."` |
| 500 | AI generation failure | `"Document Q&A failed: ..."` |

### React / TypeScript Integration Example

```typescript
import axios from 'axios';

interface DocumentQARequest {
  document_id: string;
  question: string;
}

interface DocumentQAResponse {
  answer: string;
  source_clauses: string[];
  confidence: number;       // 0.0 – 1.0
  cannot_answer: boolean;
}

export async function askDocumentQuestion(
  documentId: string,
  question: string
): Promise<DocumentQAResponse> {
  const response = await axios.post<DocumentQAResponse>(
    'http://localhost:8000/documents/ask',
    { document_id: documentId, question }
  );
  return response.data;
}
```

### Recommended UI Display Pattern

For each Q&A turn in the chat UI:

- Render user message bubbled to the right.
- Render assistant response bubbled to the left.
- If `cannot_answer` is `true`: display an informational callout indicating the document has no matching information for this question.
- If `cannot_answer` is `false`:
  - Show the **grounded answer**.
  - Render **source clause chips** (e.g. `Clause 1`) linking or highlighting the matching clause segments in the document viewer.
  - Show a small **confidence indicator** badge.

