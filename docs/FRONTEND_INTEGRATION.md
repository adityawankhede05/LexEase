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
