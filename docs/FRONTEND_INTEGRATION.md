# Frontend Integration Guide for Summarization

## Overview

This document details how the React SPA frontend integrates with the `POST /documents/summarize` backend API endpoint introduced in Sprint 5.

---

## Summarization Workflow

1. **Step 1: Document Upload & Preprocessing**
   - User uploads a PDF document via `POST /documents/upload`.
   - Frontend receives `DocumentUploadResponse` containing `clauses: list[ClauseSegment]`.

2. **Step 2: Document Summarization**
   - Frontend posts the array of preprocessed clauses to `POST /documents/summarize`.
   - Backend returns `DocumentSummaryResponse` with executive summary, key points, and document classification.

---

## API Specification

### Endpoint
`POST http://localhost:8000/documents/summarize`

### Request Headers
`Content-Type: application/json`

### Example Request Body
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

### Example Response (200 OK)
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

---

## React / TypeScript Integration Example

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
