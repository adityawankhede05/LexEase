# LexEase Backend API Documentation

**Version:** 0.1.0  
**Backend Framework:** FastAPI  
**Base URL (Development):**
```
http://localhost:8000
```

---

# Overview

LexEase is an AI-powered legal document simplification platform.

Current backend capabilities include:

- Health Check
- PDF Upload
- PDF Text Extraction
- Text Cleaning
- Clause Segmentation
- Indian PII Masking
- AI Foundation Layer (Internal)
- Google Gemini Provider Integration (Internal)
- Whole Document Summarization
- Clause-Level Legal Risk Analysis

---

# Authentication

Currently **no authentication** is required.

Future versions will use session-based APIs.

---

# Response Codes

| Code | Meaning |
|------|----------|
|200|Success|
|400|Bad Request|
|404|Not Found|
|415|Unsupported Media Type|
|422|Validation Error|
|500|Internal Server Error|

---

# API Endpoints

---

# 1. Health Check

## Endpoint

```
GET /api/health
```

## Description

Checks whether the backend server is running.

---

### Success Response

```json
{
    "status": "healthy"
}
```

---

### Status Codes

|Code|Meaning|
|----|-------|
|200|Backend running|

---

# 2. Upload Legal Document

## Endpoint

```
POST /documents/upload
```

## Description

Uploads a legal PDF document.

The backend automatically:

1. Validates the PDF
2. Extracts text
3. Cleans formatting
4. Segments clauses
5. Masks Indian PII
6. Returns structured clauses

---

## Request

Content-Type

```
multipart/form-data
```

### Parameters

|Name|Type|Required|Description|
|----|----|--------|-----------|
|file|PDF|Yes|Legal document|

---

## Success Response

```json
{
    "filename": "RentalAgreement.pdf",
    "page_count": 6,
    "character_count": 7852,
    "clauses": [
        {
            "clause_id": "clause_1",
            "clause_number": "1",
            "text": "This agreement begins on..."
        },
        {
            "clause_id": "clause_2",
            "clause_number": "2.1",
            "text": "The tenant shall pay..."
        }
    ]
}
```

---

## Response Fields

|Field|Description|
|------|-----------|
|filename|Uploaded PDF name|
|page_count|Total PDF pages|
|character_count|Characters extracted before masking|
|clauses|List of cleaned clauses|

---

## Clause Object

```json
{
    "clause_id": "clause_3",
    "clause_number": "3.2",
    "text": "Masked and cleaned clause text"
}
```

|Field|Description|
|------|-----------|
|clause_id|Internal identifier|
|clause_number|Clause number if detected|
|text|Cleaned + masked clause|

---

# Automatic Processing Pipeline

Every uploaded PDF passes through the following pipeline:

```
PDF Upload
      │
      ▼
PDF Validation
      │
      ▼
Text Extraction (PyMuPDF)
      │
      ▼
Text Cleaning
      │
      ▼
Clause Segmentation
      │
      ▼
PII Masking
      │
      ▼
JSON Response
```

---

# PII Masking

Currently supported:

|Type|Replacement|
|----|-----------|
|Email|[EMAIL]|
|Phone|[PHONE]|
|Aadhaar|[AADHAAR]|
|PAN|[PAN]|

Example

Input

```
Email:
abc@gmail.com

Phone:
9876543210

PAN:
ABCDE1234F
```

Output

```
Email:
[EMAIL]

Phone:
[PHONE]

PAN:
[PAN]
```

---

# Possible Errors

## Invalid File Type

Status

```
400
```

Response

```json
{
    "detail": "Only PDF files are supported."
}
```

---

## Empty File

```json
{
    "detail": "Uploaded file is empty."
}
```

---

## Corrupted PDF

```json
{
    "detail": "Invalid or corrupted PDF."
}
```

---

## PDF Without Extractable Text

```json
{
    "detail": "No extractable text found in PDF."
}
```

---

# Internal AI Architecture

> These components are internal and **are not exposed as REST APIs.**

Current modules

- AIService
- BaseAIProvider
- GeminiProvider
- PromptBuilder
- ResponseParser

These modules will be consumed by future endpoints.

---

# 3. Summarize Legal Document

## Endpoint

```
POST /documents/summarize
```

## Description

Summarizes a legal document represented as a list of preprocessed clause segments (produced by `/documents/upload`).

The backend automatically:
1. Validates the non-empty clause segment array (`DocumentSummaryRequest` extending `DocumentContext`).
2. Concatenates clause segments into full document text.
3. Invokes `AIService` with `AITask.DOCUMENT_SUMMARY` and `GeminiProvider`.
4. Parses and validates the AI response against `DocumentSummaryResponse`.
5. Returns a structured JSON summary.

---

## Request

Content-Type

```json
application/json
```

### Request Body Schema (`DocumentSummaryRequest`)

```json
{
  "clauses": [
    {
      "clause_id": "clause_1",
      "clause_number": "1.1",
      "text": "The tenant shall pay monthly rent on or before the 5th of each month."
    },
    {
      "clause_id": "clause_2",
      "clause_number": "2.1",
      "text": "The landlord reserves the right to inspect the property with 24 hours notice."
    }
  ]
}
```

---

## Success Response (200 OK)

```json
{
  "summary": "This rental agreement sets out the tenant's obligation to make prompt monthly rent payments by the 5th of each month and grants the landlord right of property inspection with advance notice.",
  "key_points": [
    "Rent must be paid monthly by the 5th",
    "Landlord may inspect property after giving 24 hours advance notice"
  ],
  "document_type": "Rental Agreement"
}
```

---

## Error Responses

### Empty Clause List (400 Bad Request)

```json
{
  "detail": "Clause list cannot be empty for document summarization."
}
```

### AI Generation Failure (500 Internal Server Error)

```json
{
  "detail": "Document summarization failed: Gemini API Error..."
}
```

---

---

# 4. Clause-Level Risk Analysis

## Endpoint

```
POST /clauses/analyze
```

## Description

Analyzes a list of preprocessed clause segments for legal risk from the perspective of an ordinary Indian citizen. All clauses are processed in a **single batched AI request**.

The backend automatically:
1. Validates the non-empty clause segment array (`ClauseAnalysisRequest` extending `DocumentContext`).
2. Serializes all clauses into a single JSON payload and constructs a batched prompt.
3. Invokes `AIService` with `AITask.CLAUSE_ANALYSIS` and `GeminiProvider`.
4. Parses and validates the AI response against `ClauseAnalysisAIResponse`.
5. Returns a structured JSON response with per-clause risk results.

---

## Request

Content-Type

```
application/json
```

### Request Body Schema (`ClauseAnalysisRequest`)

```json
{
  "clauses": [
    {
      "clause_id": "clause_1",
      "clause_number": "1.1",
      "text": "The tenant shall pay monthly rent on or before the 5th of each month."
    },
    {
      "clause_id": "clause_2",
      "clause_number": "2.1",
      "text": "The landlord may terminate the agreement without notice at any time."
    }
  ]
}
```

---

## Success Response (200 OK)

```json
{
  "total_clauses": 2,
  "results": [
    {
      "clause_id": "clause_1",
      "clause_number": "1.1",
      "risk_level": "low",
      "explanation": "This is a standard rent payment clause with a clear due date and no unusual penalty terms.",
      "recommendation": "Review for completeness; this clause is generally acceptable as written.",
      "confidence": 0.91
    },
    {
      "clause_id": "clause_2",
      "clause_number": "2.1",
      "risk_level": "high",
      "explanation": "This clause grants the landlord unchecked termination rights with no notice requirement, which is highly one-sided and offers the tenant no protection.",
      "recommendation": "Negotiate a minimum notice period (e.g., 30 days) and seek legal advice before signing.",
      "confidence": 0.95
    }
  ]
}
```

---

## Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `total_clauses` | int | Number of clauses analyzed |
| `results` | list | List of `ClauseRiskResult` objects |

### ClauseRiskResult Object

| Field | Type | Description |
|-------|------|-------------|
| `clause_id` | str | Echoed verbatim from input for deterministic mapping |
| `clause_number` | str \| null | Echoed verbatim from input |
| `risk_level` | `low` \| `medium` \| `high` | AI-assigned risk classification |
| `explanation` | str | 1–3 sentence plain-English risk explanation |
| `recommendation` | str | 1–2 sentence actionable advice before signing |
| `confidence` | float [0.0–1.0] | AI confidence score for the assigned risk level |

---

## Error Responses

### Empty Clause List (400 Bad Request)

```json
{
  "detail": "Clause list cannot be empty for clause risk analysis."
}
```

### AI Generation Failure (500 Internal Server Error)

```json
{
  "detail": "Clause risk analysis failed: Gemini API Error..."
}
```

---

# Upcoming Endpoints

These APIs are planned and are **not yet implemented**.

|Method|Endpoint|Purpose|
|------|--------|-------|
|POST|/chat|Grounded document Q&A|
|POST|/translate|Hindi/Marathi translation|

---

# Testing Status

Current backend test coverage:

- Health endpoint
- Upload endpoint
- Invalid file validation
- Empty file validation
- Corrupt PDF validation
- No-text PDF validation
- Text cleaning
- Clause segmentation
- PII masking
- AI Foundation Layer
- Response parsing
- Gemini provider integration
- Document summarization prompt building
- Whole document summarization service
- Large document summarization (50+ clauses)
- Summarization router endpoint (200, 400, 500 error cases)

Current Result

```
46 tests passed
```

---

# Technologies Used

|Component|Technology|
|----------|----------|
|Backend|FastAPI|
|Language|Python 3.11|
|PDF Parsing|PyMuPDF|
|Validation|Pydantic|
|Testing|Pytest|
|AI Provider|Google Gemini|
|Package Manager|uv|
|Containerization|Docker|
|Database|PostgreSQL (Upcoming Integration)|

---

# Project Status

|Sprint|Status|
|-------|------|
|Sprint 1 – Backend Architecture|Completed|
|Sprint 2 – PDF Upload & Extraction|Completed|
|Sprint 3 – Text Preprocessing|Completed|
|Sprint 4A – AI Foundation Layer|Completed|
|Sprint 4B – Gemini Integration|Completed|
|Sprint 5 – Whole Document Summarization|Completed|
|Sprint 6 – Clause Risk Analysis|Completed|