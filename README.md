# LexEase: AI-Powered Legal Document Simplification & Risk Analysis

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![React 18](https://img.shields.io/badge/React-18.2-cyan.svg)](https://react.dev/)
[![Vite 5](https://img.shields.io/badge/Vite-5.0-purple.svg)](https://vitejs.dev/)
[![Tests](https://img.shields.io/badge/Tests-157%20Passed-brightgreen.svg)]()

> **Disclaimer:** LexEase is an assistive educational and analysis tool designed for informational purposes. LexEase does not provide legal advice, and its AI-generated analyses, risk ratings, summaries, and proposed clause rewrites should never substitute for consultation with a qualified legal professional.

---

## 1. Project Overview

Legal agreements, service contracts, and commercial terms are frequently written in dense, archaic legalese with one-sided indemnities, sweeping liability caps, and buried termination penalties. For ordinary individuals, consumers, and small business owners, reading and understanding these documents before signing is often prohibitive.

**LexEase** bridges this gap. It processes legal PDF documents through an automated pipeline:
1. **Extracts and segments clauses** while redacting sensitive personal information (Indian PII: PAN, Aadhaar, phone numbers, email addresses).
2. **Evaluates legal risks** on a clause-by-clause basis, generating plain-English interpretations, confidence ratings, and actionable recommendations before signing.
3. **Produces executive document summaries** and extracts high-impact key takeaways.
4. **Supports grounded question-answering (Q&A)** against the document context with relevance threshold scoring and strict refusal of unrelated questions.
5. **Drafts safer alternative clauses** from the customer or service recipient's viewpoint and generates a proposed PDF with in-place reflow or addendum export.
6. **Collects consented feedback** for human reviewer evaluation and offline dataset curation.

---

## 2. Features

### Core Analysis & Document Understanding
* **PDF Upload & Structure Extraction:** Extracts raw text, normalizes whitespace, cleans extraction artifacts, and segments agreements into discrete, indexed clauses using PyMuPDF (`fitz`).
* **Indian PII Masking:** Automatically detects and masks sensitive personal identifiers (Aadhaar numbers, PAN cards, phone numbers, and email addresses) prior to LLM or ML processing.
* **Whole-Document Summarization (`POST /documents/summarize`):** Synthesizes document objectives into a concise executive summary, identifies the agreement type, and extracts critical bullet points.
* **Clause-by-Clause Legal Risk Analysis (`POST /clauses/analyze`):** Classifies each clause into **Low**, **Medium**, or **High** risk with:
  * Confidence score percentage.
  * Plain-English exposure explanation.
  * Practical advice and negotiating points before signing.
* **Grounded Document Q&A (`POST /documents/ask`):** Answers specific questions strictly based on the uploaded document's contents. Uses lexical relevance filtering (`min_score = 0.6`) to prevent hallucination, terminating early with `cannot_answer: true` if no supporting clauses exist.

### Proposed Drafting & PDF Export
* **Perspective-Driven Clause Rewriting (`POST /clauses/rewrite`):** Allows users to select up to 8 medium/high-risk clauses and generates balanced, customer-protective revisions via the configured AI provider after explicit user consent.
* **In-Place PDF Replacement Engine (`POST /documents/proposed-pdf`):** PyMuPDF reflows replacement clauses directly into the original document layout while preserving original headers, margins, and unmodified clauses.
* **Automatic Addendum Fallback:** If replacement text exceeds page capacity, touches embedded artwork/images, or cannot reflow safely, LexEase falls back to appending a clearly labeled **Proposed Clause Addendum** page to prevent text clipping.
* **DOCX Export (`POST /documents/proposed-draft`):** Generates a downloadable Word document (`.docx`) containing the approved revisions.

### Feedback & Adaptive Learning Reviewer Portal
* **Clause Flagging & User Suggestions (`POST /adaptive-learning/feedback`):** Users can flag questionable risk labels and propose corrections directly from the risk analysis UI. Submissions require explicit consent and are queued for reviewer inspection.
* **Document Batch Submissions (`POST /adaptive-learning/feedback/submissions`):** Users can contribute full analyzed document clause sets for reviewer annotation and dataset expansion.
* **Reviewer Administrator Dashboard (`/adaptive-learning/admin`):** Protected by an administrative reviewer secret token (`X-Adaptive-Admin-Token`):
  * Inspect and approve/reject flagged clause corrections.
  * Review and annotate batch document submissions.
  * Track approved dataset size across Low, Medium, and High risk categories.
* **Offline Dataset Isolation:** Approved feedback is stored in the database strictly as reference examples for future offline training and evaluation. Approval **does not** alter or retrain the active live inference model.
* **Evaluation Metrics & Visualizations:** Visualizes validation accuracy, Macro F1, High-Risk recall, and confusion matrix progression across candidate model evaluations.

---

## 3. Architecture & Technology Stack

```
┌────────────────────────────────────────────────────────┐
│              LexEase React SPA (Vite)                  │
│       Document Workspace · Clause Drafting · Admin     │
└───────────────────────────┬────────────────────────────┘
                            │ HTTP / REST (Axios)
                            ▼
┌────────────────────────────────────────────────────────┐
│                   FastAPI Backend                      │
│        Lifespan DB Init · CORS · Exception Handling    │
└──────┬────────────────────┬────────────────────┬───────┘
       │                    │                    │
       ▼                    ▼                    ▼
┌──────────────┐   ┌───────────────────┐   ┌────────────────────┐
│  SQLAlchemy  │   │ PyMuPDF Engine    │   │  AI Foundation     │
│  ORM Layer   │   │ Text Extraction   │   │  BaseAIProvider    │
└──────┬───────┘   │ In-Place PDF Reflow│   └─────────┬──────────┘
       │           └───────────────────┘             │
 ┌─────┴──────┐                     ┌────────────────┼────────────────┐
 │ SQLite /   │                     ▼                ▼                ▼
 │ PostgreSQL │               Google Gemini         Groq         OpenRouter
 └────────────┘             (gemini-2.0-flash) (gpt-oss-120b)   (Cerebras)
```

### Technology Breakdown
* **Frontend:** React 18, Vite 5, Tailwind CSS, React Router v6, Axios.
* **Backend:** FastAPI, Python 3.11, Pydantic v2, Uvicorn, PyMuPDF (fitz), python-docx.
* **Database & ORM:** SQLAlchemy 2.0 with automatic table creation via `init_db()` on application startup. Supports SQLite for local development and PostgreSQL for production.
* **Multi-Provider AI Foundation Layer:** Pluggable provider architecture with automatic JSON validation, retries, and token-aware batching:
  * **Google Gemini:** `gemini-2.0-flash` (via official `google-genai` SDK).
  * **Groq:** `openai/gpt-oss-120b` (optimized with 8-clause chunk targets to respect token-per-minute limits).
  * **OpenRouter:** `openrouter/free` models.
  * **Cerebras:** `llama-3.3-70b`.
* **Hybrid ML Routing & Fallback:** When local PyTorch checkpoints are present in `ml/`, a local domain detector and hierarchical classifier route clauses. In production Docker containers (where ML checkpoints are excluded to keep image size small), the pipeline routes through the cloud AI provider.

---

## 4. Repository Structure

```
LexEase/
├── backend/
│   ├── app/
│   │   ├── ai/                      # AI providers (Gemini, Groq, OpenRouter, Cerebras)
│   │   ├── api/routers/             # API endpoints (documents, clauses, rewrite, QA, adaptive)
│   │   ├── core/                    # App configuration, logging, and settings
│   │   ├── database/                # SQLAlchemy ORM models, session, and init_db
│   │   ├── prompts/                 # Prompt templates for summaries, analysis, and rewriting
│   │   ├── schemas/                 # Pydantic request/response validation schemas
│   │   ├── services/                # Business logic (preprocessing, rewrite, analysis, QA)
│   │   └── main.py                  # FastAPI entry point and CORS configuration
│   ├── tests/                       # Pytest test suite (157 unit and integration tests)
│   ├── Dockerfile                   # Production container definition (Python 3.11 slim)
│   ├── pyproject.toml               # Backend dependencies and project metadata
│   └── uv.lock                      # Pinned dependency lockfile
├── frontend/
│   ├── src/
│   │   ├── api/client.js            # Axios client, error formatting, and API methods
│   │   ├── components/              # UI components (Risk analysis, drafting, review, navbar)
│   │   ├── pages/                   # Application pages (Home, ClauseDrafting, AdaptiveLearning)
│   │   ├── documentFileStore.js     # In-memory session store for uploaded source PDF
│   │   ├── App.jsx                  # React Router definition
│   │   └── main.jsx                 # React root entry point
│   ├── package.json                 # Frontend dependencies and npm scripts
│   ├── vite.config.js               # Vite build configuration
│   └── tailwind.config.js           # Tailwind CSS theme and styling configuration
├── docker-compose.yml               # Multi-container setup (PostgreSQL, backend, frontend)
├── .env.example                     # Reference environment variable configuration
└── README.md                        # Project documentation
```

---

## 5. Prerequisites

Before running LexEase locally, ensure you have:
1. **Python:** Version `3.11.x` (`>=3.11, <3.12`).
2. **Package Manager:** [`uv`](https://github.com/astral-sh/uv) (recommended) or standard `pip`.
3. **Node.js:** Node.js `18.x` or newer and `npm`.
4. **AI Provider API Key:** An API key for at least one supported provider:
   * Google Gemini API Key, or
   * Groq Cloud API Key, or
   * OpenRouter API Key, or
   * Cerebras API Key.
5. **Database (Optional):** PostgreSQL 15+ (if not using default SQLite).

---

## 6. Local Setup & Execution

### 1. Clone Repository
```bash
git clone https://github.com/adityawankhede05/LexEase.git
cd LexEase
```

### 2. Backend Setup
Navigate to the `backend` directory:
```bash
cd backend
```

Create and activate a virtual environment, then install dependencies using `uv`:
```bash
# Using uv (recommended)
uv venv
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On macOS / Linux:
source .venv/bin/activate

uv sync
```
*(Alternatively, using standard pip: `python -m venv .venv` and `pip install -e .`)*

Create your `.env` file in `backend/.env` (or copy `.env.example` from the repository root):
```bash
cp ../.env.example .env
```
Edit `.env` to specify your `AI_PROVIDER` and corresponding API key (e.g. `GEMINI_API_KEY` or `GROQ_API_KEY`). For local development, leave `DATABASE_URL` unset to automatically use local SQLite (`sqlite:///./lexease.db`).

Run the backend development server:
```bash
uv run uvicorn app.main:app --reload --port 8000
```
> **Note on Database Initialization:** All database tables (`ClauseFeedback`, `ClauseFeedbackSubmission`, `DocumentRecord`, `ClauseAnalysisRecord`, `QAInteraction`) are automatically created on startup through the FastAPI application lifespan event (`init_db()`). No manual SQL scripts are required.

* Backend API: `http://localhost:8000`
* Interactive OpenAPI Docs: `http://localhost:8000/docs`
* Health Check: `http://localhost:8000/api/health`

### 3. Frontend Setup
Open a new terminal and navigate to the `frontend` directory:
```bash
cd frontend
npm install
```

Start the Vite development server:
```bash
npm run dev
```

* Frontend Application: `http://localhost:5173`

---

## 7. Environment Variables

The application is configured using environment variables. When running locally, place these in `backend/.env` (or in the project root):

| Variable | Component | Required? | Default | Description |
|---|---|---|---|---|
| `AI_PROVIDER` | Backend | Optional | `gemini` | Active AI provider (`gemini`, `groq`, `openrouter`, or `cerebras`). |
| `GEMINI_API_KEY` | Backend | Conditional | `None` | Required if `AI_PROVIDER=gemini`. |
| `GEMINI_MODEL` | Backend | Optional | `gemini-2.0-flash` | Gemini model name. |
| `GEMINI_TEMPERATURE` | Backend | Optional | `0.2` | Sampling temperature for Gemini. |
| `GEMINI_TIMEOUT` | Backend | Optional | `30.0` | HTTP timeout in seconds for Gemini calls. |
| `GROQ_API_KEY` | Backend | Conditional | `None` | Required if `AI_PROVIDER=groq`. |
| `GROQ_MODEL` | Backend | Optional | `openai/gpt-oss-120b` | Model identifier for Groq. |
| `GROQ_TIMEOUT` | Backend | Optional | `30.0` | Timeout in seconds for Groq requests. |
| `GROQ_REQUEST_DELAY` | Backend | Optional | `2.0` | Pacing delay between batch chunks. |
| `OPENROUTER_API_KEY` | Backend | Conditional | `None` | Required if `AI_PROVIDER=openrouter`. |
| `OPENROUTER_MODEL` | Backend | Optional | `openrouter/free` | OpenRouter model identifier. |
| `CEREBRAS_API_KEY` | Backend | Conditional | `None` | Required if `AI_PROVIDER=cerebras`. |
| `DATABASE_URL` | Backend | Optional | `sqlite:///./lexease.db` | SQLAlchemy connection string (`sqlite:///...` or `postgresql://...`). |
| `ALLOWED_ORIGINS` | Backend | Optional | `http://localhost:5173` | Comma-separated list of allowed CORS origins. |
| `PORT` | Backend | Optional | `8000` | Port for the Uvicorn web server. |
| `MAX_RETRIES` | Backend | Optional | `3` | Maximum retry attempts for failed AI provider calls. |
| `ADAPTIVE_LEARNING_ADMIN_TOKEN` | Backend | Optional | `lexease-review-secret` | Secret token required to access reviewer admin endpoints. |
| `VITE_BACKEND_URL` / `VITE_API_BASE_URL` | Frontend | Optional | `http://localhost:8000` | Base URL used by the Axios client to reach the backend. |

---

## 8. API Overview

All API endpoints are documented interactively via Swagger UI at `/docs`.

### System
* `GET /api/health` — Returns system status, timestamp, active AI provider, and database connectivity.

### Documents & Ingestion
* `POST /documents/upload` — Validates legal PDF, extracts text, removes noise, segments clauses, masks Indian PII, and returns indexed clause segments.

### Analysis & Understanding
* `POST /documents/summarize` — Generates executive summary and bullet points from document clauses.
* `POST /clauses/analyze` — Evaluates all clause segments, returning risk levels (`low`, `medium`, `high`), confidence score, plain-English legal analysis, and recommendations.
* `POST /documents/ask` — Answers grounded questions based on document clauses. Rejects unrelated queries if lexical relevance score is below `0.6`.

### Proposed Drafting & PDF Export
* `POST /clauses/rewrite` — Drafts customer-protective alternative wording for selected clauses after explicit consent.
* `POST /documents/proposed-draft` — Generates a downloadable `.docx` file containing approved clause revisions.
* `POST /documents/proposed-pdf` — Generates a downloadable PDF with in-place text replacement where space permits, or appends a formatted addendum page if text overflows.

### Adaptive Learning & Feedback
* `GET /adaptive-learning/dashboard` — Returns count of approved reference clauses and class distributions.
* `POST /adaptive-learning/feedback` — Submits a single user-consented correction proposal for a clause risk label.
* `POST /adaptive-learning/feedback/submissions` — Submits a batch of analyzed document clauses for human review.
* `GET /adaptive-learning/feedback/pending` — *(Reviewer Token Required)* Lists pending individual feedback items.
* `GET /adaptive-learning/feedback/submissions/pending` — *(Reviewer Token Required)* Lists pending document batches.
* `POST /adaptive-learning/feedback/{id}/review` — *(Reviewer Token Required)* Approves or rejects a suggested clause correction.
* `POST /adaptive-learning/feedback/submissions/{id}/review` — *(Reviewer Token Required)* Annotates or rejects a batch submission.

---

## 9. Testing & Validation

### Backend Tests
The backend test suite contains **157 unit and integration tests** covering document preprocessing, PII masking, AI provider integration, hybrid routing, summarization, grounded Q&A, proposed PDF generation, in-place reflow, and adaptive learning feedback:

```bash
cd backend
uv run pytest
```
*Result:* `157 passed in ~15-22s`

To run specific test modules:
```bash
# Test clause rewriting and in-place PDF reflow
uv run pytest tests/test_clause_rewrite.py

# Test adaptive learning feedback and reviewer authorization
uv run pytest tests/test_adaptive_learning.py

# Test grounded Q&A and lexical relevance gating
uv run pytest tests/test_qa.py
```

### Frontend Validation
To validate frontend code and compile the production build:
```bash
cd frontend
npm run build
```
*Result:* Compiles Vite bundle into `frontend/dist/` without compilation or bundling errors.

---

## 10. Deployment Overview

LexEase is architected for split-tier production deployment:
* **Frontend:** Hosted on static hosting providers such as Vercel, Netlify, or AWS S3/CloudFront. Configured with environment variable `VITE_BACKEND_URL` pointing to the public backend endpoint.
* **Backend:** Packaged via `backend/Dockerfile` as a Python 3.11 slim container and deployed to container hosting platforms (Render, Railway, Fly.io, or AWS ECS). Configured with `ALLOWED_ORIGINS` to permit requests from the production frontend domain.
* **Database:** Managed PostgreSQL (e.g. Supabase, Neon, or Render PostgreSQL). Tables are created on startup via `init_db()`.
* **AI Provider:** External HTTPS communication with Google Gemini or Groq APIs via API keys stored securely in environment variables.

---

## 11. Limitations & Responsible Use

* **Informational Purpose Only:** Risk ratings, plain-English explanations, and proposed wording drafts are generated by automated systems and large language models. They do not constitute formal legal counsel.
* **Layout Reflow Constraints:** In-place PDF editing relies on font matching and available blank space between clauses. If a suggested rewrite is substantially longer than the original clause, LexEase automatically delegates to a dedicated addendum page to guarantee that no existing text or signatures are obscured.
* **Offline Feedback Boundary:** User-submitted corrections and reviewer approvals do not trigger automated live model retraining. Verified examples are retained in the database solely for evaluation and offline supervised dataset improvement.

---

## 12. Contributing & License

### Contributing
1. Create a feature branch from `develop`:
   ```bash
   git checkout -b feature/your-feature-name develop
   ```
2. Ensure all backend tests (`uv run pytest`) and the frontend production build (`npm run build`) pass cleanly.
3. Submit a pull request targeting `develop`.

### License
This repository currently does not include an open-source license. All rights are reserved by the original project authors. Unauthorized commercial distribution is not permitted.
