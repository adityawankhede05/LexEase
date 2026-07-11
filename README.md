# LexEase - Legal Document Simplification & Clause Risk Analysis Using AI

LexEase is a web application designed to help ordinary Indian citizens understand complex legal documents by providing plain-language summaries, clause-by-clause risk classification, interactive grounded Q&A, and translations to Hindi/Marathi.

## Project Structure

*   `/backend` - FastAPI backend application using Python 3.11, `uv` package manager, and LangChain + Gemini.
*   `/frontend` - React SPA frontend built with Vite, TailwindCSS, React Router, and Axios.
*   `/docs` - Project documentation, planning, and guidelines.
*   `docker-compose.yml` - Multi-container setup for local development.

## Setup Instructions

### Environment Configuration

Copy the example environment file and configure variables:
```bash
cp .env.example .env
```

Set your Google Gemini API key:
```env
GEMINI_API_KEY=your_actual_key_here
```

### Running with Docker

Run the entire application stack:
```bash
docker compose up --build
```
*   Frontend will be available at: http://localhost:5173
*   Backend will be available at: http://localhost:8000
*   Database (PostgreSQL) is exposed on port: 5432

### Local Development Setup

#### Backend Setup

Prerequisites: Python 3.11, [uv package manager](https://github.com/astral-sh/uv).

1. Change directory to backend:
   ```bash
   cd backend
   ```
2. Initialize virtual environment and install packages:
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   uv pip install -r requirements.txt
   ```
3. Run the development server:
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```

#### Frontend Setup

Prerequisites: Node.js (v18+).

1. Change directory to frontend:
   ```bash
   cd frontend
   ```
2. Install packages:
   ```bash
   npm install
   ```
3. Run the Vite development server:
   ```bash
   npm run dev
   ```
