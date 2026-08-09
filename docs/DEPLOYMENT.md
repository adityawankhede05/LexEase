# LexEase Deployment Guide

## Overview

This guide outlines deployment considerations for the LexEase backend and AI summarization service.

---

## Sprint 6 Notes

> [!NOTE]
> **No new environment variables are required for Sprint 6.**
> Clause risk analysis (`POST /clauses/analyze`) reuses the existing Gemini configuration (`GEMINI_API_KEY`, `GEMINI_MODEL`, `GEMINI_TEMPERATURE`, `GEMINI_TIMEOUT`, `MAX_RETRIES`) and sends a single batched AI request per endpoint call.

---

## Environment Configuration

Ensure the following environment variables are set in your production environment or container environment (`docker-compose.yml` / `.env`):

```env
# Server Config
PORT=8000
ALLOWED_ORIGINS=http://localhost:5173,https://yourdomain.com

# Gemini AI Provider Config
GEMINI_API_KEY=your_production_google_gemini_api_key
GEMINI_MODEL=gemini-2.0-flash
GEMINI_TEMPERATURE=0.2
GEMINI_TIMEOUT=30.0
MAX_RETRIES=3
```

---

## Docker Deployment

To deploy via Docker:

```bash
docker compose up --build -d
```

The containerized FastAPI application automatically loads configuration and exposes endpoints on the configured port.
