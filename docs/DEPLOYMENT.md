# LexEase Deployment Guide

## Overview

This guide outlines deployment considerations for the LexEase backend and Multi-Provider AI Foundation Layer.

---

## Sprint 8 Notes (Multi-Provider AI Architecture)

> [!IMPORTANT]
> **Provider Switching via Configuration**
> LexEase now supports three AI providers: `groq`, `openrouter`, and `gemini`.
> The active provider is determined exclusively by the `AI_PROVIDER` environment variable.
> If one provider encounters rate limits or quota exhaustion (e.g. Gemini 429 quota exhaustion), switch to `AI_PROVIDER=groq` or `AI_PROVIDER=openrouter` without altering application code or restarting anything other than loading updated environment variables.

---

## Environment Configuration

Ensure the following environment variables are configured in your deployment environment (`.env` or container orchestrator):

```env
# Server Configuration
PORT=8000
ALLOWED_ORIGINS=http://localhost:5173,https://yourdomain.com

# Active AI Provider: "groq", "openrouter", or "gemini"
AI_PROVIDER=groq

# Groq Provider Settings
GROQ_API_KEY=your_production_groq_api_key
GROQ_MODEL=openai/gpt-oss-120b
GROQ_TIMEOUT=30.0

# OpenRouter Provider Settings
OPENROUTER_API_KEY=your_production_openrouter_api_key
OPENROUTER_MODEL=openrouter/free
OPENROUTER_TIMEOUT=30.0

# Google Gemini Provider Settings
GEMINI_API_KEY=your_production_gemini_api_key
GEMINI_MODEL=gemini-2.0-flash
GEMINI_TEMPERATURE=0.2
GEMINI_TIMEOUT=30.0

# Global AI Settings
MAX_RETRIES=3
```

---

## Docker Deployment

To deploy via Docker Compose:

```bash
docker compose up --build -d
```

The containerized FastAPI application automatically reads the `.env` configuration and instantiates the selected provider via `ProviderFactory`.
