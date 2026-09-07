# Use official Python 3.11 slim image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install uv (psycopg2-binary bundles its own libpq — no extra system libs needed)
RUN pip install --no-cache-dir uv

# Copy lockfile and project manifest first for optimal layer caching
COPY pyproject.toml uv.lock ./

# Install all pinned dependencies from the lockfile (reproducible)
RUN uv pip install --system --no-cache -r pyproject.toml

# Copy application code
COPY app/ ./app/

# Expose port and start FastAPI
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
