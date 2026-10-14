FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080

WORKDIR /app

# OpenCV / matplotlib system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# Writable dirs for generated artifacts (Supabase Storage is preferred in prod)
RUN mkdir -p models static/uploads static/heatmaps static/reports static/simulations

EXPOSE 8080

CMD gunicorn --bind 0.0.0.0:${PORT} --workers 1 --threads 2 --timeout 120 app:app
