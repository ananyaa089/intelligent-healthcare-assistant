FROM python:3.11-slim

WORKDIR /app

# System deps needed by faiss / torch wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
COPY index_data/ ./index_data/

EXPOSE 10000

# Render injects its own $PORT at runtime; shell form lets us expand it.
# Falls back to 10000 for local `docker run` testing.
CMD uvicorn backend.app:app --host 0.0.0.0 --port ${PORT:-10000}
