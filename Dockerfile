# Image de production de l'API (web et webhook WhatsApp).
# Prérequis avant le build : avoir lancé l'ingestion en local, pour que
# data/baremes/*.json existe. Les PDF et le cache OCR ne sont pas dans l'image.

FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.13 /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

# Dépendances d'abord, pour profiter du cache Docker.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY data ./data
RUN uv sync --frozen --no-dev

# Base SQLite (conversations, suivi d'usage) sur un volume persistant.
ENV DATA_DIR=/app/data \
    APP_DB_PATH=/var/lib/rag-fisc/app.db \
    PORT=8000
RUN mkdir -p /var/lib/rag-fisc
VOLUME ["/var/lib/rag-fisc"]

EXPOSE 8000
# Un seul processus : SQLite et les caches en mémoire le supposent.
CMD ["sh", "-c", ".venv/bin/uvicorn rag_fisc_v2.main:app --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips='*'"]
