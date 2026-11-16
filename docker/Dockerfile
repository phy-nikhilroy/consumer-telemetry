# API image: FastAPI service with the extraction pipeline and the committed models.
#
#   docker build -t consumer-telemetry -f docker/Dockerfile .
#   docker run -p 8000:8000 consumer-telemetry
#   curl http://localhost:8000/health
#
# The serving warehouse is built into the image (simulated with the seeded simulator,
# about 20 seconds), so a container answers requests as soon as it starts and a restart
# resets the demo to its initial state. Mount a volume on /app/data/processed to keep
# ingested events across restarts instead.
#
# Python 3.12 rather than the 3.11 in the original specification, to match the version
# the project is developed and tested on.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/.cache/huggingface \
    PORT=8000

# libgomp1 is the OpenMP runtime LightGBM links against; curl serves the health check.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# A non-root user with UID 1000, which is also what Hugging Face Spaces runs as.
RUN useradd --create-home --uid 1000 app
WORKDIR /app

COPY docker/requirements-api.txt /tmp/requirements-api.txt
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1 \
    && pip install -r /tmp/requirements-api.txt \
    && python -m spacy download en_core_web_sm

COPY --chown=app:app src/ src/
RUN mkdir -p data/processed data/features && chown -R app:app /app

USER app

# Download the sentence encoder and build the serving warehouse at build time, so
# neither happens on the first request.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" \
    && python src/api/warehouse.py

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT}/health" || exit 1

CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT}"]
