# Multi-stage build.
#   builder: installs dependencies into /opt/venv (pip caches and build tools stay here)
#   runtime: slim base + that venv + app code, non-root, healthcheck
#
# Core image (deterministic + LLM modes, no torch):
#   docker build -t grounded-support-assistant .
# With local models (BERT QA, sentence-transformers, cross-encoder):
#   docker build --build-arg EXTRAS=requirements-local-models.txt -t grounded-support-assistant:local .

ARG PYTHON_VERSION=3.12

FROM python:${PYTHON_VERSION}-slim AS builder
ARG EXTRAS=requirements.txt
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
WORKDIR /build
COPY requirements.txt requirements-local-models.txt ./
# Runtime needs neither pytest nor httpx (test client), so both are filtered out.
RUN grep -vE '^(pytest|httpx)' requirements.txt > requirements.runtime.txt \
 && sed -i 's/^-r requirements.txt$/-r requirements.runtime.txt/' requirements-local-models.txt \
 && pip install -r requirements.runtime.txt \
 && if [ "$EXTRAS" != "requirements.txt" ]; then pip install -r "$EXTRAS"; fi \
 && find /opt/venv -type d -name "__pycache__" -prune -exec rm -rf {} + \
 && find /opt/venv -type d -name "tests" -path "*site-packages/*" -prune -exec rm -rf {} +

FROM python:${PYTHON_VERSION}-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PATH="/opt/venv/bin:$PATH"
RUN useradd --create-home --uid 10001 appuser
COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
COPY --chown=appuser app ./app
COPY --chown=appuser kb ./kb
COPY --chown=appuser eval ./eval
COPY --chown=appuser questions.json main.py evaluate.py validate.py ./
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

# Keys are passed at run time: docker run --env-file .env -p 8000:8000 grounded-support-assistant
CMD ["python", "main.py", "--serve", "--host", "0.0.0.0", "--port", "8000"]
