# syntax=docker/dockerfile:1.7
# One Dockerfile, two images that share the same Python environment:
#   --target api       FastAPI inference server  (ghcr.io/<owner>/mandicast-api)
#   --target pipeline  DVC pipeline / scheduler   (ghcr.io/<owner>/mandicast-pipeline)
# Sharing the dependency layers means the models the pipeline trains are pickled
# with exactly the library versions the API unpickles them with.

ARG PYTHON_VERSION=3.11

# ---------------------------------------------------------------- dependencies
FROM python:${PYTHON_VERSION}-slim AS deps
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
# CPU-only torch (~200 MB instead of ~2 GB with CUDA)
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements-serve.txt /tmp/
RUN pip install -r /tmp/requirements-serve.txt

FROM deps AS deps-pipeline
COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt

# ---------------------------------------------------------------- shared runtime base
FROM python:${PYTHON_VERSION}-slim AS runtime
# libgomp1: OpenMP runtime needed by xgboost and torch
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --create-home --uid 1000 app
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf \
    MPLCONFIGDIR=/tmp/matplotlib \
    TZ=Asia/Karachi

# ---------------------------------------------------------------- api
FROM runtime AS api
COPY --from=deps /opt/venv /opt/venv
# Bake the Chronos-Bolt weights (~35 MB) into the image so the onion-h7 champion
# doesn't need the Hugging Face hub at request time. Build with
# --build-arg PRELOAD_CHRONOS=0 to skip (weights are then downloaded on first use).
ARG PRELOAD_CHRONOS=1
ARG CHRONOS_MODEL=amazon/chronos-bolt-tiny
RUN mkdir -p $HF_HOME \
 && if [ "$PRELOAD_CHRONOS" = "1" ]; then \
      python -c "from chronos import BaseChronosPipeline as P; P.from_pretrained('$CHRONOS_MODEL', device_map='cpu')"; \
    fi \
 && chown -R app:app $HF_HOME
ENV HF_HUB_OFFLINE=${PRELOAD_CHRONOS}

WORKDIR /app
COPY --chown=app:app params.yaml pyproject.toml requirements.txt ./
COPY --chown=app:app mandicast ./mandicast
RUN mkdir -p data/processed models/serving && chown -R app:app /app
ARG GIT_COMMIT=unknown
ENV GIT_COMMIT=${GIT_COMMIT}

USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"
CMD ["uvicorn", "mandicast.api.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--proxy-headers", "--forwarded-allow-ips", "*", "--timeout-keep-alive", "30"]

# ---------------------------------------------------------------- pipeline / scheduler
FROM runtime AS pipeline
RUN apt-get update \
 && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/* \
 && git config --system --add safe.directory '*' \
 && mkdir -p $HF_HOME && chown app:app $HF_HOME
COPY --from=deps-pipeline /opt/venv /opt/venv
# The repo (code, dvc.yaml, params.yaml, .git) is bind-mounted here so dvc can
# track it; the image only provides the environment.
WORKDIR /workspace
ARG GIT_COMMIT=unknown
ENV GIT_COMMIT=${GIT_COMMIT}
USER app
CMD ["python", "-m", "mandicast.jobs", "schedule"]
