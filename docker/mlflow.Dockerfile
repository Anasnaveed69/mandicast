# syntax=docker/dockerfile:1.7
# MLflow tracking server + model registry.
# Backend store: PostgreSQL. Artifacts: a local volume by default, or S3 when
# MLFLOW_ARTIFACTS_DESTINATION=s3://<bucket>/mlflow (credentials from the env / instance role).
FROM python:3.11-slim
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONUNBUFFERED=1
RUN pip install "mlflow>=3.0,<4" psycopg2-binary boto3 \
 && useradd --create-home --uid 1000 mlflow \
 && mkdir -p /mlflow/artifacts && chown -R mlflow:mlflow /mlflow
USER mlflow
EXPOSE 5000
ENV MLFLOW_BACKEND_STORE_URI=sqlite:////mlflow/mlflow.db \
    MLFLOW_ARTIFACTS_DESTINATION=/mlflow/artifacts \
    MLFLOW_SERVER_ALLOWED_HOSTS="mlflow,mlflow:5000,localhost,localhost:*,127.0.0.1,127.0.0.1:*"
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/health', timeout=4)"
# shell form so the env vars are expanded at start-up
CMD mlflow server --host 0.0.0.0 --port 5000 \
    --backend-store-uri "$MLFLOW_BACKEND_STORE_URI" \
    --artifacts-destination "$MLFLOW_ARTIFACTS_DESTINATION" \
    --serve-artifacts  --workers 1
