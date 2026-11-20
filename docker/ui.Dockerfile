# syntax=docker/dockerfile:1.7
# Streamlit dashboard. Small image: it only calls the API over HTTP.
FROM python:3.11-slim
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONUNBUFFERED=1
RUN useradd --create-home --uid 1000 app
WORKDIR /app
COPY ui/requirements.txt .
RUN pip install -r requirements.txt
COPY --chown=app:app ui/ .
USER app
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4)"
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]
