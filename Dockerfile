FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements-app.txt .
RUN python -m pip install -r requirements-app.txt

COPY app/app.py ./app/app.py
COPY src/baseline.py src/capacity.py src/features.py src/prepare_data.py ./src/
COPY data/raw/workforce_demand.csv ./data/raw/workforce_demand.csv
RUN python -m src.prepare_data --input data/raw/workforce_demand.csv --output-dir data/processed \
    && useradd --create-home --uid 10001 planner

USER planner
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4).close()"]

CMD ["streamlit", "run", "app/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--browser.gatherUsageStats=false"]
