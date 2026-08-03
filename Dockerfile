FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .

RUN python -m pip install --upgrade pip \
    && pip install -r requirements.txt

COPY app ./app
COPY src ./src
COPY models/xgboost_tuned.pkl ./models/xgboost_tuned.pkl
COPY data/processed/daily_regional_demand.csv \
     ./data/processed/daily_regional_demand.csv

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["streamlit", "run", "app/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--browser.gatherUsageStats=false"]