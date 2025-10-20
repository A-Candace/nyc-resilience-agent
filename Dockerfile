FROM python:3.11-slim

# Extra system libs that help arcgis import cleanly
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libkrb5-3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip wheel && pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY .env .env

ENV STREAMLIT_SERVER_PORT=8505
ENV STREAMLIT_SERVER_HEADLESS=true
EXPOSE 8505

CMD ["streamlit", "run", "app.py"]
