FROM python:3.11-slim

# System deps (keep what you had)
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

# Install Python deps
COPY requirements.txt .
RUN pip install --upgrade pip wheel && pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY app.py .

# IMPORTANT:
# Do NOT bake secrets into the container image.
# App Runner should provide these as environment variables.
# (Removed) COPY .env .env

# Streamlit settings
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# App Runner convention
EXPOSE 8080

# Bind to 0.0.0.0 and use $PORT when provided by the platform
CMD ["bash", "-lc", "streamlit run app.py --server.address 0.0.0.0 --server.port ${PORT:-8080} --server.headless true"]
