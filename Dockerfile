FROM python:3.11-bookworm

WORKDIR /app

# Upgrade system packages and SQLite
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    sqlite3 \
    libsqlite3-dev \
    git \
    curl \
 && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]

