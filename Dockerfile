# --- Build Stage ---
FROM python:3.10-slim as builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-optimized PyTorch first
# This is the single biggest size reduction (~2GB saved)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Final Stage ---
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy only the installed packages from the builder stage
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy the application (obeying .dockerignore)
COPY . .

# Expose port
EXPOSE 5000

# Run from the application directory
WORKDIR /app/memory-forecaster
CMD ["python", "serve.py"]
