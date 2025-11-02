FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY ml/requirements.txt ./ml/

# Install Python dependencies
RUN pip install --no-cache-dir -r ml/requirements.txt

# Copy ML code
COPY ml/ ./ml/

# Create models directory
RUN mkdir -p ml/models ml/datasets ml/checkpoints

# Expose port
EXPOSE 8000

# Start FastAPI server
CMD ["uvicorn", "ml.server:app", "--host", "0.0.0.0", "--port", "8000"]
