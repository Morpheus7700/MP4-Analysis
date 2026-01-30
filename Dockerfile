# Use Python 3.11 slim for a smaller footprint
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_PORT=8080
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
# Set Transformers cache to a permanent location in the image
ENV HF_HOME=/app/models/huggingface
ENV TORCH_HOME=/app/models/torch

# Install system dependencies for OpenCV, FFmpeg, and Python build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    libglib2.0-0 \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1. Install dependencies first (for caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 2. Pre-download AI Models into the image
# This prevents the container from timing out on the first request
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" && \
    python -c "import whisper; whisper.load_model('tiny')" && \
    python -c "from transformers import pipeline; pipeline('image-to-text', model='Salesforce/blip-image-captioning-base'); pipeline('text-generation', model='microsoft/phi-1_5'); pipeline('image-classification', model='dima806/facial_emotions_image_detection')"

# 3. Copy the application code
COPY . .

# Expose port
EXPOSE 8080

# Production Streamlit command
CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT:-8080} --server.address=0.0.0.0 --server.enableCORS=false --server.enableXsrfProtection=false"]
