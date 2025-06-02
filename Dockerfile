FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV and OCR
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    libgtk-3-0 \
    wget \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p uploads models

# Expose port (optional but good practice)
EXPOSE 8000

# Start the FastAPI app with uvicorn using PORT environment variable
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]


