FROM python:3.11-slim
WORKDIR /app

# Install FFmpeg for video conversion
RUN apt-get update && \
    apt-get install -y ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code
COPY . .

# Create directories for temporary file storage
RUN mkdir -p /app/tmp/uploads /app/tmp/processed && \
    chmod -R 777 /app/tmp

# Expose the port
EXPOSE 8000

# Run the server
CMD ["uvicorn", "detection:app", "--host", "0.0.0.0", "--port", "8000"]