# Use Python 3.11 slim image
FROM python:3.11-slim

WORKDIR /app

# Install OpenCV dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY app.py .
COPY .env .
COPY templates/ templates/
COPY prompts/ prompts/
COPY datalabs/ datalabs/

# Create uploads folder
RUN mkdir -p uploads

ENV FLASK_DEBUG=true
ENV FLASK_PORT=5050

EXPOSE 5050

CMD ["python", "app.py"]
