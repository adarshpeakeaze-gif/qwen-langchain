# Use Python 3.11 slim image
FROM python:3.11-slim

WORKDIR /app

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files (modular structure)
COPY app.py .
COPY config.py .
COPY templates/ templates/
COPY prompts/ prompts/
COPY extraction/ extraction/

# Copy .env file (contains API keys - for dev only, use env vars in production)
COPY .env .

# Default environment variables
ENV HOST=0.0.0.0
ENV PORT=5051
ENV DEBUG=false

EXPOSE 5051

CMD ["python", "app.py"]
