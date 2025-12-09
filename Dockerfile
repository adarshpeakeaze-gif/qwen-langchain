# Use Python 3.11 slim image
FROM python:3.11-slim

WORKDIR /app

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files (modular structure)
COPY app.py .
COPY templates/ templates/
COPY prompts/ prompts/
COPY extraction/ extraction/
COPY .env .

ENV FLASK_DEBUG=true
ENV FLASK_PORT=5051

EXPOSE 5051

CMD ["python", "app.py"]
