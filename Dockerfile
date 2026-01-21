FROM python:3.9-slim-buster

WORKDIR /app

# Install system dependencies if any
# RUN apt-get update && apt-get install -y ...

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --default-timeout=300 --no-cache-dir -r requirements.txt

COPY . .

# Ensure entrypoint is executable
RUN chmod +x scripts/entrypoint.sh

# Expose port
EXPOSE 8000

# Command to run the application
CMD ["./scripts/entrypoint.sh"]
