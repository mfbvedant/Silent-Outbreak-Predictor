FROM python:3.13-slim

WORKDIR /app

# Copy all project folders
COPY ai_core/ ./ai_core/
COPY backend/ ./backend/
COPY frontend/ ./frontend/

# Install dependencies
RUN pip install --no-cache-dir -r ai_core/requirements.txt && \
    pip install --no-cache-dir -r backend/requirements.txt

# Expose port 8000
EXPOSE 8000

# Run uvicorn
CMD ["uvicorn", "backend.api:app", "--host", "0.0.0.0", "--port", "8000"]
