FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

# RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/
COPY models/ models/

# ENV MODEL_PATH=/app/model/model.pkl

EXPOSE 8000

# Command to run the FastAPI app with Uvicorn
CMD ["fastapi", "run", "app/main.py", "--port", "8000"]