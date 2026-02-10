FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

# Exponer puerto de FastAPI
EXPOSE 8000

# Ejecutar API
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
