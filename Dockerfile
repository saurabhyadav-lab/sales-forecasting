# Step 22: build & run
#   docker build -t sales-forecast .
#   docker run -p 8000:8000 sales-forecast
# Run `python run_pipeline.py` first so models/ and data/processed/ exist.
FROM python:3.11-slim
WORKDIR /app
COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt
COPY src/ src/
COPY app/ app/
COPY models/ models/
COPY data/processed/ data/processed/
COPY reports/ reports/
EXPOSE 8000
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
