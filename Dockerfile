# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source files
COPY preprocess.py .
COPY handler.py .
COPY model_artifacts/ ./model_artifacts/

# Model artifacts are expected at /app/model_artifacts/lda_model.pkl
# Either COPY them in at build time:
#   COPY model_artifacts/ ./model_artifacts/
# Or mount a RunPod network volume at /app/model_artifacts at runtime.

ENV MODEL_PATH=/app/model_artifacts/lda_model.pkl
ENV PYTHONUNBUFFERED=1

CMD ["python", "-u", "handler.py"]
