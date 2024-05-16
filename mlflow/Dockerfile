# Use the official python slim image as base
FROM python:3.11-slim

# Set metadata
LABEL authors="Siverty"
LABEL owners="Research Center Data Intelligence - Zuyd University of Applied Science, Justin Schots"
LABEL nl.siverty.yolo_cl_tracking.service="mlflow server"

# Set environment variables
ENV MLFLOW_HOME /mlflow
ENV MLFLOW_VERSION 2.12.2
ENV PORT 8282
ENV BACKEND_STORE_URI sqlite:////mlflow/mlflow.db
ENV DEFAULT_ARTIFACT_ROOT /mlflow/mlflow-artifacts

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install mlflow
RUN pip install mlflow==$MLFLOW_VERSION

# Expose port
EXPOSE $PORT

# Set mlflow server as entrypoint
ENTRYPOINT ["mlflow", "server", "--host", "0.0.0.0", "--port", "8282", "--backend-store-uri", "sqlite:///mlflow/mlflow.db", "--default-artifact-root", "/mlflow/mlflow-artifacts"]
