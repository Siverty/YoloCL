# Use the official mlflow docker image as base
FROM python:3.11-slim

# Set metadata
LABEL authors="Siverty"
LABEL owners="Research Center Data Intelligence - Zuyd University of Applied Science, Justin Schots"
LABEL nl.siverty.yolo_cl_tracking.service="mlflow server"

# Set environment variables
ENV MLFLOW_HOME /mlflow
ENV MLFLOW_VERSION 2.0.1
ENV PORT 5000
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
ENTRYPOINT ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", "--backend-store-uri", "$BACKEND_STORE_URI", "--default-artifact-root", "$DEFAULT_ARTIFACT_ROOT"]
