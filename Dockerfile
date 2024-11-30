# Use llama-cpp-python's official image as base
FROM ghcr.io/abetlen/llama-cpp-python:v0.3.2

# Copy .env file
COPY .env /app/.env

# Load environment variables from .env file
ENV $(cat /app/.env | xargs)

LABEL maintainer="NeuML"
LABEL repository="paperai"

# Argument for ENTRYPOINT
ARG START=/bin/bash
ENV START=${START}

# Install additional required packages
RUN apt-get update && \
    apt-get -y --no-install-recommends install \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install ML-related Python packages
RUN pip install --no-cache-dir \
    torch \
    bitsandbytes \
    accelerate \
    transformers

# Copy your local project files
WORKDIR /app
COPY . .

# Install paperetl first
RUN cd paperetl && pip install -e .

# Install paperai with development mode
RUN cd paperai && pip install -e .

# Create paperetl directories
RUN mkdir -p paperetl/data paperetl/report

# Set working directory for mounted volume
WORKDIR /work

ENTRYPOINT ["sh", "-c", "$START"]