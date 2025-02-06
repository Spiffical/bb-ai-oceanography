#!/bin/bash

# Check if nvidia-smi is available (GPU present)
if command -v nvidia-smi &> /dev/null; then
    echo "GPU detected, building and running with GPU support..."
    # Build with GPU support
    docker build --build-arg USE_GPU=1 -f docker/Dockerfile.api -t paperai-api .
    # Run with GPU support
    docker run --rm \
        --gpus all \
        -v "$(pwd):/work" \
        --env-file .env \
        paperai-api -m paperai.report "$@"
else
    echo "No GPU detected, running with CPU only..."
    # Build without GPU support
    docker build --build-arg USE_GPU=0 -f docker/Dockerfile.api -t paperai-api .
    # Run without GPU support
    docker run --rm \
        -v "$(pwd):/work" \
        --env-file .env \
        paperai-api -m paperai.report "$@"
fi 