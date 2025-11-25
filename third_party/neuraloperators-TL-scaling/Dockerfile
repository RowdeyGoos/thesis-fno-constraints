# Dockerfile for neuraloperators-TL-scaling third party project
# Separate from main thesis project to avoid dependency conflicts

FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# Copy project files
COPY . /workspace/

# Set Python path
ENV PYTHONPATH=/workspace:$PYTHONPATH
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["/bin/bash"]
