# Docker Configuration Guide

This guide provides detailed instructions for building, configuring, and deploying VIGA using Docker.

## Table of Contents

- [For Users: Using Pre-built Images](#for-users-using-pre-built-images)
- [For Developers: Building and Publishing](#for-developers-building-and-publishing)
- [Dockerfile Configuration](#dockerfile-configuration)
- [Advanced Usage](#advanced-usage)

## For Users: Using Pre-built Images

### 1. Prerequisites

Ensure you have the following installed:

- **Docker** (version 20.10 or higher)
- **NVIDIA GPU** with CUDA 12.8+ support
- **NVIDIA Container Toolkit** for GPU access

### 2. Install Docker

```bash
# Install Docker using the official script
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add your user to the docker group (optional, to run without sudo)
sudo usermod -aG docker $USER
newgrp docker
```

### 3. Install NVIDIA Container Toolkit

```bash
# Add NVIDIA package repositories
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install nvidia-container-toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Restart Docker to apply changes
sudo systemctl restart docker
```

### 4. Verify GPU Access

```bash
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

You should see your GPU information displayed.

### 5. Pull VIGA Docker Image

```bash
docker pull <your-dockerhub-username>/viga:latest
```

### 6. Run the Container

```bash
# Basic usage
docker run --gpus all -it \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/output:/workspace/output \
  -v $(pwd)/utils:/workspace/utils \
  <your-dockerhub-username>/viga:latest

# With custom configuration
docker run --gpus all -it \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/output:/workspace/output \
  -v $(pwd)/utils:/workspace/utils \
  -v $(pwd)/custom_config:/workspace/config \
  -e OPENAI_API_KEY="your-key" \
  <your-dockerhub-username>/viga:latest
```

## For Developers: Building and Publishing

### 1. Create the Dockerfile

Create a `Dockerfile` in the project root:

```dockerfile
# Use NVIDIA CUDA base image with Ubuntu 22.04
FROM nvidia/cuda:12.8.0-cudnn9-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.11 \
    python3-pip \
    git \
    wget \
    curl \
    vim \
    libreoffice \
    unoconv \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Google Chrome for Design2Code
RUN wget -q https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb && \
    apt-get update && \
    apt-get install -y ./google-chrome-stable_current_amd64.deb && \
    rm google-chrome-stable_current_amd64.deb && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy project files
COPY . /workspace/

# Install Python packages for agent environment
RUN python3.10 -m pip install --upgrade pip && \
    python3.10 -m pip install -r requirements/requirement_agent.txt

# Install Python packages for blender environment
RUN python3.11 -m pip install --upgrade pip && \
    python3.11 -m pip install -r requirements/requirement_blender.txt

# Install Python packages for pptx environment
RUN python3.10 -m pip install -r requirements/requirement_pptx.txt

# Install Infinigen for BlenderGym (minimal install)
RUN cd utils && \
    git clone https://github.com/richard-guyunqi/infinigen.git infinigen_minimal && \
    cd infinigen_minimal && \
    INFINIGEN_MINIMAL_INSTALL=True bash scripts/install/interactive_blender.sh

# Install Infinigen for other 3D modes (full install)
RUN cd utils && \
    git clone https://github.com/princeton-vl/infinigen.git && \
    cd infinigen && \
    bash scripts/install/interactive_blender.sh

# Create output directories
RUN mkdir -p /workspace/output /workspace/logs

# Set default command
CMD ["/bin/bash"]
```

### 2. Create .dockerignore

Create a `.dockerignore` file to exclude unnecessary files:

```
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
*.so
*.egg
*.egg-info/
dist/
build/
.git/
.gitignore
.vscode/
.idea/
*.log
output/
logs/
data/*/output/
*.blend1
*.blend2
.DS_Store
```

### 3. Build the Docker Image

```bash
# Build for local testing
docker build -t viga:latest .

# Build with build arguments (optional)
docker build \
  --build-arg CUDA_VERSION=12.8.0 \
  --build-arg PYTHON_VERSION=3.10 \
  -t viga:latest .
```

### 4. Test the Image Locally

```bash
# Run interactive shell
docker run --gpus all -it viga:latest /bin/bash

# Test CUDA availability
docker run --gpus all viga:latest python3 -c "import torch; print(torch.cuda.is_available())"

# Run a quick test
docker run --gpus all -it \
  -v $(pwd)/data:/workspace/data \
  viga:latest \
  python main.py --mode blendergym --help
```

### 5. Tag and Push to Docker Hub

```bash
# Login to Docker Hub
docker login

# Tag the image
docker tag viga:latest <your-dockerhub-username>/viga:latest
docker tag viga:latest <your-dockerhub-username>/viga:v1.0.0

# Push to Docker Hub
docker push <your-dockerhub-username>/viga:latest
docker push <your-dockerhub-username>/viga:v1.0.0
```

### 6. Build Multi-Architecture Images (Optional)

If you want to support multiple architectures (AMD64, ARM64):

```bash
# Create a new builder instance
docker buildx create --name multiarch-builder --use

# Build and push for multiple platforms
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t <your-dockerhub-username>/viga:latest \
  --push .
```

## Dockerfile Configuration

### Environment Variables

You can customize the container behavior using environment variables:

```bash
docker run --gpus all -it \
  -e OPENAI_API_KEY="your-openai-key" \
  -e CLAUDE_API_KEY="your-claude-key" \
  -e GEMINI_API_KEY="your-gemini-key" \
  -e MESHY_API_KEY="your-meshy-key" \
  viga:latest
```

### Volume Mounts

Mount local directories to persist data:

- `/workspace/data` - Input data and datasets
- `/workspace/output` - Generated outputs
- `/workspace/utils` - Configuration files
- `/workspace/logs` - Log files

```bash
docker run --gpus all -it \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/output:/workspace/output \
  -v $(pwd)/utils:/workspace/utils \
  -v $(pwd)/logs:/workspace/logs \
  viga:latest
```

## Advanced Usage

### Running as a Background Service

```bash
# Run in detached mode
docker run -d --gpus all \
  --name viga-service \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/output:/workspace/output \
  viga:latest \
  python main.py --mode static_scene --model gpt-4o --max-rounds 100

# Check logs
docker logs -f viga-service

# Stop the container
docker stop viga-service

# Remove the container
docker rm viga-service
```

### Using Docker Compose

Create a `docker-compose.yml` file:

```yaml
version: '3.8'

services:
  viga:
    image: <your-dockerhub-username>/viga:latest
    container_name: viga
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - CLAUDE_API_KEY=${CLAUDE_API_KEY}
      - GEMINI_API_KEY=${GEMINI_API_KEY}
      - MESHY_API_KEY=${MESHY_API_KEY}
    volumes:
      - ./data:/workspace/data
      - ./output:/workspace/output
      - ./utils:/workspace/utils
      - ./logs:/workspace/logs
    stdin_open: true
    tty: true
    command: /bin/bash
```

Run with:

```bash
docker-compose up -d
docker-compose exec viga /bin/bash
```

### Optimizing Image Size

To reduce the Docker image size:

1. Use multi-stage builds
2. Clean up apt cache and pip cache
3. Remove unnecessary files after installation

Example optimization:

```dockerfile
# Clean up in the same layer
RUN apt-get update && apt-get install -y package \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir package
```

### GPU Memory Management

Specify GPU devices:

```bash
# Use specific GPU
docker run --gpus device=0 -it viga:latest

# Use multiple GPUs
docker run --gpus '"device=0,1"' -it viga:latest

# Limit GPU memory
docker run --gpus all --memory=16g -it viga:latest
```

## Troubleshooting

### GPU Not Detected

```bash
# Check NVIDIA driver
nvidia-smi

# Check Docker GPU runtime
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi

# Restart Docker daemon
sudo systemctl restart docker
```

### Permission Denied

```bash
# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker
```

### Out of Memory

```bash
# Increase Docker memory limit
docker run --memory=32g --gpus all -it viga:latest
```

### Network Issues During Build

```bash
# Use build cache and retry
docker build --network=host -t viga:latest .
```

## Best Practices

1. **Version Control**: Always tag images with version numbers
2. **Security**: Don't include API keys in the Dockerfile; use environment variables or mounted config files
3. **Layer Caching**: Order Dockerfile commands from least to most frequently changing
4. **Size Optimization**: Use `.dockerignore` to exclude unnecessary files
5. **Testing**: Test images locally before pushing to registry
6. **Documentation**: Keep this guide updated with any changes to the Dockerfile

## Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
