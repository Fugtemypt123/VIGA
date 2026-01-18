# VIGA Docker Image
# Base image with CUDA 12.8 support for Ubuntu 22.04
FROM nvidia/cuda:12.8.0-cudnn9-devel-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Python versions
    python3.10 \
    python3.10-dev \
    python3.11 \
    python3.11-dev \
    python3-pip \
    # Build tools
    build-essential \
    cmake \
    git \
    wget \
    curl \
    vim \
    # System utilities
    software-properties-common \
    # Graphics libraries
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglew-dev \
    libglfw3 \
    libglfw3-dev \
    # Office tools for AutoPresent
    libreoffice \
    unoconv \
    # Cleanup
    && rm -rf /var/lib/apt/lists/*

# Install Google Chrome for Design2Code
RUN wget -q https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb && \
    apt-get update && \
    apt-get install -y ./google-chrome-stable_current_amd64.deb && \
    rm google-chrome-stable_current_amd64.deb && \
    rm -rf /var/lib/apt/lists/*

# Set up Python 3.10 as default for some environments
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Upgrade pip for both Python versions
RUN python3.10 -m pip install --upgrade pip setuptools wheel
RUN python3.11 -m pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /workspace

# Copy project files
COPY requirements/ /workspace/requirements/
COPY agents/ /workspace/agents/
COPY prompts/ /workspace/prompts/
COPY utils/ /workspace/utils/
COPY tools/ /workspace/tools/
COPY models/ /workspace/models/
COPY runners/ /workspace/runners/
COPY main.py /workspace/
COPY README.md /workspace/

# Install Python packages for agent environment (Python 3.10)
RUN python3.10 -m pip install --no-cache-dir -r requirements/requirement_agent.txt

# Install Python packages for blender environment (Python 3.11)
RUN python3.11 -m pip install --no-cache-dir -r requirements/requirement_blender.txt

# Install Python packages for pptx environment (Python 3.10)
RUN python3.10 -m pip install --no-cache-dir -r requirements/requirement_pptx.txt

# Install Infinigen for BlenderGym (minimal install)
RUN cd utils && \
    git clone https://github.com/richard-guyunqi/infinigen.git infinigen_minimal && \
    cd infinigen_minimal && \
    INFINIGEN_MINIMAL_INSTALL=True bash scripts/install/interactive_blender.sh || \
    echo "Warning: Infinigen minimal installation failed, continuing..."

# Install Infinigen for other 3D modes (full install)
RUN cd utils && \
    git clone https://github.com/princeton-vl/infinigen.git && \
    cd infinigen && \
    bash scripts/install/interactive_blender.sh || \
    echo "Warning: Infinigen full installation failed, continuing..."

# Create necessary directories
RUN mkdir -p /workspace/data \
    /workspace/output \
    /workspace/logs \
    /workspace/cache

# Set environment variables for tool paths (can be overridden at runtime)
ENV AGENT_PYTHON=/usr/bin/python3.10
ENV BLENDER_PYTHON=/usr/bin/python3.11
ENV PPTX_PYTHON=/usr/bin/python3.10
ENV WEB_PYTHON=/usr/bin/python3.10

# Create a startup script
RUN echo '#!/bin/bash\n\
echo "================================="\n\
echo "VIGA Docker Container"\n\
echo "================================="\n\
echo ""\n\
echo "Python versions:"\n\
python3.10 --version\n\
python3.11 --version\n\
echo ""\n\
echo "CUDA version:"\n\
nvcc --version 2>/dev/null || echo "CUDA not available"\n\
echo ""\n\
echo "GPU status:"\n\
nvidia-smi 2>/dev/null || echo "No GPU detected"\n\
echo ""\n\
echo "To run VIGA:"\n\
echo "  python main.py --mode <mode> --model <model> [options]"\n\
echo ""\n\
echo "Available modes:"\n\
echo "  - blendergym: Single-step 3D graphics editing"\n\
echo "  - blenderstudio: Multi-step 3D graphics editing"\n\
echo "  - static_scene: Single-view 3D scene reconstruction"\n\
echo "  - dynamic_scene: 4D dynamic scene reconstruction"\n\
echo "  - autopresent: 2D slide/document layout synthesis"\n\
echo ""\n\
echo "For help: python main.py --help"\n\
echo "================================="\n\
exec "$@"\n\
' > /workspace/startup.sh && chmod +x /workspace/startup.sh

# Set the entrypoint to show info and start bash
ENTRYPOINT ["/workspace/startup.sh"]
CMD ["/bin/bash"]
