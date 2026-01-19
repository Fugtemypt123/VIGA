# Conda Environment Setup Guide

This guide provides detailed instructions for setting up VIGA using Conda environments. This is an alternative to the Docker-based setup.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation Steps](#installation-steps)
- [External Dependencies](#external-dependencies)
- [Configuration](#configuration)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)

## Prerequisites

- Python 3.10+ (Python 3.11 for Blender tools)
- Conda or Miniconda
- NVIDIA GPU with CUDA 12.8+ support (for GPU-accelerated tasks)
- At least 20GB of free disk space

## Installation Steps

### 1. Install Conda

If you don't have Conda installed:

```bash
# Download Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Install Miniconda
bash Miniconda3-latest-Linux-x86_64.sh

# Follow the prompts and restart your shell
source ~/.bashrc
```

### 2. Create Agent Environment (Required)

This is the main environment for running VIGA agents:

```bash
# Create environment
conda create -n agent python=3.10 -y

# Activate environment
conda activate agent

# Install dependencies
pip install -r requirements/requirement_agent.txt
```

### 3. Create Tool Environments

Install tool environments based on the modes you want to use:

#### For 3D Modes (BlenderGym, BlenderStudio, Static/Dynamic Scene)

```bash
# Create Blender environment
conda create -n blender python=3.11 -y

# Activate environment
conda activate blender

# Install dependencies
pip install -r requirements/requirement_blender.txt
```

#### For AutoPresent Mode

```bash
# Create PPTX environment
conda create -n pptx python=3.10 -y

# Activate environment
conda activate pptx

# Install dependencies
pip install -r requirements/requirement_pptx.txt
```

#### For Design2Code Mode

```bash
# Create Web environment
conda create -n web python=3.10 -y

# Activate environment
conda activate web

# Install dependencies (if requirement_web.txt exists)
pip install -r requirements/requirement_web.txt
```

### 4. Optional Evaluation Environments

If you plan to run evaluations:

```bash
# For Blender evaluation
conda create -n eval-blender python=3.11 -y
conda activate eval-blender
pip install -r requirements/requirement_eval-blender.txt

# For PPTX evaluation
conda create -n eval-pptx python=3.10 -y
conda activate eval-pptx
pip install -r requirements/requirement_eval-pptx.txt
```

## External Dependencies

### Blender and Infinigen (for 3D modes)

#### For BlenderGym

```bash
# Navigate to utils directory
cd utils

# Clone Infinigen (minimal installation)
git clone git@github.com:richard-guyunqi/infinigen.git infinigen_minimal

# Navigate to Infinigen directory
cd infinigen_minimal

# Activate blender environment
conda activate blender

# Install Infinigen with minimal dependencies
INFINIGEN_MINIMAL_INSTALL=True bash scripts/install/interactive_blender.sh

# Return to project root
cd ../..
```

#### For Other 3D Modes (Static Scene, Dynamic Scene, BlenderStudio)

```bash
# Navigate to utils directory
cd utils

# Clone Infinigen (full installation)
git clone https://github.com/princeton-vl/infinigen.git

# Navigate to Infinigen directory
cd infinigen

# Activate blender environment
conda activate blender

# Install Infinigen with full dependencies
bash scripts/install/interactive_blender.sh

# Return to project root
cd ../..
```

**Note**: The Infinigen installation will download and install Blender automatically. This may take 10-30 minutes depending on your internet connection.

### LibreOffice (for AutoPresent)

Required for converting PowerPoint presentations to PDF:

```bash
# Install LibreOffice and unoconv
sudo apt-get update
sudo apt-get install -y libreoffice unoconv

# Verify installation
libreoffice --version
```

### Google Chrome (for Design2Code)

Required for rendering and evaluating HTML/CSS outputs:

```bash
# Download Chrome
wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb

# Install Chrome
sudo apt install ./google-chrome-stable_current_amd64.deb

# Verify installation
google-chrome --version

# Clean up
rm google-chrome-stable_current_amd64.deb
```

## Configuration

### 1. API Keys Configuration

Create `utils/_api_keys.py` with your API keys:

```python
# OpenAI API Configuration
OPENAI_API_KEY = "your-openai-api-key"
OPENAI_BASE_URL = "https://api.openai.com/v1"

# Claude API Configuration
CLAUDE_API_KEY = "your-claude-api-key"
CLAUDE_BASE_URL = "https://api.anthropic.com/v1"

# Google Gemini API Configuration
GEMINI_API_KEY = "your-gemini-api-key"

# Meshy API Configuration (for 3D asset generation)
MESHY_API_KEY = "your-meshy-api-key"
```

**Security Note**: Never commit this file to version control. Add it to `.gitignore`.

### 2. Environment Paths Configuration

Create `utils/_path.py` to configure paths for different tool environments:

```python
import os

# Get conda environment paths
conda_prefix = os.environ.get('CONDA_PREFIX', '')

# Configure paths for different tool environments
path_to_cmd = {
    # Blender tools (Python 3.11)
    "tools/blender/exec.py": "/path/to/conda/envs/blender/bin/python",
    "tools/blender/investigator.py": "/path/to/conda/envs/blender/bin/python",

    # PPTX tools (Python 3.10)
    "tools/slides/exec.py": "/path/to/conda/envs/pptx/bin/python",

    # Web tools (Python 3.10)
    "tools/exec_html.py": "/path/to/conda/envs/web/bin/python",

    # Add other tool paths as needed
}

# Helper function to get conda environment path
def get_conda_env_python(env_name):
    """Get Python path for a conda environment"""
    conda_root = os.path.expanduser("~/miniconda3")  # Adjust if different
    return f"{conda_root}/envs/{env_name}/bin/python"

# Alternative: Use helper function
path_to_cmd = {
    "tools/blender/exec.py": get_conda_env_python("blender"),
    "tools/blender/investigator.py": get_conda_env_python("blender"),
    "tools/slides/exec.py": get_conda_env_python("pptx"),
    "tools/exec_html.py": get_conda_env_python("web"),
}
```

**Finding Your Conda Paths**:

```bash
# Find conda installation directory
conda info --base

# Find specific environment path
conda env list

# Example output:
# blender     /home/user/miniconda3/envs/blender
# pptx        /home/user/miniconda3/envs/pptx
```

### 3. Blender Configuration

After installing Infinigen, note the Blender executable path:

```bash
# Find Blender executable (installed by Infinigen)
find ~/miniconda3/envs/blender -name "blender" -type f

# Example path: ~/miniconda3/envs/blender/lib/python3.11/site-packages/infinigen/bin/blender
```

If needed, add this to your `_path.py`:

```python
BLENDER_EXECUTABLE = "/path/to/blender/executable"
```

## Verification

### Verify Agent Environment

```bash
conda activate agent
python -c "import openai; import mcp; print('Agent environment OK')"
```

### Verify Blender Environment

```bash
conda activate blender
python -c "import bpy; import infinigen; print('Blender environment OK')"
```

### Verify PPTX Environment

```bash
conda activate pptx
python -c "import pptx; print('PPTX environment OK')"
```

### Verify GPU Access (if using CUDA)

```bash
conda activate blender
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Run a Quick Test

```bash
conda activate agent
python main.py --mode blendergym --help
```

If this displays the help message without errors, your setup is correct.

## Troubleshooting

### Import Errors

**Problem**: `ModuleNotFoundError` when running VIGA

**Solution**:
```bash
# Ensure you're in the correct environment
conda activate agent

# Reinstall dependencies
pip install -r requirements/requirement_agent.txt

# Verify installation
pip list | grep mcp
```

### Infinigen Installation Fails

**Problem**: Errors during Infinigen installation

**Solution**:
```bash
# Check Python version
python --version  # Should be 3.11 for Blender

# Check system dependencies
sudo apt-get install -y build-essential cmake

# Try manual installation
cd utils/third_party/infinigen
bash scripts/install/interactive_blender.sh 2>&1 | tee install.log

# Check install.log for specific errors
```

### CUDA/GPU Issues

**Problem**: GPU not detected or CUDA errors

**Solution**:
```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA version
nvcc --version

# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

### LibreOffice Not Converting Files

**Problem**: PPTX to PDF conversion fails

**Solution**:
```bash
# Reinstall LibreOffice
sudo apt-get remove --purge libreoffice*
sudo apt-get install libreoffice

# Test conversion manually
libreoffice --headless --convert-to pdf test.pptx
```

### Chrome/Playwright Issues

**Problem**: Browser-based tasks fail

**Solution**:
```bash
# Install Chrome dependencies
sudo apt-get install -y libgbm1 libnss3 libxss1 libasound2

# If using Playwright, install browsers
playwright install chromium
```

### Path Configuration Issues

**Problem**: Tool scripts can't find correct Python interpreters

**Solution**:
```bash
# Check environment paths
conda env list

# Update _path.py with correct absolute paths
# Use which command to find exact paths:
conda activate blender
which python  # Copy this path to _path.py

conda activate pptx
which python  # Copy this path to _path.py
```

### Environment Conflicts

**Problem**: Dependency conflicts between environments

**Solution**:
```bash
# Remove and recreate environment
conda deactivate
conda env remove -n blender
conda create -n blender python=3.11 -y
conda activate blender
pip install -r requirements/requirement_blender.txt
```

## Managing Environments

### List All Environments

```bash
conda env list
```

### Remove an Environment

```bash
conda deactivate
conda env remove -n environment_name
```

### Export Environment

```bash
conda activate agent
conda env export > environment_agent.yml
```

### Create from Exported File

```bash
conda env create -f environment_agent.yml
```

### Update Dependencies

```bash
conda activate agent
pip install -r requirements/requirement_agent.txt --upgrade
```

## Performance Tips

1. **Use mamba**: For faster package installation
   ```bash
   conda install -c conda-forge mamba
   mamba install package_name
   ```

2. **Clean conda cache**: Free up disk space
   ```bash
   conda clean --all
   ```

3. **Use pip cache**: Speed up repeated installations
   ```bash
   pip cache dir  # Check cache location
   pip cache purge  # Clear cache if needed
   ```

## Additional Resources

- [Conda Documentation](https://docs.conda.io/)
- [Infinigen Installation Guide](https://github.com/princeton-vl/infinigen)
- [Blender Python API](https://docs.blender.org/api/current/)
- [PyTorch Installation](https://pytorch.org/get-started/locally/)
