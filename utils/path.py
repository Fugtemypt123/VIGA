"""Tool script to Python environment path mappings.

Configure these paths to point to the appropriate conda environment Python interpreters.
See requirements/ for the environment setup instructions.

Usage:
    1. Copy this file to utils/_path.py (which is gitignored)
    2. Update the CONDA_BASE path to your conda installation
    3. The tool paths will be automatically resolved

Alternatively, set the VIGA_CONDA_BASE environment variable.
"""
import os
import shutil
from typing import Dict

# Base path for conda environments
# Override by setting VIGA_CONDA_BASE environment variable
CONDA_BASE = os.environ.get(
    "VIGA_CONDA_BASE",
    os.path.expanduser("~/anaconda3/envs")  # Default conda envs location
)

# Environment name to tool script mapping
ENV_MAPPING: Dict[str, str] = {
    # Blender tools (Python 3.11)
    "tools/exec_blender.py": "blender",
    "tools/investigator.py": "blender",
    # PPTX tools (Python 3.10)
    "tools/exec_slides.py": "pptx",
    # Chrome/HTML tools (Python 3.10)
    "tools/exec_html.py": "chrome",
    # Core agent tools (Python 3.10)
    "tools/generator_base.py": "agent",
    "tools/initialize_plan.py": "agent",
    "tools/meshy.py": "agent",
    "tools/verifier_base.py": "agent",
    "tools/undo.py": "agent",
    # SAM segmentation tools
    "tools/sam.py": "sam3d-objects",
    "tools/sam_init.py": "sam3d-objects",
    "tools/sam_worker.py": "sam",
    "tools/sam3_worker.py": "sam3",
    "tools/sam3d_worker.py": "sam3d-objects",
}


def get_python_path(env_name: str) -> str:
    """Get Python interpreter path for a conda environment.

    Args:
        env_name: Name of the conda environment.

    Returns:
        Path to the Python interpreter.
    """
    conda_python = os.path.join(CONDA_BASE, env_name, "bin", "python")
    if os.path.exists(conda_python):
        return conda_python
    # Fallback: try to find python in PATH
    python_path = shutil.which("python")
    if python_path:
        return python_path
    return "python"


# Build the path_to_cmd mapping
path_to_cmd: Dict[str, str] = {
    tool: get_python_path(env) for tool, env in ENV_MAPPING.items()
}

# Try to import user overrides from _path.py (gitignored)
try:
    from utils._path import path_to_cmd as user_paths
    path_to_cmd.update(user_paths)
except ImportError:
    pass
