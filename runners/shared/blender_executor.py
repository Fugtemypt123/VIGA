"""Blender code execution utilities for alchemy runners."""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple


def execute_blender_code(
    blender_command: str,
    blender_file: str,
    blender_script: str,
    code: str,
    round_name: str,
    script_save_dir: Optional[Path],
    render_save_dir: Path,
    gpu_devices: Optional[str] = None
) -> Tuple[bool, str, str]:
    """Execute Blender Python code and render images.

    Args:
        blender_command: Path to Blender executable.
        blender_file: Path to Blender file.
        blender_script: Path to Blender execution script.
        code: Python code to execute.
        round_name: Name for the round (e.g., "1", "temp_1_0").
        script_save_dir: Directory to save the code file (None to skip saving).
        render_save_dir: Directory to save rendered images.
        gpu_devices: GPU devices string (e.g., "0,1").

    Returns:
        Tuple of (success, error_message, render_dir_path).
        render_dir_path is the path to the render directory if successful.
    """
    # Save code to temporary file for execution
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp_code:
        tmp_code.write(code)
        tmp_code_path = tmp_code.name

    try:
        # Create render directory
        render_dir = render_save_dir / round_name
        render_dir.mkdir(parents=True, exist_ok=True)
        # Clear existing files in render directory
        for img in render_dir.glob("*.png"):
            img.unlink()

        # Execute Blender
        cmd = [
            blender_command,
            "--background", blender_file,
            "--python", blender_script,
            "--", tmp_code_path, str(render_dir)
        ]

        if gpu_devices:
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu_devices
        else:
            env = None

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
            env=env
        )

        if result.returncode != 0:
            os.unlink(tmp_code_path)
            return False, result.stderr + result.stdout, ""

        # Check if render files exist
        render_files = list(render_dir.glob("*.png"))
        if len(render_files) == 0:
            os.unlink(tmp_code_path)
            return False, "No render output generated", ""

        os.unlink(tmp_code_path)
        return True, "", str(render_dir)

    except subprocess.TimeoutExpired:
        os.unlink(tmp_code_path)
        return False, "Execution timeout", ""
    except Exception as e:
        if os.path.exists(tmp_code_path):
            os.unlink(tmp_code_path)
        return False, str(e), ""
