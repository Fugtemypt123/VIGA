"""SAM Scene Initialization MCP Server.

This module provides an MCP server for scene reconstruction using SAM (Segment
Anything Model) and SAM-3D. It detects objects in images, reconstructs them
in 3D, and imports them into Blender scenes.
"""

import json
import os
import shutil
import subprocess
import sys
import traceback
from typing import Dict, List, Optional, Tuple

import numpy as np
from mcp.server.fastmcp import FastMCP

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.path import path_to_cmd

# Tool configurations for the agent (empty as tools are auto-discovered)
tool_configs: List[Dict[str, object]] = []

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SAM_WORKER = os.path.join(os.path.dirname(__file__), "segmentation", "sam_worker.py")
SAM3D_WORKER = os.path.join(os.path.dirname(__file__), "segmentation", "sam3d_worker.py")
IMPORT_SCRIPT = os.path.join(os.path.dirname(__file__), "blender", "glb_import.py")

mcp = FastMCP("sam-init")

# Global state variables
_target_image: Optional[str] = None
_output_dir: Optional[str] = None
_sam3_cfg: Optional[str] = None
_blender_command: Optional[str] = None
_blender_file: Optional[str] = None
_log_file: Optional[object] = None
_sam_env_bin: Optional[str] = None
_sam3d_env_bin: Optional[str] = None

# Safely get paths to avoid uncaught KeyError exceptions
try:
    _sam_env_bin = path_to_cmd.get("tools/segmentation/sam_worker.py")
    _sam3d_env_bin = path_to_cmd.get("tools/segmentation/sam3d_worker.py")
except Exception as e:
    print(f"[SAM_INIT] Error initializing paths: {e}", file=sys.stderr)


def log(message: str) -> None:
    """Output to both stderr and log file."""
    # Output to stderr (for real-time viewing)
    print(message, file=sys.stderr)
    # Output to log file (for later viewing)
    if _log_file is not None:
        try:
            _log_file.write(message + "\n")
            _log_file.flush()  # Flush immediately to ensure content is written
        except Exception:
            pass  # If log file write fails, don't affect the main process


def get_conda_prefix_from_python_path(python_path: str) -> Optional[str]:
    """Infer CONDA_PREFIX from Python executable path.

    Example: /path/to/envs/env_name/bin/python -> /path/to/envs/env_name
    """
    if not python_path:
        return None
    
    # Normalize path (handle both relative and absolute paths)
    if os.path.isabs(python_path):
        normalized_path = python_path
    else:
        normalized_path = os.path.abspath(python_path)

    # Method 1: If path ends with /bin/python or /bin/python3, remove the last two directory levels
    if normalized_path.endswith('/bin/python') or normalized_path.endswith('/bin/python3'):
        conda_prefix = os.path.dirname(os.path.dirname(normalized_path))
        return conda_prefix

    # Method 2: If path contains /envs/, extract environment path
    if '/envs/' in normalized_path:
        parts = normalized_path.split('/envs/')
        if len(parts) == 2:
            env_part = parts[1].split('/')[0]
            conda_prefix = os.path.join(parts[0], 'envs', env_part)
            return conda_prefix

    # Method 3: If path contains /bin/python (for relative path cases)
    if '/bin/python' in normalized_path:
        idx = normalized_path.rfind('/bin/python')
        conda_prefix = normalized_path[:idx]
        return conda_prefix
    
    return None


def prepare_env_with_conda_prefix(python_path: str) -> Dict[str, str]:
    """Prepare environment variable dictionary with CONDA_PREFIX."""
    env = os.environ.copy()

    # Always infer CONDA_PREFIX from Python path to ensure subprocess uses the correct environment
    conda_prefix = get_conda_prefix_from_python_path(python_path)
    if conda_prefix:
        env["CONDA_PREFIX"] = conda_prefix
        log(f"[SAM_INIT] Set CONDA_PREFIX={conda_prefix} from Python path: {python_path}")
    else:
        # If unable to infer, but current environment has CONDA_PREFIX, keep it
        if "CONDA_PREFIX" in env:
            log(f"[SAM_INIT] Could not infer CONDA_PREFIX from {python_path}, keeping existing: {env['CONDA_PREFIX']}")
        else:
            log(f"[SAM_INIT] Warning: Could not infer CONDA_PREFIX from {python_path} and no existing CONDA_PREFIX")
    
    return env


@mcp.tool()
def initialize(args: Dict[str, object]) -> Dict[str, object]:
    """Initialize SAM scene reconstruction with configuration."""
    global _target_image, _output_dir, _sam3_cfg, _blender_command, _sam_env_bin, _blender_file, _log_file
    try:
        _target_image = args["target_image_path"]
        _output_dir = args.get("output_dir") + "/sam_init"
        if os.path.exists(_output_dir):
            shutil.rmtree(_output_dir)
        os.makedirs(_output_dir, exist_ok=True)
        
        # Initialize log file
        log_path = os.path.join(_output_dir, "sam_init.log")
        _log_file = open(log_path, 'w', encoding='utf-8')
        log(f"[SAM_INIT] Initialized. Log file: {log_path}")
        _sam3_cfg = args.get("sam3d_config_path") or os.path.join(
            ROOT, "utils", "sam3d", "checkpoints", "hf", "pipeline.yaml"
        )
        _blender_command = args.get("blender_command") or "utils/infinigen/blender/blender"
        # Record the passed blender_file parameter for later writing directly to that path during reconstruction
        _blender_file = args.get("blender_file")

        # Try to get the python path for sam_worker.py
        # If not configured, use sam3d environment (assuming they might be in the same environment)
        _sam_env_bin = path_to_cmd.get("tools/segmentation/sam_worker.py") or _sam3d_env_bin
        
        log("[SAM_INIT] sam init initialized")
        return {"status": "success", "output": {"text": ["sam init initialized"], "tool_configs": tool_configs}}
    except Exception as e:
        error_msg = f"Error in initialize: {str(e)}"
        if _log_file:
            log(f"[SAM_INIT] {error_msg}")
        else:
            print(f"[SAM_INIT] {error_msg}", file=sys.stderr)
        return {"status": "error", "output": {"text": [error_msg]}}


def process_single_object(
    args_tuple: Tuple[int, object, str, str, str, str, str, str, str, str]
) -> Tuple[bool, Optional[str], Optional[Dict[str, object]], Optional[str]]:
    """Process 3D reconstruction task for a single object.

    Args:
        args_tuple: Tuple containing (idx, mask, object_name, target_image,
                    output_dir, sam3_cfg, blender_command, sam3d_env_bin,
                    ROOT, SAM3D_WORKER).

    Returns:
        Tuple of (success, glb_path, object_transform, error_msg).
    """
    idx, mask, object_name, _target_image, _output_dir, _sam3_cfg, _blender_command, _sam3d_env_bin, ROOT, SAM3D_WORKER = args_tuple
    
    try:
        # Use mask file already saved by sam_worker.py (if exists), otherwise save a new one
        mask_path = os.path.join(_output_dir, f"{object_name}.npy")
        if not os.path.exists(mask_path):
            # If file doesn't exist, save the mask (this shouldn't happen, but kept for robustness)
            np.save(mask_path, mask)
        else:
            log(f"[SAM_INIT] Using existing mask file: {mask_path}")

        # Reconstruct 3D using the same object name
        glb_path = os.path.join(_output_dir, f"{object_name}.glb")
        info_path = os.path.join(_output_dir, f"{object_name}.json")

        # If file already exists (possibly generated in previous run), skip reconstruction
        if os.path.exists(glb_path) and os.path.exists(info_path):
            log(f"[SAM_INIT] GLB file already exists, skipping reconstruction: {glb_path}")
            with open(info_path, 'r') as f:
                info = json.load(f)
            return (True, glb_path, info, None)

        # Run SAM-3D reconstruction, using --info parameter to write JSON output to file instead of stdout
        # Redirect subprocess output to log file to avoid polluting stdout
        log_path = os.path.join(_output_dir, f"{object_name}_sam3d.log")
        # Prepare environment variables, ensuring CONDA_PREFIX is included (inferred from Python path)
        env = prepare_env_with_conda_prefix(_sam3d_env_bin)
        with open(log_path, 'w') as log_file:
            r = subprocess.run(
                [
                    _sam3d_env_bin,
                    SAM3D_WORKER,
                    "--image",
                    _target_image,
                    "--mask",
                    mask_path,
                    "--config",
                    _sam3_cfg,
                    "--glb",
                    glb_path,
                    "--info",
                    info_path,  # Specify JSON output file path
                ],
                cwd=ROOT,
                check=True,
                text=True,
                stdout=log_file,
                stderr=subprocess.STDOUT,  # Also redirect stderr to stdout
                env=env,  # Pass environment variables
            )

        # Read JSON output from file instead of parsing from stdout (avoid stdout pollution)
        if not os.path.exists(info_path):
            error_msg = f"Object {idx} ({object_name}) reconstruction failed: info file not created"
            log(f"[SAM_INIT] Warning: {error_msg}")
            return (False, None, None, error_msg)
        
        try:
            with open(info_path, 'r') as f:
                info = json.load(f)
        except json.JSONDecodeError as e:
            error_msg = f"Object {idx} ({object_name}) failed to parse info file: {str(e)}"
            log(f"[SAM_INIT] Warning: {error_msg}")
            return (False, None, None, error_msg)
        glb_path_value = info.get("glb_path")
        if glb_path_value:
            object_transform = {
                "glb_path": glb_path_value,
                "translation": info.get("translation"),
                "rotation": info.get("rotation"),
                "scale": info.get("scale"),
            }
            log(f"[SAM_INIT] Successfully reconstructed object {idx} ({object_name})")
            return (True, glb_path_value, object_transform, None)
        else:
            error_msg = f"Object {idx} ({object_name}) reconstruction failed or no GLB generated"
            log(f"[SAM_INIT] Warning: {error_msg}")
            return (False, None, None, error_msg)
            
    except subprocess.CalledProcessError as e:
        # Read log file content to get detailed error information
        log_path = os.path.join(_output_dir, f"{object_name}_sam3d.log")
        log_content = ""
        if os.path.exists(log_path):
            try:
                with open(log_path, 'r') as f:
                    log_content = f.read()
                    # Only take last 500 characters to avoid overly long logs
                    if len(log_content) > 500:
                        log_content = "..." + log_content[-500:]
            except Exception:
                pass
        error_msg = f"Failed to reconstruct object {idx} ({object_name})"
        if log_content:
            error_msg += f". Log: {log_content}"
        log(f"[SAM_INIT] Warning: {error_msg}. Full log: {log_path}")
        return (False, None, None, error_msg)
    except json.JSONDecodeError as e:
        error_msg = f"Failed to parse output for object {idx} ({object_name}): {str(e)}"
        log(f"[SAM_INIT] Warning: {error_msg}")
        return (False, None, None, error_msg)
    except Exception as e:
        error_msg = f"Unexpected error processing object {idx} ({object_name}): {str(e)}"
        log(f"[SAM_INIT] Error: {error_msg}")
        return (False, None, None, error_msg)


@mcp.tool()
def reconstruct_full_scene() -> Dict[str, object]:
    """Reconstruct complete 3D scene from input image.

    Steps:
        1. Use SAM to detect all objects
        2. Use SAM-3D to reconstruct each object
        3. Import all objects into Blender and save as .blend file
    """
    global _target_image, _output_dir, _sam3_cfg, _blender_command, _sam_env_bin, _sam3d_env_bin, _blender_file
    
    if not _target_image or not _output_dir:
        return {"status": "error", "output": {"text": ["call initialize first"]}}
    
    if _sam_env_bin is None:
        return {"status": "error", "output": {"text": ["SAM worker python path not configured"]}}
    
    try:
        # Step 1: Use SAM to get masks for all objects
        all_masks_path = os.path.join(_output_dir, "all_masks.npy")
        sam_log_path = os.path.join(_output_dir, "sam_worker.log")
        log(f"[SAM_INIT] Step 1: Detecting all objects with SAM...")
        log(f"[SAM_INIT] SAM worker output will be saved to: {sam_log_path}")
        # Prepare environment variables, ensuring CONDA_PREFIX is included (inferred from Python path)
        env = prepare_env_with_conda_prefix(_sam_env_bin)
        with open(sam_log_path, 'w') as log_file:
            subprocess.run(
                [
                    _sam_env_bin,
                    SAM_WORKER,
                    "--image",
                    _target_image,
                    "--out",
                    all_masks_path,
                ],
                cwd=ROOT,
                check=True,
                text=True,
                stdout=log_file,
                stderr=subprocess.STDOUT,  # Also redirect stderr to stdout
                env=env,  # Pass environment variables
            )

        # Step 2: Load masks and object name mapping
        masks = np.load(all_masks_path, allow_pickle=True)

        # Handle case where masks might be an object array
        if masks.dtype == object:
            masks = [m for m in masks]
        elif masks.ndim == 3:
            # If it's a 3D array (N, H, W), convert to list
            masks = [masks[i] for i in range(masks.shape[0])]
        else:
            # Single mask case
            masks = [masks]

        # Load object name mapping information
        object_names_json_path = all_masks_path.replace('.npy', '_object_names.json')
        object_mapping = None
        if os.path.exists(object_names_json_path):
            with open(object_names_json_path, 'r') as f:
                object_names_info = json.load(f)
                object_mapping = object_names_info.get("object_mapping", [])
                log(f"[SAM_INIT] Loaded object names mapping from: {object_names_json_path}")
        else:
            log(f"[SAM_INIT] Warning: Object names mapping file not found: {object_names_json_path}, using default names")

        log(f"[SAM_INIT] Step 2: Reconstructing {len(masks)} objects with SAM-3D (parallel processing)...")

        # Prepare parameter list
        tasks = []
        for idx, mask in enumerate(masks):
            # Get object name (if available, otherwise use default name)
            if object_mapping and idx < len(object_mapping):
                object_name = object_mapping[idx]
            else:
                object_name = f"object_{idx}"
            
            tasks.append((
                idx, mask, object_name, _target_image, _output_dir, _sam3_cfg,
                _blender_command, _sam3d_env_bin, ROOT, SAM3D_WORKER
            ))
        
        # Process all tasks serially
        glb_paths = []
        object_transforms = []  # Store position information for each object

        # Process each task in order
        for task in tasks:
            idx = task[0]
            try:
                success, glb_path, object_transform, error_msg = process_single_object(task)
                if success and glb_path and object_transform:
                    glb_paths.append(glb_path)
                    object_transforms.append(object_transform)
            except Exception as e:
                log(f"[SAM_INIT] Error processing object {idx}: {str(e)}")
        
        if len(glb_paths) == 0:
            return {"status": "error", "output": {"text": ["No objects were successfully reconstructed"]}}

        # Step 3: Import all GLBs into Blender and save as .blend file
        log(f"[SAM_INIT] Step 3: Importing {len(glb_paths)} objects into Blender...")
        # If blender_file was provided in initialize, write directly to that path
        # Otherwise default to scene.blend in the output directory
        blend_path = _blender_file or os.path.join(_output_dir, "scene.blend")

        # Save position information to JSON file
        transforms_json_path = os.path.join(_output_dir, "object_transforms.json")
        with open(transforms_json_path, 'w') as f:
            json.dump(object_transforms, f, indent=2)

        # Build Blender command
        blender_cmd = [
            _blender_command,
            "-b",  # Background mode
            "-P",  # Run Python script
            IMPORT_SCRIPT,
            "--",
            transforms_json_path,  # Pass position information JSON file path
            blend_path,  # Output .blend file path
        ]

        blender_log_path = os.path.join(_output_dir, "blender_import.log")
        log(f"[SAM_INIT] Blender import output will be saved to: {blender_log_path}")
        # For Blender, use current environment's environment variables (Blender typically doesn't need CONDA_PREFIX)
        env = os.environ.copy()
        with open(blender_log_path, 'w') as log_file:
            subprocess.run(
                blender_cmd,
                cwd=ROOT,
                check=True,
                text=True,
                stdout=log_file,
                stderr=subprocess.STDOUT,  # Also redirect stderr to stdout
                env=env,  # Pass environment variables
            )

        log(f"[SAM_INIT] Deleting .blend1 files in {os.path.dirname(_output_dir)}")
        # Delete any extra .blend1 files that may have been created
        for file in os.listdir(os.path.dirname(_output_dir)):
            if file.endswith(".blend1"):
                os.remove(os.path.join(os.path.dirname(_output_dir), file))
        
        return {
            "status": "success",
            "output": {
                "text": [
                    f"Successfully reconstructed {len(glb_paths)} objects, Blender scene saved to: {blend_path}",
                ],
                "data": {
                    "blend_path": blend_path,
                    "num_objects": len(glb_paths),
                    "num_masks": len(masks),
                    "glb_paths": glb_paths
                }
            }
        }
    
    except subprocess.CalledProcessError as e:
        # Try to read relevant log file contents
        error_msg = str(e)
        log_files = []
        if os.path.exists(os.path.join(_output_dir, "sam_worker.log")):
            log_files.append("sam_worker.log")
        if os.path.exists(os.path.join(_output_dir, "blender_import.log")):
            log_files.append("blender_import.log")
        
        if log_files:
            error_msg += f". Check log files: {', '.join(log_files)}"
        
        log(f"[SAM_INIT] Process failed: {error_msg}")
        return {"status": "error", "output": {"text": [f"Process failed: {error_msg}"]}}
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        log(f"[SAM_INIT] {error_msg}")
        log(f"[SAM_INIT] Traceback: {traceback.format_exc()}")
        return {"status": "error", "output": {"text": [error_msg]}}


def main() -> None:
    """Run MCP server or execute test mode."""
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        task = sys.argv[2]
        initialize(
            {
                "target_image_path": f"data/static_scene/{task}/target.png",
                "output_dir": os.path.join(ROOT, "output", "test", task),
                "blender_command": "utils/infinigen/blender/blender",
            }
        )
        result = reconstruct_full_scene()
        print(json.dumps(result, indent=2), file=sys.stderr)
    else:
        mcp.run()


if __name__ == "__main__":
    main()