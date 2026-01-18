"""SAM Bridge MCP Server for 3D Asset Generation.

This server bridges SAM3 (Segment Anything Model 3) for image segmentation
and SAM3D for 3D reconstruction, enabling extraction of objects from images
and converting them into 3D GLB assets.
"""

import json
import os
import subprocess
import sys
from typing import Dict, List, Optional

from mcp.server.fastmcp import FastMCP

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.path import path_to_cmd

# Tool configurations for the agent
tool_configs: List[Dict[str, object]] = [
    {
        "type": "function",
        "function": {
            "name": "get_better_object",
            "description": (
                "Generate high-quality 3D assets, download them locally, and provide "
                "their paths for later use. The textures, materials, and finishes of "
                "these objects are already high-quality with fine-grained detail; "
                "please do not repaint them. If you do, you will need to re-import "
                "the object."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "object_name": {
                        "type": "string",
                        "description": (
                            "The name of the object to download. For example, "
                            "'chair', 'table', 'lamp', etc."
                        ),
                    },
                },
                "required": ["object_name"]
            }
        }
    }
]

ROOT: str = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SAM3_WORKER: str = os.path.join(os.path.dirname(__file__), "sam3_worker.py")
SAM3D_WORKER: str = os.path.join(os.path.dirname(__file__), "sam3d_worker.py")

mcp = FastMCP("sam-bridge")

# Global state
_target_image: Optional[str] = None
_output_dir: Optional[str] = None
_sam3_cfg: Optional[str] = None
_blender_command: Optional[str] = None
_sam3_env_bin: str = path_to_cmd["tools/sam3_worker.py"]
_sam3d_env_bin: str = path_to_cmd["tools/sam3d_worker.py"]


@mcp.tool()
def initialize(args: Dict[str, object]) -> Dict[str, object]:
    """Initialize the SAM bridge with configuration.

    Args:
        args: Configuration dictionary with keys:
            - target_image_path: Path to the target image for segmentation.
            - output_dir: Base directory for output files.
            - sam3d_config_path: Optional path to SAM3D config file.
            - blender_command: Optional path to Blender executable.

    Returns:
        Dictionary with status and tool configurations.
    """
    global _target_image, _output_dir, _sam3_cfg, _blender_command
    _target_image = args["target_image_path"]
    _output_dir = args.get("output_dir") + "/sam"
    os.makedirs(_output_dir, exist_ok=True)
    _sam3_cfg = args.get("sam3d_config_path") or os.path.join(
        ROOT, "utils", "sam3d", "checkpoints", "hf", "pipeline.yaml"
    )
    _blender_command = args.get("blender_command") or "utils/infinigen/blender/blender"
    return {
        "status": "success",
        "output": {"text": ["SAM bridge initialized successfully"], "tool_configs": tool_configs}
    }


@mcp.tool()
def get_better_object(object_name: str) -> Dict[str, object]:
    """Generate a 3D asset from an object in the target image.

    Uses SAM3 to segment the object from the image, then SAM3D to
    reconstruct it as a 3D GLB model.

    Args:
        object_name: Name of the object to extract (e.g., 'chair', 'table').

    Returns:
        Dictionary with status and either the GLB path or error message.
    """
    original_object_name = object_name
    object_name = object_name.replace(' ', '_')
    object_name = object_name.replace('-', '_')

    if not _target_image or not _output_dir:
        return {"status": "error", "output": {"text": ["Call initialize first"]}}

    mask_path = os.path.join(_output_dir, f"{object_name}_mask.npy")
    glb_path = os.path.join(_output_dir, f"{object_name}.glb")

    try:
        # Step 1: Run SAM3 to generate segmentation mask
        subprocess.run(
            [
                _sam3_env_bin,
                SAM3_WORKER,
                "--image",
                _target_image,
                "--object",
                original_object_name,
                "--out",
                mask_path,
            ],
            check=True,
            text=True,
            capture_output=True,
        )

        # Step 2: Run SAM3D to reconstruct 3D model from mask
        r2 = subprocess.run(
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
            ],
            cwd=ROOT,
            check=True,
            text=True,
            capture_output=True,
        )

        info = json.loads(r2.stdout.strip().splitlines()[-1])
        info["glb_path"] = info.get("glb_path") or glb_path
        return {
            "status": "success",
            "output": {
                "text": [
                    f"Successfully generated asset, downloaded to: {info['glb_path']}",
                    f"Asset information: {json.dumps(info)}"
                ]
            }
        }
    except subprocess.CalledProcessError:
        return {
            "status": "error",
            "output": {"text": [f"Object {object_name} not found in the image."]}
        }


def main() -> None:
    """Run the MCP server or execute test mode."""
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        initialize(
            {
                "target_image_path": "data/static_scene/blackhouse25/target.jpeg",
                "output_dir": os.path.join(ROOT, "output", "test", "sam3"),
                "blender_command": "utils/infinigen/blender/blender",
            }
        )
        print(get_better_object("house"))
    else:
        mcp.run()


if __name__ == "__main__":
    main()
