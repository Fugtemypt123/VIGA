import os, sys, json, subprocess
from mcp.server.fastmcp import FastMCP
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.path import path_to_cmd

# tool_configs for agent (only the function w/ @mcp.tools)
tool_configs = [
    {
        "type": "function",
        "function": {
            "name": "get_better_object",
            "description": "Generate high-quality 3D assets, download them locally, and provide their paths for later use. The textures, materials, and finishes of these objects are already high-quality with fine-grained detail; please do not repaint them. If you do, you will need to re-import the object.",
            "parameters": {
                "type": "object",
                "properties": {
                    "object_name": {"type": "string", "description": "The name of the object to download. For example, 'chair', 'table', 'lamp', etc."},
                },
                "required": ["object_name"]
            }
        }
    }
]

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SAM3_WORKER = os.path.join(os.path.dirname(__file__), "sam3_worker.py")
SAM3D_WORKER = os.path.join(os.path.dirname(__file__), "sam3d_worker.py")

mcp = FastMCP("sam-bridge")
_target_image = _output_dir = _sam3_cfg = _blender_command = None
_sam3_env_bin = path_to_cmd["tools/sam3_worker.py"]
_sam3d_env_bin = path_to_cmd["tools/sam3d_worker.py"]


@mcp.tool()
def initialize(args: dict) -> dict:
    global _target_image, _output_dir, _sam3_cfg, _blender_command
    _target_image = args["target_image_path"]
    _output_dir = args.get("output_dir") + "/sam"
    os.makedirs(_output_dir, exist_ok=True)
    _sam3_cfg = args.get("sam3d_config_path") or os.path.join(
        ROOT, "utils", "sam3d", "checkpoints", "hf", "pipeline.yaml"
    )
    _blender_command = args.get("blender_command") or "utils/infinigen/blender/blender"
    return {"status": "success", "output": {"text": ["sam bridge init ok"], "tool_configs": tool_configs}}


@mcp.tool()
def get_better_object(object_name: str) -> dict:
    original_object_name = object_name
    object_name = object_name.replace(' ', '_')
    object_name = object_name.replace('-', '_')
    if not _target_image or not _output_dir:
        return {"status": "error", "output": {"text": ["call initialize first"]}}
    mask_path = os.path.join(_output_dir, f"{object_name}_mask.npy")
    glb_path = os.path.join(_output_dir, f"{object_name}.glb")
    
    try:
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
        # subprocess.run(
        #     [
        #         _blender_command,
        #         "-b",
        #         "-P",
        #         "tools/fix_glb.py",
        #         "--",
        #         glb_path,
        #         glb_path,
        #     ],
        #     cwd=ROOT,
        #     check=True,
        #     text=True,
        #     capture_output=True,
        # )
        info = json.loads(r2.stdout.strip().splitlines()[-1])
        info["glb_path"] = info.get("glb_path") or glb_path
        return {"status": "success", "output": {"text": [f"Successfully generated asset, downloaded to: {info['glb_path']}", f"Asset information: {json.dumps(info)}"]}}
    except subprocess.CalledProcessError as e:
      return {"status": "error", "output": {"text": [f"Object {object_name} not found in the image."]}}


def main():
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