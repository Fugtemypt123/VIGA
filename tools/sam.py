import os, sys, json, subprocess
from mcp.server.fastmcp import FastMCP

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SAM3_WORKER = os.path.join(os.path.dirname(__file__), "sam3_worker.py")
SAM3D_WORKER = os.path.join(os.path.dirname(__file__), "sam3d_worker.py")

mcp = FastMCP("sam-bridge")
_target_image = _output_dir = _sam3_cfg = None
_sam3_env = "sam3"
_sam3d_env = "sam3d-objects"


@mcp.tool()
def initialize(args: dict) -> dict:
    global _target_image, _output_dir, _sam3_cfg, _sam3_env, _sam3d_env
    _target_image = args["target_image_path"]
    _output_dir = args.get("output_dir") or os.path.join(ROOT, "output", "sam_bridge")
    os.makedirs(_output_dir, exist_ok=True)
    _sam3_cfg = args.get("sam3d_config_path") or os.path.join(
        ROOT, "utils", "sam3d", "checkpoints", "hf", "pipeline.yaml"
    )
    _sam3_env = args.get("sam3_env_name") or _sam3_env
    _sam3d_env = args.get("sam3d_env_name") or _sam3d_env
    return {"status": "success", "output": {"text": ["sam bridge init ok"]}}


@mcp.tool()
def extract_3d_object(object_name: str) -> dict:
    if not _target_image or not _output_dir:
        return {"status": "error", "output": {"text": ["call initialize first"]}}
    mask_path = os.path.join(_output_dir, f"{object_name}_mask.npy")
    glb_path = os.path.join(_output_dir, f"{object_name}.glb")

    def _run(env, script, args):
        cmd = ["conda", "run", "-n", env, "python", script] + args
        return subprocess.run(cmd, check=True, text=True, capture_output=True)

    _run(
        _sam3_env,
        SAM3_WORKER,
        ["--image", _target_image, "--object", object_name, "--out", mask_path],
    )
    r2 = _run(
        _sam3d_env,
        SAM3D_WORKER,
        ["--image", _target_image, "--mask", mask_path, "--config", _sam3_cfg, "--glb", glb_path],
    )
    info = json.loads(r2.stdout.strip().splitlines()[-1])
    info["glb_path"] = info.get("glb_path") or glb_path
    return {"status": "success", "output": {"text": [f"glb: {info['glb_path']}"], "data": info}}


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        initialize(
            {
                "target_image_path": "data/static_scene/christmas/target.png",
                "output_dir": os.path.join(ROOT, "output", "sam_bridge_test"),
            }
        )
        print(extract_3d_object("snowman"))
    else:
        mcp.run()


if __name__ == "__main__":
    main()


