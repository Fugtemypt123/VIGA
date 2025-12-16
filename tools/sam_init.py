import os, sys, json, subprocess, shutil
import numpy as np
from mcp.server.fastmcp import FastMCP
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.path import path_to_cmd

# tool_configs for agent (only the function w/ @mcp.tool)
tool_configs = [
    {
        "type": "function",
        "function": {
            "name": "reconstruct_full_scene",
            "description": "Reconstruct a complete 3D scene from an input image by detecting all objects with SAM and reconstructing each with SAM-3D. Outputs a .blend file containing all reconstructed objects.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SAM_WORKER = os.path.join(os.path.dirname(__file__), "sam_worker.py")
SAM3D_WORKER = os.path.join(os.path.dirname(__file__), "sam3d_worker.py")

mcp = FastMCP("sam-init")
_target_image = _output_dir = _sam3_cfg = _blender_command = None
_sam_env_bin = path_to_cmd["tools/sam_worker.py"]
_sam3d_env_bin = path_to_cmd.get("tools/sam3d_worker.py")


@mcp.tool()
def initialize(args: dict) -> dict:
    global _target_image, _output_dir, _sam3_cfg, _blender_command, _sam_env_bin
    _target_image = args["target_image_path"]
    _output_dir = args.get("output_dir") + "/sam_init"
    if os.path.exists(_output_dir):
        shutil.rmtree(_output_dir)
    os.makedirs(_output_dir, exist_ok=True)
    _sam3_cfg = args.get("sam3d_config_path") or os.path.join(
        ROOT, "utils", "sam3d", "checkpoints", "hf", "pipeline.yaml"
    )
    _blender_command = args.get("blender_command") or "utils/infinigen/blender/blender"
    
    # 尝试获取 sam_worker.py 的 python 路径
    # 如果没有配置，使用 sam3d 的环境（假设它们可能在同一环境）
    _sam_env_bin = path_to_cmd.get("tools/sam_worker.py") or _sam3d_env_bin
    
    return {"status": "success", "output": {"text": ["sam init initialized"], "tool_configs": tool_configs}}




@mcp.tool()
def reconstruct_full_scene() -> dict:
    """
    从输入图片重建完整的 3D 场景
    1. 使用 SAM 检测所有物体
    2. 对每个物体使用 SAM-3D 重建
    3. 将所有物体导入 Blender 并保存为 .blend 文件
    """
    global _target_image, _output_dir, _sam3_cfg, _blender_command, _sam_env_bin, _sam3d_env_bin
    
    if not _target_image or not _output_dir:
        return {"status": "error", "output": {"text": ["call initialize first"]}}
    
    if _sam_env_bin is None:
        return {"status": "error", "output": {"text": ["SAM worker python path not configured"]}}
    
    try:
        # Step 1: 使用 SAM 获取所有物体的 masks
        all_masks_path = os.path.join(_output_dir, "all_masks.npy")
        print(f"[SAM_INIT] Step 1: Detecting all objects with SAM...")
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
            capture_output=True,
        )
        
        # Step 2: 加载 masks 和物体名称映射
        masks = np.load(all_masks_path, allow_pickle=True)
        
        # 处理 masks 可能是 object array 的情况
        if masks.dtype == object:
            masks = [m for m in masks]
        elif masks.ndim == 3:
            # 如果是 3D 数组 (N, H, W)，转换为列表
            masks = [masks[i] for i in range(masks.shape[0])]
        else:
            # 单个 mask 的情况
            masks = [masks]
        
        # 加载物体名称映射信息
        object_names_json_path = all_masks_path.replace('.npy', '_object_names.json')
        object_mapping = None
        if os.path.exists(object_names_json_path):
            with open(object_names_json_path, 'r') as f:
                object_names_info = json.load(f)
                object_mapping = object_names_info.get("object_mapping", [])
                print(f"[SAM_INIT] Loaded object names mapping from: {object_names_json_path}")
        else:
            print(f"[SAM_INIT] Warning: Object names mapping file not found: {object_names_json_path}, using default names")
        
        print(f"[SAM_INIT] Step 2: Reconstructing {len(masks)} objects with SAM-3D and combining into scene...")
        
        # 保存所有 masks 到文件
        mask_paths = []
        object_names = []
        for idx, mask in enumerate(masks):
            # 获取物体名称（如果可用，否则使用默认名称）
            if object_mapping and idx < len(object_mapping):
                object_name = object_mapping[idx]
            else:
                object_name = f"object_{idx}"
            
            mask_path = os.path.join(_output_dir, f"{object_name}.npy")
            np.save(mask_path, mask)
            mask_paths.append(mask_path)
            object_names.append(object_name)
        
        # Step 3: 使用新的 sam3d_worker.py 一次性处理所有 masks 并生成 .blend 文件
        blend_path = os.path.join(_output_dir, "scene.blend")
        
        # 设置 Blender 命令环境变量（如果未设置）
        env = os.environ.copy()
        if _blender_command and "BLENDER_CMD" not in env:
            env["BLENDER_CMD"] = _blender_command
        
        # 运行 SAM-3D worker 处理所有 masks
        cmd = [
            _sam3d_env_bin,
            SAM3D_WORKER,
            "--image",
            _target_image,
            "--masks",
        ] + mask_paths + [
            "--config",
            _sam3_cfg,
            "--blend",
            blend_path,
        ]
        
        if object_names:
            cmd.extend(["--object-names"] + object_names)
        
        r = subprocess.run(
            cmd,
            cwd=ROOT,
            check=True,
            text=True,
            capture_output=True,
            env=env,
        )
        
        # 解析输出
        try:
            result = json.loads(r.stdout.strip().splitlines()[-1])
            if result.get("status") != "success":
                return {"status": "error", "output": {"text": [f"Failed to generate scene: {result.get('message', 'Unknown error')}"]}}
        except json.JSONDecodeError:
            # If output is not JSON, check if blend file was created
            if not os.path.exists(blend_path):
                return {"status": "error", "output": {"text": ["Failed to generate scene: No output received"]}}
        
        return {
            "status": "success",
            "output": {
                "text": [
                    f"Successfully reconstructed {len(masks)} objects",
                    f"Blender scene saved to: {blend_path}",
                    f"Total masks detected: {len(masks)}"
                ],
                "data": {
                    "blend_path": blend_path,
                    "num_objects": len(masks),
                    "num_masks": len(masks),
                }
            }
        }
    
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr if e.stderr else str(e)
        return {"status": "error", "output": {"text": [f"Process failed: {error_msg}"]}}
    except Exception as e:
        return {"status": "error", "output": {"text": [f"Error: {str(e)}"]}}


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        initialize(
            {
                "target_image_path": "data/static_scene/items19/target.png",
                "output_dir": os.path.join(ROOT, "output", "test", "items19"),
                "blender_command": "utils/infinigen/blender/blender",
            }
        )
        print(reconstruct_full_scene())
    else:
        mcp.run()


if __name__ == "__main__":
    main()

# python tools/sam_init.py --test
# utils/infinigen/blender/blender -b -P /mnt/data/users/shaofengyin/AgenticVerifier/tools/import_glbs_to_blend.py -- /mnt/data/users/shaofengyin/AgenticVerifier/output/test/sam_init/object_transforms.json /mnt/data/users/shaofengyin/AgenticVerifier/output/test/sam_init/scene.blend