import os, sys, json, subprocess
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
IMPORT_SCRIPT = os.path.join(os.path.dirname(__file__), "import_glbs_to_blend.py")

mcp = FastMCP("sam-init")
_target_image = _output_dir = _sam3_cfg = _blender_command = None
_sam_env_bin = path_to_cmd["tools/sam_worker.py"]
_sam3d_env_bin = path_to_cmd.get("tools/sam3d_worker.py")


@mcp.tool()
def initialize(args: dict) -> dict:
    global _target_image, _output_dir, _sam3_cfg, _blender_command, _sam_env_bin
    _target_image = args["target_image_path"]
    _output_dir = args.get("output_dir") + "/sam_init"
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
        
        # Step 2: 加载 masks 并重建每个物体
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
        
        print(f"[SAM_INIT] Step 2: Reconstructing {len(masks)} objects with SAM-3D...")
        
        glb_paths = []
        object_transforms = []  # 存储每个物体的位置信息
        for idx, mask in enumerate(masks):
            # 保存单个 mask
            mask_path = os.path.join(_output_dir, f"mask_{idx}.npy")
            np.save(mask_path, mask)
            
            # 重建 3D
            glb_path = os.path.join(_output_dir, f"object_{idx}.glb")
            try:
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
                    ],
                    cwd=ROOT,
                    check=True,
                    text=True,
                    capture_output=True,
                )
                
                # 解析输出，检查是否成功生成 glb
                info = json.loads(r.stdout.strip().splitlines()[-1])
                glb_path_value = info.get("glb") or info.get("glb_path")
                if glb_path_value:
                    glb_paths.append(glb_path_value)
                    # 保存位置信息（新格式）
                    object_transforms.append({
                        "glb": glb_path_value,  # 使用 "glb" 键
                        "translation": info.get("translation"),
                        "translation_scale": info.get("translation_scale", 1.0),
                        "rotation": info.get("rotation"),  # 3x3 旋转矩阵
                        "scale": info.get("scale", 1.0),
                    })
                    print(f"[SAM_INIT] Successfully reconstructed object {idx}")
                else:
                    print(f"[SAM_INIT] Warning: Object {idx} reconstruction failed or no GLB generated")
            except subprocess.CalledProcessError as e:
                print(f"[SAM_INIT] Warning: Failed to reconstruct object {idx}: {e.stderr}")
                continue
            except json.JSONDecodeError:
                print(f"[SAM_INIT] Warning: Failed to parse output for object {idx}")
                continue
        
        if len(glb_paths) == 0:
            return {"status": "error", "output": {"text": ["No objects were successfully reconstructed"]}}
        
        # Step 3: 将所有 GLB 导入 Blender 并保存为 .blend 文件
        print(f"[SAM_INIT] Step 3: Importing {len(glb_paths)} objects into Blender...")
        blend_path = os.path.join(_output_dir, "scene.blend")
        
        # 保存位置信息到 JSON 文件
        transforms_json_path = os.path.join(_output_dir, "object_transforms.json")
        with open(transforms_json_path, 'w') as f:
            json.dump(object_transforms, f, indent=2)
        
        # 构建 Blender 命令
        blender_cmd = [
            _blender_command,
            "-b",  # 后台模式
            "-P",  # 运行 Python 脚本
            IMPORT_SCRIPT,
            "--",
            transforms_json_path,  # 传递位置信息 JSON 文件路径
            blend_path,  # 输出 .blend 文件路径
        ]
        
        subprocess.run(
            blender_cmd,
            cwd=ROOT,
            check=True,
            text=True,
            capture_output=True,
        )
        
        return {
            "status": "success",
            "output": {
                "text": [
                    f"Successfully reconstructed {len(glb_paths)} objects",
                    f"Blender scene saved to: {blend_path}",
                    f"Total masks detected: {len(masks)}"
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
        error_msg = e.stderr if e.stderr else str(e)
        return {"status": "error", "output": {"text": [f"Process failed: {error_msg}"]}}
    except Exception as e:
        return {"status": "error", "output": {"text": [f"Error: {str(e)}"]}}


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        initialize(
            {
                "target_image_path": "data/static_scene/christmas/target.png",
                "output_dir": os.path.join(ROOT, "output", "test", "sam_init"),
                "blender_command": "utils/infinigen/blender/blender",
            }
        )
        print(reconstruct_full_scene())
    else:
        mcp.run()


if __name__ == "__main__":
    main()

