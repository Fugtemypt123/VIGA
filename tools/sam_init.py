import os, sys, json, subprocess
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def process_single_object(args_tuple):
    """
    处理单个物体的 3D 重建任务（用于并行处理）
    
    Args:
        args_tuple: (idx, mask, object_name, _target_image, _output_dir, _sam3_cfg, 
                     _blender_command, _sam3d_env_bin, ROOT, SAM3D_WORKER)
    
    Returns:
        tuple: (success: bool, glb_path: str or None, object_transform: dict or None, error_msg: str or None)
    """
    idx, mask, object_name, _target_image, _output_dir, _sam3_cfg, _blender_command, _sam3d_env_bin, ROOT, SAM3D_WORKER = args_tuple
    
    try:
        # 使用 sam_worker.py 已经保存的 mask 文件（如果存在），否则保存新的
        mask_path = os.path.join(_output_dir, f"{object_name}.npy")
        if not os.path.exists(mask_path):
            # 如果文件不存在，保存 mask（这种情况不应该发生，但为了健壮性保留）
            np.save(mask_path, mask)
        else:
            print(f"[SAM_INIT] Using existing mask file: {mask_path}")
        
        # 重建 3D，使用相同的物体名称
        glb_path = os.path.join(_output_dir, f"{object_name}.glb")
        info_path = os.path.join(_output_dir, f"{object_name}.json")
        
        # 如果文件已存在（可能在之前的运行中生成），跳过重建
        if os.path.exists(glb_path) and os.path.exists(info_path):
            print(f"[SAM_INIT] GLB file already exists, skipping reconstruction: {glb_path}")
            with open(info_path, 'r') as f:
                info = json.load(f)
            return (True, glb_path, info, None)
        
        # 运行 SAM-3D 重建
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
        glb_path_value = info.get("glb_path")
        if glb_path_value:
            # 修复 GLB 文件的贴图问题
            # subprocess.run(
            #     [
            #         _blender_command,
            #         "-b",
            #         "-P",
            #         "tools/fix_glb.py",
            #         "--",
            #         glb_path_value,
            #         glb_path_value,
            #     ],
            #     cwd=ROOT,
            #     check=True,
            #     text=True,
            #     capture_output=True,
            # )
            # 构建变换信息
            object_transform = {
                "glb_path": glb_path_value,
                "translation": info.get("translation"),
                "translation_scale": info.get("translation_scale", 1.0),
                "rotation": info.get("rotation"),  # 四元数 [w, x, y, z]
                "scale": info.get("scale", 1.0),
            }
            # 保存 info 到 json
            with open(os.path.join(_output_dir, f"{object_name}.json"), 'w') as f:
                json.dump(info, f, indent=2)
            print(f"[SAM_INIT] Successfully reconstructed object {idx} ({object_name})")
            return (True, glb_path_value, object_transform, None)
        else:
            error_msg = f"Object {idx} ({object_name}) reconstruction failed or no GLB generated"
            print(f"[SAM_INIT] Warning: {error_msg}")
            return (False, None, None, error_msg)
            
    except subprocess.CalledProcessError as e:
        error_msg = f"Failed to reconstruct object {idx} ({object_name}): {e.stderr}"
        print(f"[SAM_INIT] Warning: {error_msg}")
        return (False, None, None, error_msg)
    except json.JSONDecodeError as e:
        error_msg = f"Failed to parse output for object {idx} ({object_name}): {str(e)}"
        print(f"[SAM_INIT] Warning: {error_msg}")
        return (False, None, None, error_msg)
    except Exception as e:
        error_msg = f"Unexpected error processing object {idx} ({object_name}): {str(e)}"
        print(f"[SAM_INIT] Error: {error_msg}")
        return (False, None, None, error_msg)


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
        
        print(f"[SAM_INIT] Step 2: Reconstructing {len(masks)} objects with SAM-3D (parallel processing)...")
        
        # 准备参数列表
        tasks = []
        for idx, mask in enumerate(masks):
            # 获取物体名称（如果可用，否则使用默认名称）
            if object_mapping and idx < len(object_mapping):
                object_name = object_mapping[idx]
            else:
                object_name = f"object_{idx}"
            
            tasks.append((
                idx, mask, object_name, _target_image, _output_dir, _sam3_cfg,
                _blender_command, _sam3d_env_bin, ROOT, SAM3D_WORKER
            ))
        
        # 使用线程池并行处理
        glb_paths = []
        object_transforms = []  # 存储每个物体的位置信息
        max_workers = min(2, len(tasks))  # 限制并发数，避免资源耗尽
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_idx = {executor.submit(process_single_object, task): task[0] for task in tasks}
            
            # 按完成顺序收集结果（保持顺序）
            results = [None] * len(tasks)
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    success, glb_path, object_transform, error_msg = future.result()
                    results[idx] = (success, glb_path, object_transform, error_msg)
                except Exception as e:
                    print(f"[SAM_INIT] Error processing object {idx}: {str(e)}")
                    results[idx] = (False, None, None, str(e))
        
        # 按原始顺序处理结果
        for idx, result in enumerate(results):
            if result is None:
                continue
            success, glb_path, object_transform, error_msg = result
            if success and glb_path and object_transform:
                glb_paths.append(glb_path)
                object_transforms.append(object_transform)
        
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
                "target_image_path": "data/static_scene/christmas1/target.png",
                "output_dir": os.path.join(ROOT, "output", "test", "christmas1"),
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