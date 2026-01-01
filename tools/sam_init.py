import os, sys, json, subprocess, shutil
import numpy as np
from mcp.server.fastmcp import FastMCP
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.path import path_to_cmd

# tool_configs for agent (only the function w/ @mcp.tool)
# For initialized tools, agent is not able to call them during the conversation
tool_configs = [
    # {
    #     "type": "function",
    #     "function": {
    #         "name": "reconstruct_full_scene",
    #         "description": "Reconstruct a complete 3D scene from an input image by detecting all objects with SAM and reconstructing each with SAM-3D. Outputs a .blend file containing all reconstructed objects.",
    #         "parameters": {
    #             "type": "object",
    #             "properties": {},
    #             "required": []
    #         }
    #     }
    # }
]

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SAM_WORKER = os.path.join(os.path.dirname(__file__), "sam_worker.py")
SAM3D_WORKER = os.path.join(os.path.dirname(__file__), "sam3d_worker.py")
IMPORT_SCRIPT = os.path.join(os.path.dirname(__file__), "import_glbs_to_blend.py")

mcp = FastMCP("sam-init")
_target_image = _output_dir = _sam3_cfg = _blender_command = _blender_file = None
_log_file = None  # 日志文件句柄

# 安全地获取路径，避免 KeyError 导致未捕获的异常
try:
    _sam_env_bin = path_to_cmd.get("tools/sam_worker.py")
    _sam3d_env_bin = path_to_cmd.get("tools/sam3d_worker.py")
except Exception as e:
    print(f"[SAM_INIT] Error initializing paths: {e}", file=sys.stderr)
    _sam_env_bin = None
    _sam3d_env_bin = None


def log(message: str):
    """
    同时输出到 stderr 和日志文件
    """
    # 输出到 stderr（用于实时查看）
    print(message, file=sys.stderr)
    # 输出到日志文件（用于后续查看）
    if _log_file is not None:
        try:
            _log_file.write(message + "\n")
            _log_file.flush()  # 立即刷新，确保内容写入
        except Exception:
            pass  # 如果日志文件写入失败，不影响主流程


def get_conda_prefix_from_python_path(python_path: str) -> str:
    """
    从 Python 可执行文件路径推断 CONDA_PREFIX
    例如: /path/to/envs/env_name/bin/python -> /path/to/envs/env_name
    """
    if not python_path:
        return None
    
    # 标准化路径（处理相对路径和绝对路径）
    if os.path.isabs(python_path):
        normalized_path = python_path
    else:
        normalized_path = os.path.abspath(python_path)
    
    # 方法1: 如果路径以 /bin/python 或 /bin/python3 结尾，去掉最后两级目录
    if normalized_path.endswith('/bin/python') or normalized_path.endswith('/bin/python3'):
        conda_prefix = os.path.dirname(os.path.dirname(normalized_path))
        return conda_prefix
    
    # 方法2: 如果路径包含 /envs/，提取环境路径
    if '/envs/' in normalized_path:
        parts = normalized_path.split('/envs/')
        if len(parts) == 2:
            env_part = parts[1].split('/')[0]
            conda_prefix = os.path.join(parts[0], 'envs', env_part)
            return conda_prefix
    
    # 方法3: 如果路径以 /bin/python 结尾（相对路径的情况）
    if '/bin/python' in normalized_path:
        idx = normalized_path.rfind('/bin/python')
        conda_prefix = normalized_path[:idx]
        return conda_prefix
    
    return None


def prepare_env_with_conda_prefix(python_path: str) -> dict:
    """
    准备环境变量字典，确保包含 CONDA_PREFIX
    总是从 Python 路径推断 CONDA_PREFIX，因为子进程可能运行在不同的 conda 环境中
    """
    env = os.environ.copy()
    
    # 总是从 Python 路径推断 CONDA_PREFIX，确保子进程使用正确的环境
    conda_prefix = get_conda_prefix_from_python_path(python_path)
    if conda_prefix:
        env["CONDA_PREFIX"] = conda_prefix
        log(f"[SAM_INIT] Set CONDA_PREFIX={conda_prefix} from Python path: {python_path}")
    else:
        # 如果无法推断，但当前环境有 CONDA_PREFIX，保留它
        if "CONDA_PREFIX" in env:
            log(f"[SAM_INIT] Could not infer CONDA_PREFIX from {python_path}, keeping existing: {env['CONDA_PREFIX']}")
        else:
            log(f"[SAM_INIT] Warning: Could not infer CONDA_PREFIX from {python_path} and no existing CONDA_PREFIX")
    
    return env


@mcp.tool()
def initialize(args: dict) -> dict:
    global _target_image, _output_dir, _sam3_cfg, _blender_command, _sam_env_bin, _blender_file, _log_file
    try:
        _target_image = args["target_image_path"]
        _output_dir = args.get("output_dir") + "/sam_init"
        if os.path.exists(_output_dir):
            shutil.rmtree(_output_dir)
        os.makedirs(_output_dir, exist_ok=True)
        
        # 初始化日志文件
        log_path = os.path.join(_output_dir, "sam_init.log")
        _log_file = open(log_path, 'w', encoding='utf-8')
        log(f"[SAM_INIT] Initialized. Log file: {log_path}")
        _sam3_cfg = args.get("sam3d_config_path") or os.path.join(
            ROOT, "utils", "sam3d", "checkpoints", "hf", "pipeline.yaml"
        )
        _blender_command = args.get("blender_command") or "utils/infinigen/blender/blender"
        # 记录传入的 blender_file 参数，用于后续重建时直接写入到该路径
        _blender_file = args.get("blender_file")
        
        # 尝试获取 sam_worker.py 的 python 路径
        # 如果没有配置，使用 sam3d 的环境（假设它们可能在同一环境）
        _sam_env_bin = path_to_cmd.get("tools/sam_worker.py") or _sam3d_env_bin
        
        log("[SAM_INIT] sam init initialized")
        return {"status": "success", "output": {"text": ["sam init initialized"], "tool_configs": tool_configs}}
    except Exception as e:
        error_msg = f"Error in initialize: {str(e)}"
        if _log_file:
            log(f"[SAM_INIT] {error_msg}")
        else:
            print(f"[SAM_INIT] {error_msg}", file=sys.stderr)
        return {"status": "error", "output": {"text": [error_msg]}}


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
            log(f"[SAM_INIT] Using existing mask file: {mask_path}")
        
        # 重建 3D，使用相同的物体名称
        glb_path = os.path.join(_output_dir, f"{object_name}.glb")
        info_path = os.path.join(_output_dir, f"{object_name}.json")
        
        # 如果文件已存在（可能在之前的运行中生成），跳过重建
        if os.path.exists(glb_path) and os.path.exists(info_path):
            log(f"[SAM_INIT] GLB file already exists, skipping reconstruction: {glb_path}")
            with open(info_path, 'r') as f:
                info = json.load(f)
            return (True, glb_path, info, None)
        
        # 运行 SAM-3D 重建，使用 --info 参数将 JSON 输出写入文件而不是 stdout
        # 将 subprocess 的输出重定向到日志文件，避免污染 stdout
        log_path = os.path.join(_output_dir, f"{object_name}_sam3d.log")
        # 准备环境变量，确保包含 CONDA_PREFIX（从 Python 路径推断）
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
                    info_path,  # 指定 JSON 输出文件路径
                ],
                cwd=ROOT,
                check=True,
                text=True,
                stdout=log_file,
                stderr=subprocess.STDOUT,  # 将 stderr 也重定向到 stdout
                env=env,  # 传递环境变量
            )
        
        # 从文件读取 JSON 输出，而不是从 stdout 解析（避免 stdout 污染）
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
        # 读取日志文件内容以获取详细错误信息
        log_path = os.path.join(_output_dir, f"{object_name}_sam3d.log")
        log_content = ""
        if os.path.exists(log_path):
            try:
                with open(log_path, 'r') as f:
                    log_content = f.read()
                    # 只取最后 500 个字符，避免日志过长
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
def reconstruct_full_scene() -> dict:
    """
    从输入图片重建完整的 3D 场景
    1. 使用 SAM 检测所有物体
    2. 对每个物体使用 SAM-3D 重建
    3. 将所有物体导入 Blender 并保存为 .blend 文件
    """
    global _target_image, _output_dir, _sam3_cfg, _blender_command, _sam_env_bin, _sam3d_env_bin, _blender_file
    
    if not _target_image or not _output_dir:
        return {"status": "error", "output": {"text": ["call initialize first"]}}
    
    if _sam_env_bin is None:
        return {"status": "error", "output": {"text": ["SAM worker python path not configured"]}}
    
    try:
        # Step 1: 使用 SAM 获取所有物体的 masks
        all_masks_path = os.path.join(_output_dir, "all_masks.npy")
        sam_log_path = os.path.join(_output_dir, "sam_worker.log")
        log(f"[SAM_INIT] Step 1: Detecting all objects with SAM...")
        log(f"[SAM_INIT] SAM worker output will be saved to: {sam_log_path}")
        # 准备环境变量，确保包含 CONDA_PREFIX（从 Python 路径推断）
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
                stderr=subprocess.STDOUT,  # 将 stderr 也重定向到 stdout
                env=env,  # 传递环境变量
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
                log(f"[SAM_INIT] Loaded object names mapping from: {object_names_json_path}")
        else:
            log(f"[SAM_INIT] Warning: Object names mapping file not found: {object_names_json_path}, using default names")
        
        log(f"[SAM_INIT] Step 2: Reconstructing {len(masks)} objects with SAM-3D (parallel processing)...")
        
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
        
        # 串行处理所有任务
        glb_paths = []
        object_transforms = []  # 存储每个物体的位置信息
        
        # 按顺序处理每个任务
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
        
        # Step 3: 将所有 GLB 导入 Blender 并保存为 .blend 文件
        log(f"[SAM_INIT] Step 3: Importing {len(glb_paths)} objects into Blender...")
        # 如果在 initialize 中提供了 blender_file，则直接写入到该路径
        # 否则默认写入到输出目录下的 scene.blend
        blend_path = _blender_file or os.path.join(_output_dir, "scene.blend")
        
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
        
        # 将可能多出的.blend1文件删除
        for file in os.listdir(_output_dir):
            if file.endswith(".blend1"):
                os.remove(os.path.join(_output_dir, file))
        
        blender_log_path = os.path.join(_output_dir, "blender_import.log")
        log(f"[SAM_INIT] Blender import output will be saved to: {blender_log_path}")
        # 对于 Blender，使用当前环境的环境变量（Blender 通常不需要 CONDA_PREFIX）
        env = os.environ.copy()
        with open(blender_log_path, 'w') as log_file:
            subprocess.run(
                blender_cmd,
                cwd=ROOT,
                check=True,
                text=True,
                stdout=log_file,
                stderr=subprocess.STDOUT,  # 将 stderr 也重定向到 stdout
                env=env,  # 传递环境变量
            )
        
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
        # 尝试读取相关日志文件内容
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
        import traceback
        error_msg = f"Error: {str(e)}"
        log(f"[SAM_INIT] {error_msg}")
        log(f"[SAM_INIT] Traceback: {traceback.format_exc()}")
        return {"status": "error", "output": {"text": [error_msg]}}


def main():
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

# python tools/sam_init.py --test
# utils/infinigen/blender/blender -b -P /mnt/data/users/shaofengyin/AgenticVerifier/tools/import_glbs_to_blend.py -- /mnt/data/users/shaofengyin/AgenticVerifier/output/test/sam_init/object_transforms.json /mnt/data/users/shaofengyin/AgenticVerifier/output/test/sam_init/scene.blend