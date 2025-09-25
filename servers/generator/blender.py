# blender_executor_server.py
import os
import subprocess
import base64
import io
from typing import Optional
from pathlib import Path
from PIL import Image
import logging
from typing import Tuple, Dict
from mcp.server.fastmcp import FastMCP
import json
import requests
import tempfile
import zipfile
import shutil
import math
import cv2
import numpy as np
import time
from openai import OpenAI
import re
import PIL

mcp = FastMCP("blender-executor")

# Global executor instance
_executor = None

# Global meshy API instance
_meshy_api = None

# Global image cropper instance
_image_cropper = None

# Global save_dir
_save_dir = None

class Executor:
    def __init__(self,
                 blender_command: str,
                 blender_file: str,
                 blender_script: str,
                 script_save: str,
                 render_save: str,
                 blender_save: Optional[str] = None,
                 gpu_devices: Optional[str] = None):
        self.blender_command = blender_command
        self.blender_file = blender_file
        self.blender_script = blender_script
        self.script_path = Path(script_save)
        self.render_path = Path(render_save)
        self.blend_path = blender_save
        self.gpu_devices = gpu_devices  # 例如: "0,1" 或 "0"

        self.script_path.mkdir(parents=True, exist_ok=True)
        self.render_path.mkdir(parents=True, exist_ok=True)

    def _execute_blender(self, script_path: str, render_path: str) -> Tuple[bool, str, str]:
        cmd = [
            self.blender_command,
            "--background", self.blender_file,
            "--python", self.blender_script,
            "--", script_path, render_path
        ]
        if self.blend_path:
            cmd.append(self.blend_path)
        cmd_str = " ".join(cmd)
        
        # 设置环境变量以控制GPU设备
        env = os.environ.copy()
        if self.gpu_devices:
            env['CUDA_VISIBLE_DEVICES'] = self.gpu_devices
            logging.info(f"Setting CUDA_VISIBLE_DEVICES to: {self.gpu_devices}")
        
        try:
            proc = subprocess.run(cmd_str, shell=True, check=True, capture_output=True, text=True, env=env)
            out = proc.stdout
            err = proc.stderr
            # We do not consider intermediate errors that do not affect the result.
            # if 'Error:' in out:
            #     logging.error(f"Error in Blender stdout: {out}")
            #     return False, err, out
            # find rendered image(s)
            if os.path.isdir(render_path):
                imgs = sorted([str(p) for p in Path(render_path).glob("*") if p.suffix in ['.png','.jpg']])
                if not imgs:
                    return False, "No images", out
                paths = imgs[:2]
                return True, "PATH:" + ",".join(paths), out
            return True, out, err
        except subprocess.CalledProcessError as e:
            logging.error(f"Blender failed: {e}")
            return False, e.stderr, e.stdout

    def _encode_image(self, img_path: str) -> str:
        img = Image.open(img_path)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    def execute(self, code: str, round: int) -> Dict:
        script_file = self.script_path / f"{round}.py"
        render_file = self.render_path / f"{round}"
        with open(script_file, "w") as f:
            f.write(code)
        success, stdout, stderr = self._execute_blender(str(script_file), str(render_file))
        if not success or not os.path.exists(render_file):
            return {"status": "failure", "output": stderr or stdout}
        return {"status": "success", "output": str(render_file), "stdout": stdout, "stderr": stderr}

# ======================
# Meshy API（从scene.py迁移）
# ======================

class MeshyAPI:
    """Meshy API 客户端：Text-to-3D 生成 + 轮询 + 下载"""
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("MESHY_API_KEY")
        if not self.api_key:
            raise ValueError("Meshy API key is required. Set MESHY_API_KEY environment variable or pass api_key parameter.")
        self.base_url = "https://api.meshy.ai"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
        }

    def create_text_to_3d_preview(self, prompt: str, **kwargs) -> str:
        """
        创建 Text-to-3D 预览任务（无贴图）
        Returns: task_id (str)
        """
        url = f"{self.base_url}/openapi/v2/text-to-3d"
        payload = {
            "mode": "preview",
            "prompt": prompt[:600],
        }
        payload.update(kwargs or {})
        resp = requests.post(url, headers=self.headers, data=json.dumps(payload))
        resp.raise_for_status()
        data = resp.json()
        # 有的环境返回 {"result": "<id>"}，有的返回 {"id": "<id>"}
        return data.get("result") or data.get("id")

    def poll_text_to_3d(self, task_id: str, interval_sec: float = 5.0, timeout_sec: int = 1800) -> dict:
        """
        轮询 Text-to-3D 任务直到结束
        Returns: 任务 JSON（包含 status / model_urls 等）
        """
        import time
        url = f"{self.base_url}/openapi/v2/text-to-3d/{task_id}"
        deadline = time.time() + timeout_sec
        while True:
            r = requests.get(url, headers=self.headers)
            r.raise_for_status()
            js = r.json()
            status = js.get("status")
            if status in ("SUCCEEDED", "FAILED", "CANCELED"):
                return js
            if time.time() > deadline:
                raise TimeoutError(f"Meshy task {task_id} polling timeout")
            time.sleep(interval_sec)

    def create_text_to_3d_refine(self, preview_task_id: str, **kwargs) -> str:
        """
        基于 preview 发起 refine 贴图任务
        Returns: refine_task_id (str)
        """
        url = f"{self.base_url}/openapi/v2/text-to-3d"
        payload = {
            "mode": "refine",
            "preview_task_id": preview_task_id,
        }
        payload.update(kwargs or {})
        resp = requests.post(url, headers=self.headers, data=json.dumps(payload))
        resp.raise_for_status()
        data = resp.json()
        return data.get("result") or data.get("id")

    def download_model_url(self, file_url: str, output_path: str) -> None:
        """
        从 model_urls 的直链下载文件到本地
        """
        r = requests.get(file_url, stream=True)
        r.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    def create_image_to_3d_preview(self, image_path: str, prompt: str = None, **kwargs) -> str:
        """
        创建 Image-to-3D 预览任务（无贴图）
        
        Args:
            image_path: 输入图片路径
            prompt: 可选的文本提示
            **kwargs: 其他参数
            
        Returns: task_id (str)
        """
        url = f"{self.base_url}/openapi/v1/image-to-3d"
        
        # 准备文件上传
        with open(image_path, 'rb') as f:
            # 将image转为base64格式
            image_base64 = base64.b64encode(f.read()).decode('utf-8')
            files = {
                'image_url': f"data:image/png;base64,{image_base64}",
                'enable_pbr': True,
            }
            
            # 发送请求（注意：这里不使用JSON headers，因为要上传文件）
            headers = {
                "Authorization": f"Bearer {self.api_key}"
            }
            
            resp = requests.post(url, headers=headers, json=files)
            resp.raise_for_status()
            data = resp.json()
            return data.get("result") or data.get("id")

    def poll_image_to_3d(self, task_id: str, interval_sec: float = 5.0, timeout_sec: int = 1800) -> dict:
        """
        轮询 Image-to-3D 任务直到结束
        Returns: 任务 JSON（包含 status / model_urls 等）
        """
        import time
        url = f"{self.base_url}/openapi/v1/image-to-3d/{task_id}"
        deadline = time.time() + timeout_sec
        while True:
            r = requests.get(url, headers=self.headers)
            r.raise_for_status()
            js = r.json()
            status = js.get("status")
            if status in ("SUCCEEDED", "FAILED", "CANCELED"):
                return js
            if time.time() > deadline:
                raise TimeoutError(f"Meshy Image-to-3D task {task_id} polling timeout")
            time.sleep(interval_sec)


# ======================
# 图片截取工具
# ======================

class ImageCropper:
    """图片截取工具，支持基于文本描述的智能截取（Grounding DINO + SAM / YOLO / OpenAI 兜底）"""

    def __init__(self, api_key: str, target_image_path: str):
        self.url = "https://api.va.landing.ai/v1/tools/agentic-object-detection"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
        }
        self.target_image_path = target_image_path

    # ---------------------------
    # 对外：裁剪（签名与返回结构保留）
    # ---------------------------
    def crop_image_by_text(self, object_name: str) -> dict:
        files = {
            "image": open(self.target_image_path, "rb")
        }
        data = {
            "prompts": object_name,
            "model": "agentic"
        }
        response = requests.post(self.url, files=files, data=data, headers=self.headers)
        return response.json()


def download_meshy_asset(
    object_name: str,
    description: str,
    save_dir: str = "assets",
    refine: bool = True,
) -> dict:
    try:
        # 初始化 Meshy API
        meshy = _meshy_api

        # 1) 创建 preview 任务
        print(f"[Meshy] Creating preview task for: {description}")
        preview_id = meshy.create_text_to_3d_preview(description)

        # 2) 轮询 preview
        preview_task = meshy.poll_text_to_3d(preview_id, interval_sec=5, timeout_sec=900)
        if preview_task.get("status") != "SUCCEEDED":
            return {"status": "error", "error": f"Preview failed: {preview_task.get('status')}"}
        final_task = preview_task

        # 3) 可选 refine（贴图）
        if refine:
            print(f"[Meshy] Starting refine for preview task: {preview_id}")
            refine_id = meshy.create_text_to_3d_refine(preview_id)
            refine_task = meshy.poll_text_to_3d(refine_id, interval_sec=5, timeout_sec=1800)
            if refine_task.get("status") != "SUCCEEDED":
                return {"status": "error", "error": f"Refine failed: {refine_task.get('status')}"}
            final_task = refine_task

        # 4) 从 model_urls 取下载链接
        model_urls = (final_task or {}).get("model_urls", {}) or {}
        candidate_keys = ["glb", "fbx", "obj", "zip"]
        file_url = None
        for k in candidate_keys:
            if model_urls.get(k):
                file_url = model_urls[k]
                break
        if not file_url:
            return {"status": "error", "error": "No downloadable model_urls found"}
        
        # 5) 下载模型到本地持久目录
        os.makedirs(save_dir, exist_ok=True)
        # 处理无扩展名直链：默认 .glb
        guessed_ext = os.path.splitext(file_url.split("?")[0])[1].lower()
        if guessed_ext not in [".glb", ".gltf", ".fbx", ".obj", ".zip"]:
            guessed_ext = ".glb"
        local_path = os.path.join(save_dir, f"{object_name}{guessed_ext}")
        print(f"[Meshy] Downloading model to: {local_path}")
        meshy.download_model_url(file_url, local_path)
        
        return {
            'status': 'success',
            'message': f'Meshy Text-to-3D asset downloaded to {local_path}',
            'object_name': object_name,
            'local_path': local_path,
            'save_dir': save_dir
        }

    except Exception as e:
        logging.error(f"Failed to download Meshy asset: {e}")
        return {"status": "error", "error": str(e)}


def download_meshy_asset_from_image(
    object_name: str,
    image_path: str,
    save_dir: str = "assets",
    prompt: str = None,
) -> dict:
    """
    使用 Meshy Image-to-3D 根据输入图片生成资产并下载到本地（生成→轮询→下载）

    Args:
        object_name: 对象名称
        image_path: 输入图片路径
        save_dir: 保存目录
        prompt: 可选的文本提示，用于指导生成
    """
    try:
        # 检查图片文件是否存在
        if not os.path.exists(image_path):
            return {"status": "error", "error": f"Image file not found: {image_path}"}

        # 初始化 Meshy API
        meshy = _meshy_api

        # 1) 创建 Image-to-3D preview 任务
        print(f"[Meshy] Creating Image-to-3D preview task for: {image_path}")
        if prompt:
            print(f"[Meshy] Using prompt: {prompt}")
        
        preview_id = meshy.create_image_to_3d_preview(image_path, prompt)

        # 2) 轮询 preview
        preview_task = meshy.poll_image_to_3d(preview_id, interval_sec=5, timeout_sec=900)
        if preview_task.get("status") != "SUCCEEDED":
            return {"status": "error", "error": f"Image-to-3D preview failed: {preview_task.get('status')}"}
        final_task = preview_task

        # 3) 从 model_urls 取下载链接
        model_urls = (final_task or {}).get("model_urls", {}) or {}
        candidate_keys = ["glb", "fbx", "obj", "zip"]
        file_url = None
        for k in candidate_keys:
            if model_urls.get(k):
                file_url = model_urls[k]
                break
        if not file_url:
            return {"status": "error", "error": "No downloadable model_urls found"}

        # 4) 下载模型到本地持久目录
        os.makedirs(save_dir, exist_ok=True)
        # 处理无扩展名直链：默认 .glb
        guessed_ext = os.path.splitext(file_url.split("?")[0])[1].lower()
        if guessed_ext not in [".glb", ".gltf", ".fbx", ".obj", ".zip"]:
            guessed_ext = ".glb"
        local_path = os.path.join(save_dir, f"{object_name}{guessed_ext}")
        print(f"[Meshy] Downloading Image-to-3D model to: {local_path}")
        meshy.download_model_url(file_url, local_path)

        return {
            'status': 'success',
            'message': f'Meshy Image-to-3D asset downloaded to {local_path}',
            'object_name': object_name,
            'local_path': local_path,
            'save_dir': save_dir
        }
        
    except Exception as e:
        logging.error(f"Failed to download Meshy asset from image: {e}")
        return {"status": "error", "error": str(e)}


@mcp.tool()
def generate_and_download_3d_asset(
    object_name: str,
    reference_type: str,
    object_description: str = None,
) -> dict:
    if reference_type == "text":
        generate_result = download_meshy_asset(object_name=object_name, description=object_description, save_dir=_save_dir)
    elif reference_type == "image":
        cropped_bbox = _image_cropper.crop_image_by_text(object_name=object_name)
        cropped_bbox = cropped_bbox['data'][0][0]['bounding_box']
        cropped_image = PIL.Image.open(_image_cropper.target_image_path).crop(cropped_bbox)
        save_path = os.path.join(_save_dir, f"cropped_{object_name}.png")
        os.makedirs(_save_dir, exist_ok=True)
        cropped_image.save(save_path)
        generate_result = download_meshy_asset_from_image(image_path=save_path, object_name=object_name, save_dir=_save_dir)
    
    return generate_result
    
@mcp.tool()
def initialize_executor(args: dict) -> dict:
    """
    初始化 Blender 执行器，设置所有必要的参数。
    
    Args:
        args: 包含以下键的字典:
            - blender_command: Blender可执行文件路径
            - blender_file: Blender文件路径
            - blender_script: Blender脚本路径
            - script_save: 脚本保存目录
            - render_save: 渲染结果保存目录
            - blender_save: Blender文件保存路径（可选）
            - gpu_devices: GPU设备ID，如"0"或"0,1"（可选）
            - meshy_api_key: Meshy API密钥（可选）
            - va_api_key: VA API密钥（可选）
            - target_image_path: 目标图片路径（可选）
    """
    global _executor
    global _meshy_api
    global _image_cropper
    global _save_dir
    try:
        _executor = Executor(
            blender_command=args.get("blender_command"),
            blender_file=args.get("blender_file"),
            blender_script=args.get("blender_script"),
            script_save=args.get("script_save"),
            render_save=args.get("render_save"),
            blender_save=args.get("blender_save"),
            gpu_devices=args.get("gpu_devices")
        )
        _save_dir = os.path.dirname(args.get("script_save")) + '/assets'
        if args.get("meshy_api_key"):
            _meshy_api = MeshyAPI(args.get("meshy_api_key"))
        if args.get("va_api_key") and args.get("target_image_path"):
            _image_cropper = ImageCropper(args.get("va_api_key"), args.get("target_image_path"))
        return {"status": "success", "message": "Executor initialized successfully"}
    except Exception as e:
        return {"status": "error", "error": str(e)}

@mcp.tool()
def exec_script(code: str, round: int) -> dict:
    """
    执行传入的 Blender Python 脚本 code，并返回 base64 编码后的渲染图像。
    需要先调用 initialize_executor 进行初始化。
    """
    global _executor
    if _executor is None:
        return {"status": "error", "error": "Executor not initialized. Call initialize_executor first."}
    
    try:
        result = _executor.execute(code, round)
        return {"status": "success", "result": result}
    except Exception as e:
        return {"status": "error", "error": str(e)}
    

def main():
    # 如果直接运行此脚本，执行测试
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        # 测试：先初始化执行器，再调用生成与导入资产接口
        meshy_api_key = os.getenv("MESHY_API_KEY")
        va_api_key = os.getenv("VA_API_KEY")
        if not meshy_api_key or not va_api_key:
            print("[TEST] Missing MESHY_API_KEY or VA_API_KEY in environment. Skipping online API test.")
            print("Set both environment variables to run this test.")
            sys.exit(0)

        blender_command = "utils/blender/infinigen/blender/blender"
        blender_file = "data/blendergym_hard/level4/christmas1/blender_file_empty.blend"
        blender_script = "data/blendergym_hard/level4/christmas1/pipeline_render_script.py"
        script_save = "output/test/christmas1/scripts"
        render_save = "output/test/christmas1/renders"
        target_image_path = "data/blendergym_hard/level4/christmas1/renders/goal/visprompt1.png"
        save_dir = os.path.dirname(script_save) + '/assets'
        os.makedirs(script_save, exist_ok=True)
        os.makedirs(render_save, exist_ok=True)

        print("[TEST] Initializing executor...")
        init_resp = initialize_executor(
            blender_command=blender_command,
            blender_file=blender_file,
            blender_script=blender_script,
            script_save=script_save,
            render_save=render_save,
            meshy_api_key=meshy_api_key,
            va_api_key=va_api_key,
            target_image_path=target_image_path,
        )
        print(f"[TEST] initialize_executor response: {init_resp}")
        if init_resp.get("status") != "success":
            print("[TEST] initialize_executor failed. Abort.")
            sys.exit(1)

        print("[TEST] Calling generate_and_download_3d_asset (text reference)...")
        # 使用文本参考分支
        gen_resp = generate_and_download_3d_asset(
            object_name="snowman",
            reference_type="text",
            object_description="A white snowman with a black hat and a red scarf.",
            save_dir=_save_dir
        )
        print(f"[TEST] generate_and_download_3d_asset response: {gen_resp}")
        sys.exit(0)
    else:
        # 正常运行 MCP 服务
        mcp.run(transport="stdio")

if __name__ == "__main__":
    main()


# # 首先检查本地assets目录是否有匹配的文件
# if os.path.exists(assets_dir):
#     for asset_file in os.listdir(assets_dir):
#         # 模糊匹配：将object_name和asset_file中的空格移除，转换为小写，判断是否互相包含
#         new_object_name = object_name.replace(" ", "")
#         new_asset_file = asset_file.replace(" ", "")
#         new_asset_file = new_asset_file.split(".")[0]
#         if new_object_name.lower() in new_asset_file.lower() or new_asset_file.lower() in new_object_name.lower():
#             if asset_file.endswith('.glb') or asset_file.endswith('.obj'):
#                 generate_result = {
#                     'status': 'success',
#                     'message': 'Local asset found',
#                     'object_name': object_name,
#                     'local_path': os.path.join(assets_dir, asset_file),
#                     'save_dir': save_dir
#                 }
#                 break
#         elif os.path.isdir(os.path.join(assets_dir, asset_file)):
#             for asset_file_ in os.listdir(os.path.join(assets_dir, asset_file)):
#                 if object_name.lower() in asset_file_.lower() or asset_file_.lower() in object_name.lower():
#                     if asset_file_.endswith('.glb') or asset_file_.endswith('.obj'):
#                         generate_result = {
#                             'status': 'success',
#                             'message': 'Local asset found',
#                             'object_name': object_name,
#                             'local_path': os.path.join(assets_dir, asset_file, asset_file_),
#                             'save_dir': save_dir
#                         }
#                         break
#             if generate_result is not None:
#                 break