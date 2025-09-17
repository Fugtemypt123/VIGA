# blender_executor_server.py
from optparse import Option
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
import bpy
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

# Global asset importer instance
_asset_importer = None

# Global meshy API instance
_meshy_api = None

# Global image cropper instance
_image_cropper = None

class Executor:
    def __init__(self,
                 blender_command: str,
                 blender_file: str,
                 blender_script: str,
                 script_save: str,
                 render_save: str,
                 blender_save: Optional[str] = None):
        self.blender_command = blender_command
        self.blender_file = blender_file
        self.blender_script = blender_script
        self.script_path = Path(script_save)
        self.render_path = Path(render_save)
        self.blend_path = blender_save

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
        try:
            proc = subprocess.run(cmd_str, shell=True, check=True, capture_output=True, text=True)
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
                'image_url': f"data:image/jpeg;base64,{image_base64}",
                'enable_pbr': True
            }
            
            # 发送请求（注意：这里不使用JSON headers，因为要上传文件）
            headers = {
                "Authorization": f"Bearer {self.api_key}"
            }
            
            resp = requests.post(url, headers=headers, json=files)
            resp.raise_for_status()
            data = resp.json()
            return data.get("result") or data.get("id")

    def create_image_to_3d_refine(self, preview_task_id: str, **kwargs) -> str:
        """
        基于 preview 发起 refine 贴图任务（Image-to-3D）
        Returns: refine_task_id (str)
        """
        url = f"{self.base_url}/openapi/v1/image-to-3d"
        payload = {
            "mode": "refine",
            "preview_task_id": preview_task_id,
        }
        payload.update(kwargs or {})
        resp = requests.post(url, headers=self.headers, data=json.dumps(payload))
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


class AssetImporter:
    """3D资产导入器，支持多种格式"""
    def __init__(self, blender_path: str):
        self.blender_path = blender_path

    def import_asset(self, asset_path: str, location: tuple = (0, 0, 0), scale: float = 1.0, name: str = "new_asset") -> str:
        """导入3D资产到Blender场景"""
        try:
            # 确保文件存在
            if not os.path.exists(asset_path):
                raise FileNotFoundError(f"Asset file not found: {asset_path}")

            # 根据文件扩展名选择导入方法
            ext = os.path.splitext(asset_path)[1].lower()

            if ext in ['.fbx', '.obj', '.gltf', '.glb', '.dae', '.3ds', '.blend']:
                # 使用Blender的导入操作符
                if ext == '.fbx':
                    bpy.ops.import_scene.fbx(filepath=asset_path)
                elif ext == '.obj':
                    bpy.ops.import_scene.obj(filepath=asset_path)
                elif ext in ['.gltf', '.glb']:
                    bpy.ops.import_scene.gltf(filepath=asset_path)
                elif ext == '.dae':
                    bpy.ops.wm.collada_import(filepath=asset_path)
                elif ext == '.3ds':
                    bpy.ops.import_scene.autodesk_3ds(filepath=asset_path)
                elif ext == '.blend':
                    # 附注：append 需要 directory + filename（指向 .blend 内部路径）
                    # 这里保留占位，以防未来确实需要 .blend 的 append
                    bpy.ops.wm.append(filepath=asset_path)

                # 获取导入的对象
                imported_objects = [obj for obj in bpy.context.selected_objects]
                if not imported_objects:
                    raise RuntimeError("No objects were imported")

                # 设置位置和缩放
                for obj in imported_objects:
                    obj.location = location
                    obj.scale = (scale, scale, scale)
                    
                # 将对象名称设置为name
                imported_objects[0].name = name

                # 返回导入的对象名称
                return imported_objects[0].name
            else:
                raise ValueError(f"Unsupported file format: {ext}")

        except Exception as e:
            logging.error(f"Failed to import asset: {e}")
            raise

    def extract_zip_asset(self, zip_path: str, extract_dir: str) -> str:
        """从ZIP文件中提取3D资产"""
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # 查找3D文件
                asset_files = []
                for file_info in zip_ref.filelist:
                    filename = file_info.filename
                    ext = os.path.splitext(filename)[1].lower()
                    if ext in ['.fbx', '.obj', '.gltf', '.glb', '.dae', '.3ds', '.blend']:
                        asset_files.append(filename)

                if not asset_files:
                    raise ValueError("No supported 3D files found in ZIP")

                # 提取第一个找到的3D文件
                asset_file = asset_files[0]
                zip_ref.extract(asset_file, extract_dir)

                return os.path.join(extract_dir, asset_file)

        except Exception as e:
            logging.error(f"Failed to extract ZIP asset: {e}")
            raise


def add_meshy_asset(
    object_name: str,
    description: str,
    location: str = "0,0,0",
    scale: float = 1.0,
    refine: bool = True,
    save_dir: str = "output/meshy_assets",
) -> dict:
    try:
        # 解析位置参数
        try:
            loc_parts = [float(x.strip()) for x in location.split(",")]
            if len(loc_parts) != 3:
                return {"status": "error", "error": "Location must be in format 'x,y,z'"}
            asset_location = tuple(loc_parts)
        except Exception:
            return {"status": "error", "error": "Invalid location format. Use 'x,y,z'"}

        # 初始化 Meshy API
        meshy = _meshy_api
        
        # 将save_path定义为_asset_importer.blender_path的父目录+'/assets'
        save_path = os.path.join(os.path.dirname(_asset_importer.blender_path), "assets")

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
        
        # save_dir是_asset_importer.blender_path的父目录+'/assets'
        save_dir = os.path.join(os.path.dirname(_asset_importer.blender_path), "assets")

        # 5) 下载模型到本地持久目录
        os.makedirs(save_dir, exist_ok=True)
        # 处理无扩展名直链：默认 .glb
        guessed_ext = os.path.splitext(file_url.split("?")[0])[1].lower()
        if guessed_ext not in [".glb", ".gltf", ".fbx", ".obj", ".zip"]:
            guessed_ext = ".glb"
        local_path = os.path.join(save_dir, f"{object_name}{guessed_ext}")
        print(f"[Meshy] Downloading model to: {local_path}")
        meshy.download_model_url(file_url, local_path)

        # 6) 若为 ZIP，解压出 3D 文件到保存目录下的同名子目录
        if _asset_importer:
            importer = _asset_importer
            if local_path.endswith(".zip"):
                extract_subdir = os.path.join(save_dir, object_name)
                os.makedirs(extract_subdir, exist_ok=True)
                extracted = importer.extract_zip_asset(local_path, extract_subdir)
                import_path = extracted
            else:
                import_path = local_path

            # 7) 导入 Blender
            imported_object_name = importer.import_asset(import_path, location=asset_location, scale=scale, name=object_name)
  
            print(f"[Meshy] Imported object: {imported_object_name}")

            # 8) 保存 Blender 文件
            try:
                blender_path = _asset_importer.blender_path
                bpy.ops.wm.save_mainfile(filepath=blender_path)
                print(f"Blender file saved to: {blender_path}")
                
                # 备份一份blender_path到save_dir
                blender_path_backup = os.path.join(save_dir, f"{object_name}.blend")
                shutil.copy(blender_path, blender_path_backup)
                
                # 清理备份文件以避免生成 .blend1 文件
                backup_file = blender_path + "1"
                if os.path.exists(backup_file):
                    os.remove(backup_file)
                    print(f"Removed backup file: {backup_file}")
                    
            except Exception as save_error:
                print(f"Warning: Failed to save blender file: {save_error}")

        return {
            "status": "success",
            "message": "Meshy Text-to-3D asset generated and imported",
            "asset_name": object_name,
            "object_name": imported_object_name,
            "location": asset_location,
            "scale": scale,
            "saved_model_path": import_path
        }

    except Exception as e:
        logging.error(f"Failed to add Meshy asset: {e}")
        return {"status": "error", "error": str(e)}


def add_meshy_asset_from_image(
    object_name: str,
    image_path: str,
    location: str = "0,0,0",
    scale: float = 1.0,
    prompt: str = None,
    refine: bool = True,
    save_dir: str = "output/meshy_assets",
) -> dict:
    """
    使用 Meshy Image-to-3D 根据输入图片生成资产并导入到当前场景（生成→轮询→下载→导入）

    Args:
        image_path: 输入图片路径
        blender_path: Blender 文件路径
        location: 资产位置 "x,y,z"
        scale: 缩放比例
        prompt: 可选的文本提示，用于指导生成
        api_key: Meshy API 密钥（可选，默认读 MESHY_API_KEY）
        refine: 是否在 preview 后进行 refine（含贴图）
    """
    try:
        # 检查图片文件是否存在
        if not os.path.exists(image_path):
            return {"status": "error", "error": f"Image file not found: {image_path}"}
        
        # 解析位置参数
        try:
            loc_parts = [float(x.strip()) for x in location.split(",")]
            if len(loc_parts) != 3:
                return {"status": "error", "error": "Location must be in format 'x,y,z'"}
            asset_location = tuple(loc_parts)
        except Exception:
            return {"status": "error", "error": "Invalid location format. Use 'x,y,z'"}

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

        # 3) 可选 refine（贴图）
        if refine:
            print(f"[Meshy] Starting refine for Image-to-3D preview task: {preview_id}")
            refine_id = meshy.create_image_to_3d_refine(preview_id)
            refine_task = meshy.poll_image_to_3d(refine_id, interval_sec=5, timeout_sec=1800)
            if refine_task.get("status") != "SUCCEEDED":
                return {"status": "error", "error": f"Image-to-3D refine failed: {refine_task.get('status')}"}
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
        safe_source = re.sub(r"[^a-zA-Z0-9_-]+", "_", os.path.splitext(os.path.basename(image_path))[0])[:60] or "image"
        base_name = object_name
        local_path = os.path.join(save_dir, f"{base_name}{guessed_ext}")
        print(f"[Meshy] Downloading Image-to-3D model to: {local_path}")
        meshy.download_model_url(file_url, local_path)

        # 6) 若为 ZIP，解压出 3D 文件到保存目录下的同名子目录
        if _asset_importer:
            importer = _asset_importer
            if local_path.endswith(".zip"):
                extract_subdir = os.path.join(save_dir, base_name)
                os.makedirs(extract_subdir, exist_ok=True)
                extracted = importer.extract_zip_asset(local_path, extract_subdir)
                import_path = extracted
            else:
                import_path = local_path

            # 7) 导入 Blender
            imported_object_name = importer.import_asset(import_path, location=asset_location, scale=scale, name=object_name)
            print(f"[Meshy] Imported Image-to-3D object: {imported_object_name}")

            # 8) 保存 Blender 文件
            try:
                blender_path = _asset_importer.blender_path
                bpy.ops.wm.save_mainfile(filepath=blender_path)
                print(f"Blender file saved to: {blender_path}")
                
                # 备份一份blender_path到save_dir
                blender_path_backup = os.path.join(save_dir, f"{object_name}.blend")
                shutil.copy(blender_path, blender_path_backup)
                
                # 清理备份文件以避免生成 .blend1 文件
                backup_file = os.path.join(save_dir, f"{object_name}.blend")
                if os.path.exists(backup_file):
                    os.remove(backup_file)
                    print(f"Removed backup file: {backup_file}")
                    
            except Exception as save_error:
                print(f"Warning: Failed to save blender file: {save_error}")

        return {
            "status": "success",
            "message": "Meshy Image-to-3D asset generated and imported",
            "asset_name": object_name,
            "image_path": image_path,
            "prompt": prompt,
            "location": asset_location,
            "scale": scale,
            "saved_model_path": local_path
        }
        
    except Exception as e:
        logging.error(f"Failed to add Meshy asset from image: {e}")
        return {"status": "error", "error": str(e)}


@mcp.tool()
def generate_and_import_3d_asset(
    object_name: str,
    reference_type: str,
    object_description: str = None,
) -> dict:
    if reference_type == "text":
        return add_meshy_asset(description=object_description)
    elif reference_type == "image":
        cropped_bbox = _image_cropper.crop_image_by_text(object_name=object_name)
        cropped_image = PIL.Image.open(_image_cropper.target_image_path).crop(cropped_bbox['bounding_box'])
        time_stamp = int(time.time())
        temp_save_path = f"output/temp/temp_cropped_image_{time_stamp}.png"
        cropped_image.save(temp_save_path)
        result = add_meshy_asset_from_image(image_path=temp_save_path, object_name=object_name)
        # delete the temp_save_path
        os.remove(temp_save_path)
        return result
    
@mcp.tool()
def initialize_executor(blender_command: str,
                       blender_file: str,
                       blender_script: str,
                       script_save: str,
                       render_save: str,
                       meshy_api_key: str,
                       va_api_key: str,
                       target_image_path: str,
                       blender_save: Optional[str] = None) -> dict:
    """
    初始化 Blender 执行器，设置所有必要的参数。
    """
    global _executor
    global _meshy_api
    global _asset_importer
    global _image_cropper
    try:
        _executor = Executor(
            blender_command=blender_command,
            blender_file=blender_file,
            blender_script=blender_script,
            script_save=script_save,
            render_save=render_save,
            blender_save=blender_save
        )
        _meshy_api = MeshyAPI(meshy_api_key)
        _asset_importer = AssetImporter(blender_file)
        _image_cropper = ImageCropper(va_api_key, target_image_path)
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
        # crop the image to [551.0, 711.0, 653.0, 830.0]
        import PIL.Image
        image_path = "data/blendergym_hard/level4/christmas1/renders/goal/visprompt1.png"
        output_image_path = "output/test_crop.png"
        image = PIL.Image.open(image_path)
        cropped_image = image.crop((551.0, 711.0, 653.0, 830.0))
        cropped_image.save(output_image_path)
        print(f"Cropped image saved to: {output_image_path}")
        exit(0)
        # test_cropper = ImageCropper(api_key=os.environ["VA_API_KEY"])
        # print(test_cropper.crop_image_by_text(image_path="data/blendergym_hard/level4/christmas1/renders/goal/visprompt1.png", object_name="snowman"))
    else:
        # 正常运行 MCP 服务
        mcp.run(transport="stdio")

if __name__ == "__main__":
    main()
