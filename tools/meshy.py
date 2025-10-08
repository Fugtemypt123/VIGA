# meshy.py
import os
import base64
import json
import requests
import time
import logging
import PIL
from typing import Optional
from mcp.server.fastmcp import FastMCP


mcp = FastMCP("meshy-executor")
_image_cropper = None
_meshy_api = None

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

    def create_rigging_task(self, model_url: str, target_formats: list = None, topology: str = "triangle", target_polycount: int = 30000) -> str:
        """
        创建自动绑定任务
        
        Args:
            model_url: 需要绑定的3D模型的URL
            target_formats: 目标格式列表，默认 ["glb"]
            topology: 拓扑结构，默认 "triangle"
            target_polycount: 目标多边形数量，默认 30000
            
        Returns: task_id (str)
        """
        url = f"{self.base_url}/openapi/v1/rigging"
        payload = {
            "model_url": model_url,
            "target_formats": target_formats or ["glb"],
            "topology": topology,
            "target_polycount": target_polycount
        }
        resp = requests.post(url, headers=self.headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data.get("result") or data.get("id")

    def poll_rigging_task(self, task_id: str, interval_sec: float = 5.0, timeout_sec: int = 1800) -> dict:
        """
        轮询绑定任务直到结束
        
        Args:
            task_id: 任务ID
            interval_sec: 轮询间隔（秒）
            timeout_sec: 超时时间（秒）
            
        Returns: 任务 JSON（包含 status / result 等）
        """
        url = f"{self.base_url}/openapi/v1/rigging/{task_id}"
        deadline = time.time() + timeout_sec
        while True:
            r = requests.get(url, headers=self.headers)
            r.raise_for_status()
            js = r.json()
            status = js.get("status")
            if status in ("SUCCEEDED", "FAILED", "CANCELED"):
                return js
            if time.time() > deadline:
                raise TimeoutError(f"Meshy rigging task {task_id} polling timeout")
            time.sleep(interval_sec)

    def create_animation_task(self, rig_task_id: str, action_id: int = 92, post_process: dict = None) -> str:
        """
        创建动画任务
        
        Args:
            rig_task_id: 成功完成的绑定任务的ID
            action_id: 要应用的动画动作的标识符，默认 92
            post_process: 动画文件的后处理参数
            
        Returns: task_id (str)
        """
        url = f"{self.base_url}/openapi/v1/animations"
        payload = {
            "rig_task_id": rig_task_id,
            "action_id": action_id
        }
        if post_process:
            payload["post_process"] = post_process
            
        resp = requests.post(url, headers=self.headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data.get("result") or data.get("id")

    def poll_animation_task(self, task_id: str, interval_sec: float = 5.0, timeout_sec: int = 1800) -> dict:
        """
        轮询动画任务直到结束
        
        Args:
            task_id: 任务ID
            interval_sec: 轮询间隔（秒）
            timeout_sec: 超时时间（秒）
            
        Returns: 任务 JSON（包含 status / result 等）
        """
        url = f"{self.base_url}/openapi/v1/animations/{task_id}"
        deadline = time.time() + timeout_sec
        while True:
            r = requests.get(url, headers=self.headers)
            r.raise_for_status()
            js = r.json()
            status = js.get("status")
            if status in ("SUCCEEDED", "FAILED", "CANCELED"):
                return js
            if time.time() > deadline:
                raise TimeoutError(f"Meshy animation task {task_id} polling timeout")
            time.sleep(interval_sec)


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
    meshy_api: MeshyAPI = None,
) -> dict:
    """
    下载 Meshy Text-to-3D 资产
    
    Args:
        object_name: 对象名称
        description: 对象描述
        save_dir: 保存目录
        refine: 是否进行贴图细化
        meshy_api: MeshyAPI 实例，如果为 None 则创建新实例
    """
    try:
        # 初始化 Meshy API
        if meshy_api is None:
            meshy_api = MeshyAPI()

        # 1) 创建 preview 任务
        print(f"[Meshy] Creating preview task for: {description}")
        preview_id = meshy_api.create_text_to_3d_preview(description)

        # 2) 轮询 preview
        preview_task = meshy_api.poll_text_to_3d(preview_id, interval_sec=5, timeout_sec=900)
        if preview_task.get("status") != "SUCCEEDED":
            return {"status": "error", "error": f"Preview failed: {preview_task.get('status')}"}
        final_task = preview_task

        # 3) 可选 refine（贴图）
        if refine:
            print(f"[Meshy] Starting refine for preview task: {preview_id}")
            refine_id = meshy_api.create_text_to_3d_refine(preview_id)
            refine_task = meshy_api.poll_text_to_3d(refine_id, interval_sec=5, timeout_sec=1800)
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
        meshy_api.download_model_url(file_url, local_path)
        
        return {
            'status': 'success',
            'message': f'Meshy Text-to-3D asset downloaded to {local_path}',
            'object_name': object_name,
            'local_path': local_path,
            'save_dir': save_dir,
            'model_url': file_url
        }

    except Exception as e:
        logging.error(f"Failed to download Meshy asset: {e}")
        return {"status": "error", "error": str(e)}


def download_meshy_asset_from_image(
    object_name: str,
    image_path: str,
    save_dir: str = "assets",
    prompt: str = None,
    meshy_api: MeshyAPI = None,
) -> dict:
    """
    使用 Meshy Image-to-3D 根据输入图片生成资产并下载到本地（生成→轮询→下载）

    Args:
        object_name: 对象名称
        image_path: 输入图片路径
        save_dir: 保存目录
        prompt: 可选的文本提示，用于指导生成
        meshy_api: MeshyAPI 实例，如果为 None 则创建新实例
    """
    try:
        # 检查图片文件是否存在
        if not os.path.exists(image_path):
            return {"status": "error", "error": f"Image file not found: {image_path}"}

        # 初始化 Meshy API
        if meshy_api is None:
            meshy_api = MeshyAPI()

        # 1) 创建 Image-to-3D preview 任务
        print(f"[Meshy] Creating Image-to-3D preview task for: {image_path}")
        if prompt:
            print(f"[Meshy] Using prompt: {prompt}")
        
        preview_id = meshy_api.create_image_to_3d_preview(image_path, prompt)

        # 2) 轮询 preview
        preview_task = meshy_api.poll_image_to_3d(preview_id, interval_sec=5, timeout_sec=900)
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
        meshy_api.download_model_url(file_url, local_path)

        return {
            'status': 'success',
            'message': f'Meshy Image-to-3D asset downloaded to {local_path}',
            'object_name': object_name,
            'local_path': local_path,
            'save_dir': save_dir,
            'model_url': file_url
        }
        
    except Exception as e:
        logging.error(f"Failed to download Meshy asset from image: {e}")
        return {"status": "error", "error": str(e)}


def create_rigged_character(
    model_url: str,
    save_dir: str = "assets",
    target_formats: list = None,
    topology: str = "triangle",
    target_polycount: int = 30000,
    meshy_api: MeshyAPI = None,
) -> dict:
    """
    创建带有绑定的角色模型
    
    Args:
        model_url: 3D模型的URL
        save_dir: 保存目录
        target_formats: 目标格式列表
        topology: 拓扑结构
        target_polycount: 目标多边形数量
        meshy_api: MeshyAPI 实例
        
    Returns:
        dict: 包含绑定结果的字典
    """
    try:
        # 初始化 Meshy API
        if meshy_api is None:
            meshy_api = MeshyAPI()

        # 1) 创建绑定任务
        print(f"[Meshy] Creating rigging task for: {model_url}")
        rig_task_id = meshy_api.create_rigging_task(
            model_url=model_url,
            target_formats=target_formats,
            topology=topology,
            target_polycount=target_polycount
        )

        # 2) 轮询绑定任务
        rig_task = meshy_api.poll_rigging_task(rig_task_id, interval_sec=5, timeout_sec=1800)
        if rig_task.get("status") != "SUCCEEDED":
            return {"status": "error", "error": f"Rigging failed: {rig_task.get('status')}"}

        # 3) 从结果中获取绑定的模型下载链接
        result = rig_task.get("result", {})
        rigged_model_url = result.get("rigged_character_fbx_url")
        if not rigged_model_url:
            return {"status": "error", "error": "No rigged model URL found in result"}

        # 4) 下载绑定的模型到本地
        os.makedirs(save_dir, exist_ok=True)
        local_path = os.path.join(save_dir, f"rigged_character_{rig_task_id}.fbx")
        print(f"[Meshy] Downloading rigged model to: {local_path}")
        meshy_api.download_model_url(rigged_model_url, local_path)

        return {
            'status': 'success',
            'message': f'Rigged character downloaded to {local_path}',
            'rig_task_id': rig_task_id,
            'local_path': local_path,
            'save_dir': save_dir,
            'rigged_model_url': rigged_model_url
        }

    except Exception as e:
        logging.error(f"Failed to create rigged character: {e}")
        return {"status": "error", "error": str(e)}


def create_animated_character(
    rig_task_id: str,
    action_id: int = 92,
    save_dir: str = "assets",
    post_process: dict = None,
    meshy_api: MeshyAPI = None,
) -> dict:
    """
    为绑定的角色创建动画
    
    Args:
        rig_task_id: 绑定任务的ID
        action_id: 动画动作ID
        save_dir: 保存目录
        post_process: 后处理参数
        meshy_api: MeshyAPI 实例
        
    Returns:
        dict: 包含动画结果的字典
    """
    try:
        # 初始化 Meshy API
        if meshy_api is None:
            meshy_api = MeshyAPI()

        # 1) 创建动画任务
        print(f"[Meshy] Creating animation task for rig_task_id: {rig_task_id}")
        anim_task_id = meshy_api.create_animation_task(
            rig_task_id=rig_task_id,
            action_id=action_id,
            post_process=post_process
        )

        # 2) 轮询动画任务
        anim_task = meshy_api.poll_animation_task(anim_task_id, interval_sec=5, timeout_sec=1800)
        if anim_task.get("status") != "SUCCEEDED":
            return {"status": "error", "error": f"Animation failed: {anim_task.get('status')}"}

        # 3) 从结果中获取动画文件下载链接
        result = anim_task.get("result", {})
        animated_model_url = result.get("animated_model_url")
        if not animated_model_url:
            return {"status": "error", "error": "No animated model URL found in result"}

        # 4) 下载动画模型到本地
        os.makedirs(save_dir, exist_ok=True)
        local_path = os.path.join(save_dir, f"animated_character_{anim_task_id}.glb")
        print(f"[Meshy] Downloading animated model to: {local_path}")
        meshy_api.download_model_url(animated_model_url, local_path)

        return {
            'status': 'success',
            'message': f'Animated character downloaded to {local_path}',
            'anim_task_id': anim_task_id,
            'rig_task_id': rig_task_id,
            'local_path': local_path,
            'save_dir': save_dir,
            'animated_model_url': animated_model_url
        }

    except Exception as e:
        logging.error(f"Failed to create animated character: {e}")
        return {"status": "error", "error": str(e)}


def create_rigged_and_animated_character(
    model_url: str,
    action_id: int = 92,
    save_dir: str = "assets",
    target_formats: list = None,
    topology: str = "triangle",
    target_polycount: int = 30000,
    post_process: dict = None,
    meshy_api: MeshyAPI = None,
) -> dict:
    """
    完整的流程：创建绑定角色并添加动画
    
    Args:
        model_url: 3D模型的URL
        action_id: 动画动作ID
        save_dir: 保存目录
        target_formats: 目标格式列表
        topology: 拓扑结构
        target_polycount: 目标多边形数量
        post_process: 后处理参数
        meshy_api: MeshyAPI 实例
        
    Returns:
        dict: 包含完整结果的字典
    """
    try:
        # 1) 首先创建绑定角色
        rigging_result = create_rigged_character(
            model_url=model_url,
            save_dir=save_dir,
            target_formats=target_formats,
            topology=topology,
            target_polycount=target_polycount,
            meshy_api=meshy_api
        )
        
        if rigging_result.get("status") != "success":
            return rigging_result

        # 2) 然后创建动画
        rig_task_id = rigging_result["rig_task_id"]
        animation_result = create_animated_character(
            rig_task_id=rig_task_id,
            action_id=action_id,
            save_dir=save_dir,
            post_process=post_process,
            meshy_api=meshy_api
        )

        if animation_result.get("status") != "success":
            return animation_result

        return {
            'status': 'success',
            'message': 'Rigged and animated character created successfully',
            'rig_task_id': rig_task_id,
            'anim_task_id': animation_result["anim_task_id"],
            'rigged_model_path': rigging_result["local_path"],
            'animated_model_path': animation_result["local_path"],
            'save_dir': save_dir
        }

    except Exception as e:
        logging.error(f"Failed to create rigged and animated character: {e}")
        return {"status": "error", "error": str(e)}
    
@mcp.tool()
def initialize(args: dict) -> dict:
    """
    Initialize Meshy server context.
    
    Args:
        args:
          - va_api_key: VA API key for ImageCropper (optional)
          - target_image_path: path to the target image for cropping (optional)
    """
    global _image_cropper
    try:
        va_api_key = args.get("va_api_key")
        target_image_path = args.get("target_image_path")
        meshy_api_key = args.get("meshy_api_key") or os.getenv("MESHY_API_KEY")
        if va_api_key and target_image_path:
            _image_cropper = ImageCropper(va_api_key, target_image_path)
        # Initialize global MeshyAPI if key available
        if meshy_api_key:
            global _meshy_api
            _meshy_api = MeshyAPI(meshy_api_key)
        return {"status": "success", "message": "Meshy initialize completed"}
    except Exception as e:
        return {"status": "error", "error": str(e)}

@mcp.tool()
def generate_and_download_3d_asset(object_name: str, reference_type: str, object_description: str = None, save_dir: str = "assets", rig_and_animate: bool = False, action_id: int = 92) -> dict:
    """
    Unified Meshy tool per system prompt: generate and download a 3D asset.
    Uses text description or an image (cropped if not provided) for generation.

    Args:
        object_name: Asset/object name, e.g., 'table', 'chair'.
        reference_type: 'text' or 'image'.
        object_description: Optional detailed description; if absent, falls back to object_name.
        image_path: Optional path to a reference image; if not provided and reference_type=='image', will crop from target image using ImageCropper.
        save_dir: Local directory to save the generated asset.
    """
    try:
        os.makedirs(save_dir, exist_ok=True)
        if reference_type == "text":
            description = (object_description or object_name or "").strip()
            if not description:
                return {"status": "error", "error": "object_description or object_name must be provided"}
            base_result = download_meshy_asset(object_name=object_name, description=description, save_dir=save_dir, meshy_api=_meshy_api)
            if base_result.get('status') != 'success' or not rig_and_animate:
                return base_result
            # Rig + animate using model_url if provided
            model_url = base_result.get('model_url')
            if not model_url:
                return base_result
            rigged = create_rigged_and_animated_character(model_url=model_url, action_id=action_id, save_dir=save_dir, meshy_api=_meshy_api)
            if rigged.get('status') == 'success':
                base_result['rig_task_id'] = rigged.get('rig_task_id')
                base_result['anim_task_id'] = rigged.get('anim_task_id')
                base_result['animated_model_path'] = rigged.get('animated_model_path')
            return base_result
        elif reference_type == "image":
            # crop from target image via ImageCropper
            if _image_cropper is None:
                return {"status": "error", "error": "ImageCropper not initialized. Call initialize with va_api_key and target_image_path."}
            crop_resp = _image_cropper.crop_image_by_text(object_name=object_name)
            # Expecting {'data': [[{'bounding_box': [x1,y1,x2,y2], ...}]]}
            try:
                bbox = crop_resp['data'][0][0]['bounding_box']
                from PIL import Image as _PILImage
                img = _PILImage.open(_image_cropper.target_image_path)
                x1, y1, x2, y2 = map(int, bbox)
                cropped = img.crop((x1, y1, x2, y2))
                local_image_path = os.path.join(save_dir, f"cropped_{object_name}.png")
                cropped.save(local_image_path)
            except Exception as e:
                return {"status": "error", "error": f"Cropping failed: {e}"}
            if not os.path.exists(local_image_path):
                return {"status": "error", "error": f"Image file not found: {local_image_path}"}
            base_result = download_meshy_asset_from_image(object_name=object_name, image_path=local_image_path, save_dir=save_dir, prompt=object_description, meshy_api=_meshy_api)
            if base_result.get('status') != 'success' or not rig_and_animate:
                return base_result
            model_url = base_result.get('model_url')
            if not model_url:
                return base_result
            rigged = create_rigged_and_animated_character(model_url=model_url, action_id=action_id, save_dir=save_dir, meshy_api=_meshy_api)
            if rigged.get('status') == 'success':
                base_result['rig_task_id'] = rigged.get('rig_task_id')
                base_result['anim_task_id'] = rigged.get('anim_task_id')
                base_result['animated_model_path'] = rigged.get('animated_model_path')
            return base_result
        else:
            return {"status": "error", "error": f"Unsupported reference_type: {reference_type}"}
    except Exception as e:
        return {"status": "error", "error": str(e)}

def main():
    mcp.run(transport="stdio")

if __name__ == "__main__":
    main()