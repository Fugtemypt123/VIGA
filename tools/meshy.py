# meshy.py
import os
import sys
import base64
import json
import requests
import time
import logging
import PIL
import re
from typing import Optional, List
from mcp.server.fastmcp import FastMCP

# tool_configs for agent (only the function w/ @mcp.tools)
tool_configs = [
    {
        "type": "function",
        "function": {
            "name": "meshy_get_better_object",
            "description": "Use the Meshy API to generate a 3D object and download it to local path. You may provide either text or image as the reference: If the target 3D asset in the reference image is clear and unobstructed, use reference_type=\"image\". Otherwise, use reference_type=\"text\". The tool downloads the generated asset locally and returns its file path for later import in code. We have unlimited meshy resources, please use this tool to generate complex objects whenever possible.",
            "parameters": {
                "type": "object",
                "properties": {
                    "object_name": {"type": "string", "description": "The name of the object to generate. For example, 'chair', 'table', 'lamp', etc."},
                    "reference_type": {"type": "string", "enum": ["text", "image"], "description": 'The type of reference to use. If the target 3D asset in the reference image is clear and unobstructed, use reference_type=\"image\". Otherwise, use reference_type=\"text\".'},
                    "object_description": {"type": "string", "description": "If you use reference_type=\"text\", you must provide a detailed description of the object to generate."},
                    "rig_and_animate": {"type": "boolean", "description": "Whether to rig and animate the generated asset. True for dynamic scene, False for static scene"},
                    "action_description": {"type": "string", "description": "If you use rig_and_animate=True, you must provide a description of the action to apply to the generated asset. Only input verbs here, e.g. walk, run, jump, etc."}
                },
                "required": ["object_name", "reference_type", "rig_and_animate"]
            }
        }
    }
]


mcp = FastMCP("meshy-executor")
_image_cropper = None
_meshy_api = None

class MeshyAPI:
    """Meshy API client: Text-to-3D generation + polling + download"""
    def __init__(self, api_key: str = None, save_dir: str = None, previous_assets_dir: str = None):
        self.api_key = api_key or os.getenv("MESHY_API_KEY")
        if not self.api_key:
            raise ValueError("Meshy API key is required. Set MESHY_API_KEY environment variable or pass api_key parameter.")
        self.base_url = "https://api.meshy.ai"
        self.headers = {"Authorization": f"Bearer {self.api_key}"}
        self.save_dir = save_dir
        self.previous_assets_dir = previous_assets_dir
        os.makedirs(self.save_dir, exist_ok=True)
        if self.previous_assets_dir:
            os.makedirs(self.previous_assets_dir, exist_ok=True)
        with open('logs/meshy.log', 'w') as f:
            f.write(f"MeshyAPI initialized with save_dir: {self.save_dir} and previous_assets_dir: {self.previous_assets_dir}\n")

    def normalize_name(self, name: str) -> str:
        if not name:
            return ""
        normalized = name.lower()
        normalized = re.sub(r'[\s_]+', '', normalized)
        plural_endings = ['s', 'es', 'ies', 'ves']
        for ending in plural_endings:
            if normalized.endswith(ending) and len(normalized) > len(ending) + 1:
                if ending == 'ies' and normalized.endswith('ies'):
                    normalized = normalized[:-3] + 'y'
                elif ending == 'ves' and (normalized.endswith('ves') and 
                                         (normalized[-4] == 'f' or normalized[-4] == 'e')):
                    normalized = normalized[:-3] + 'f'
                else:
                    normalized = normalized[:-len(ending)]
                break
        return normalized

    def find_matching_files(self, target_name: str, extensions: List[str], prefix: str = "") -> List[str]:
        if not self.previous_assets_dir or not os.path.exists(self.previous_assets_dir):
            return []
        target_normalized = self.normalize_name(target_name)
        matching_files = []
        try:
            for filename in os.listdir(self.previous_assets_dir):
                if not any(filename.lower().endswith(ext.lower()) for ext in extensions):
                    continue
                base_name = filename
                for ext in extensions:
                    if base_name.lower().endswith(ext.lower()):
                        base_name = base_name[:-len(ext)]
                        break
                if prefix:
                    if base_name.lower().startswith(prefix.lower()):
                        base_name = base_name[len(prefix):]
                    else:
                        continue
                base_normalized = self.normalize_name(base_name)
                if base_normalized in target_normalized or target_normalized in base_normalized:
                    matching_files.append(os.path.join(self.previous_assets_dir, filename))
        except OSError as e:
            logging.warning(f"Error reading previous_assets directory: {e}")
        return matching_files

    def check_previous_asset(self, object_name: str, is_animated: bool = False, is_rigged: bool = False) -> Optional[str]:
        if not self.previous_assets_dir:
            return None
        if is_animated:
            extensions = [".glb"]
            prefix = "animated_"
        elif is_rigged:
            extensions = [".fbx"]
            prefix = "rigged_"
        else:
            extensions = [".glb", ".gltf", ".fbx", ".obj", ".zip"]
            prefix = ""
        matching_files = self.find_matching_files(object_name, extensions, prefix)
        if matching_files:
            matched_file = matching_files[0]
            logging.info(f"Found previous asset (fuzzy match): {matched_file}")
            return matched_file
        return None

    def create_text_to_3d_preview(self, prompt: str, **kwargs) -> str:
        url = f"{self.base_url}/openapi/v2/text-to-3d"
        payload = {"mode": "preview", "prompt": prompt[:600]}
        payload.update(kwargs or {})
        resp = requests.post(url, headers=self.headers, data=json.dumps(payload))
        resp.raise_for_status()
        data = resp.json()
        return data.get("result") or data.get("id")

    def poll_text_to_3d(self, task_id: str, interval_sec: float = 5.0, timeout_sec: int = 1800) -> dict:
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
        url = f"{self.base_url}/openapi/v2/text-to-3d"
        payload = {"mode": "refine", "preview_task_id": preview_task_id}
        payload.update(kwargs or {})
        resp = requests.post(url, headers=self.headers, data=json.dumps(payload))
        resp.raise_for_status()
        data = resp.json()
        return data.get("result") or data.get("id")

    def download_model_url(self, file_url: str, file_name: str) -> str:
        r = requests.get(file_url, stream=True)
        r.raise_for_status()
        output_path = os.path.join(self.save_dir, file_name)
        with open(output_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        with open('logs/meshy.log', 'a') as f:
            f.write(f"Downloaded {file_name} from {file_url} to {output_path}\n")
        return output_path

    def create_image_to_3d_preview(self, image_path: str, prompt: str = None, **kwargs) -> str:
        url = f"{self.base_url}/openapi/v1/image-to-3d"
        with open(image_path, 'rb') as f:
            image_base64 = base64.b64encode(f.read()).decode('utf-8')
            files = {'image_url': f"data:image/png;base64,{image_base64}", 'enable_pbr': True}
            resp = requests.post(url, headers=self.headers, json=files)
            resp.raise_for_status()
            data = resp.json()
            return data.get("result") or data.get("id")

    def poll_image_to_3d(self, task_id: str, interval_sec: float = 5.0, timeout_sec: int = 1800) -> dict:
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

    def create_rigging_task(self, model_url: str) -> str:
        url = f"{self.base_url}/openapi/v1/rigging"
        payload = {"model_url": model_url}
        resp = requests.post(url, headers=self.headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data.get("result") or data.get("id")

    def poll_rigging_task(self, task_id: str, interval_sec: float = 5.0, timeout_sec: int = 1800) -> dict:
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

    def create_animation_task(self, rig_task_id: str, action_description: str) -> str:
        from knowledge_base.meshy_builder import search
        action = search(action_description)
        if not action:
            return {"status": "error", "output": "No action found"}
        url = f"{self.base_url}/openapi/v1/animations"
        payload = {"rig_task_id": rig_task_id, "action_id": action['action_id']}
        resp = requests.post(url, headers=self.headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data.get("result") or data.get("id")

    def poll_animation_task(self, task_id: str, interval_sec: float = 5.0, timeout_sec: int = 1800) -> dict:
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
    def __init__(self, api_key: str, target_image_path: str):
        self.url = "https://api.va.landing.ai/v1/tools/agentic-object-detection"
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.target_image_path = target_image_path

    def crop_image_by_text(self, object_name: str) -> dict:
        files = {"image": open(self.target_image_path, "rb")}
        data = {"prompts": object_name, "model": "agentic"}
        response = requests.post(self.url, files=files, data=data, headers=self.headers)
        return response.json()


def download_meshy_asset(object_name: str, description: str) -> dict:
    """
    Download Meshy Text-to-3D asset
    
    Args:
        object_name: Object name
        description: Object description
    """
    try:
        global _meshy_api
        if _meshy_api is None:
            return {"status": "error", "output": "Meshy API not initialized"}

        previous_asset = _meshy_api.check_previous_asset(object_name, is_animated=False, is_rigged=False)
        if previous_asset:
            logging.info(f"[Meshy] Using previous static asset: {previous_asset}")
            return {'status': 'success', 'output': {'path': previous_asset, 'model_url': None, 'from_cache': True}}

        logging.info(f"[Meshy] Creating preview task for: {description}")
        preview_id = _meshy_api.create_text_to_3d_preview(description)
        with open('logs/meshy.log', 'a') as f:
            f.write(f"Preview ID: {preview_id}\n")

        preview_task = _meshy_api.poll_text_to_3d(preview_id, interval_sec=5, timeout_sec=900)
        if preview_task.get("status") != "SUCCEEDED":
            return {"status": "error", "output": f"Preview failed: {preview_task.get('status')}"}
        final_task = preview_task

        logging.info(f"[Meshy] Starting refine for preview task: {preview_id}")
        refine_id = _meshy_api.create_text_to_3d_refine(preview_id)
        with open('logs/meshy.log', 'a') as f:
            f.write(f"Refine ID: {refine_id}\n")
            
        refine_task = _meshy_api.poll_text_to_3d(refine_id, interval_sec=5, timeout_sec=1800)
        if refine_task.get("status") != "SUCCEEDED":
            return {"status": "error", "output": f"Refine failed: {refine_task.get('status')}"}
        final_task = refine_task

        model_urls = (final_task or {}).get("model_urls", {}) or {}
        candidate_keys = ["glb", "fbx", "obj", "zip"]
        file_url = None
        for k in candidate_keys:
            if model_urls.get(k):
                file_url = model_urls[k]
                break
        if not file_url:
            return {"status": "error", "output": "No downloadable model_urls found"}
 
        result_path = _meshy_api.download_model_url(file_url, f"{object_name}.glb")
        logging.info(f"[Meshy] Downloading Meshy asset to: {result_path}")
        
        return {'status': 'success', 'output': {'path': result_path, 'model_url': file_url}}

    except Exception as e:
        logging.error(f"Failed to download Meshy asset: {e}")
        return {"status": "error", "output": str(e)}


def download_meshy_asset_from_image(object_name: str, image_path: str, prompt: str = None) -> dict:
    """
    Use Meshy Image-to-3D to generate asset from input image and download to local (generate→poll→download)

    Args:
        object_name: Object name
        image_path: Input image path
        prompt: Optional text prompt for guiding generation
    """
    try:
        if not os.path.exists(image_path):
            return {"status": "error", "output": f"Image file not found: {image_path}"}

        global _meshy_api
        if _meshy_api is None:
            return {"status": "error", "output": "Meshy API not initialized"}

        previous_asset = _meshy_api.check_previous_asset(object_name, is_animated=False, is_rigged=False)
        if previous_asset:
            logging.info(f"[Meshy] Using previous static asset from image: {previous_asset}")
            return {'status': 'success', 'output': {'path': previous_asset, 'model_url': None, 'from_cache': True}}

        logging.info(f"[Meshy] Creating Image-to-3D preview task for: {image_path}")
        if prompt:
            logging.info(f"[Meshy] Using prompt: {prompt}")
        
        preview_id = _meshy_api.create_image_to_3d_preview(image_path, prompt)
        with open('logs/meshy.log', 'a') as f:
            f.write(f"Preview ID: {preview_id}\n")

        preview_task = _meshy_api.poll_image_to_3d(preview_id, interval_sec=5, timeout_sec=900)
        if preview_task.get("status") != "SUCCEEDED":
            return {"status": "error", "output": f"Image-to-3D preview failed: {preview_task.get('status')}"}
        final_task = preview_task

        model_urls = (final_task or {}).get("model_urls", {}) or {}
        candidate_keys = ["glb", "fbx", "obj", "zip"]
        file_url = None
        for k in candidate_keys:
            if model_urls.get(k):
                file_url = model_urls[k]
                break
        if not file_url:
            return {"status": "error", "output": "No downloadable model_urls found"}

        result_path = _meshy_api.download_model_url(file_url, f"{object_name}.glb")
        logging.info(f"[Meshy] Downloading Image-to-3D model to: {result_path}")
        return {'status': 'success', 'output': {'path': result_path, 'model_url': file_url}}
        
    except Exception as e:
        logging.error(f"Failed to download Meshy asset from image: {e}")
        return {"status": "error", "output": str(e)}


def create_rigged_character(model_url: str, object_name: str) -> dict:
    """
    Create character model with rigging
    
    Args:
        model_url: URL of 3D model
        
    Returns:
        dict: Dictionary containing rigging results
    """
    try:
        global _meshy_api
        if _meshy_api is None:
            return {"status": "error", "output": "Meshy API not initialized"}

        previous_asset = _meshy_api.check_previous_asset(object_name, is_animated=False, is_rigged=True)
        if previous_asset:
            logging.info(f"[Meshy] Using previous rigged asset: {previous_asset}")
            return {
                'status': 'success',
                'output': {'task_id': 'cached', 'path': previous_asset, 'model_url': None, 'from_cache': True}
            }

        logging.info(f"[Meshy] Creating rigging task for: {model_url}")
        rig_task_id = _meshy_api.create_rigging_task(model_url=model_url)
        with open('logs/meshy.log', 'a') as f:
            f.write(f"Rig task ID: {rig_task_id}\n")

        rig_task = _meshy_api.poll_rigging_task(rig_task_id, interval_sec=5, timeout_sec=1800)
        if rig_task.get("status") != "SUCCEEDED":
            return {"status": "error", "output": f"Rigging failed: {rig_task.get('status')}"}

        result = rig_task.get("result", {})
        rigged_model_url = result.get("rigged_character_fbx_url")
        if not rigged_model_url:
            return {"status": "error", "output": "No rigged model URL found in result"}

        local_path = _meshy_api.download_model_url(rigged_model_url, f"rigged_{object_name}.fbx")
        logging.info(f"[Meshy] Downloading rigged model to: {local_path}")

        return {
            'status': 'success',
            'output': {'task_id': rig_task_id, 'path': local_path, 'model_url': rigged_model_url}
        }

    except Exception as e:
        logging.error(f"Failed to create rigged character: {e}")
        return {"status": "error", "output": str(e)}


def create_animated_character(rig_task_id: str, action_description: str, object_name: str) -> dict:
    """
    Create animation for rigged character
    
    Args:
        rig_task_id: ID of rigging task
        action_description: Animation action description
        
    Returns:
        dict: Dictionary containing animation results
    """
    try:
        global _meshy_api
        if _meshy_api is None:
            return {"status": "error", "output": "Meshy API not initialized"}

        previous_asset = _meshy_api.check_previous_asset(object_name, is_animated=True, is_rigged=False)
        if previous_asset:
            logging.info(f"[Meshy] Using previous animated asset: {previous_asset}")
            return {
                'status': 'success',
                'output': {'task_id': 'cached', 'path': previous_asset, 'model_url': None, 'from_cache': True}
            }

        logging.info(f"[Meshy] Creating animation task for rig_task_id: {rig_task_id}")
        anim_task_id = _meshy_api.create_animation_task(rig_task_id=rig_task_id, action_description=action_description)
        with open('logs/meshy.log', 'a') as f:
            f.write(f"Anim task ID: {anim_task_id}\n")

        anim_task = _meshy_api.poll_animation_task(anim_task_id, interval_sec=5, timeout_sec=1800)
        if anim_task.get("status") != "SUCCEEDED":
            return {"status": "error", "output": f"Animation failed: {anim_task.get('status')}"}

        result = anim_task.get("result", {})
        animated_model_url = result.get("animation_glb_url")
        if not animated_model_url:
            return {"status": "error", "output": "No animated model URL found in result"}

        local_path = _meshy_api.download_model_url(animated_model_url, f"animated_{object_name}.glb")
        logging.info(f"[Meshy] Downloading animated model to: {local_path}")

        return {
            'status': 'success',
            'output': {'task_id': anim_task_id, 'path': local_path, 'model_url': animated_model_url}
        }

    except Exception as e:
        logging.error(f"Failed to create animated character: {e}")
        return {"status": "error", "output": str(e)}


def create_rigged_and_animated_character(model_url: str, action_description: str, object_name: str) -> dict:
    """
    Complete process: create rigged character and add animation
    
    Args:
        model_url: URL of 3D model
        action_description: Animation action description
        
    Returns:
        dict: Dictionary containing complete results
    """
    try:
        global _meshy_api
        if _meshy_api is None:
            return {"status": "error", "output": "Meshy API not initialized"}
        
        rigging_result = create_rigged_character(model_url=model_url, object_name=object_name)
        if rigging_result.get("status") != "success":
            return rigging_result

        rig_task_id = rigging_result["output"]["task_id"]
        if rigging_result["output"].get("from_cache"):
            animation_result = create_animated_character(rig_task_id="cached", action_description=action_description, object_name=object_name)
            if animation_result["output"].get("from_cache"):
                return animation_result
            if not model_url:
                return {"status": "error", "output": "Cannot create animation without model_url when rigging is cached"}
            rigging_result = create_rigged_character(model_url=model_url, object_name=object_name)
            if rigging_result.get("status") != "success":
                return rigging_result
            rig_task_id = rigging_result["output"]["task_id"]
        
        animation_result = create_animated_character(rig_task_id=rig_task_id, action_description=action_description, object_name=object_name)

        if animation_result.get("status") != "success":
            return animation_result

        return {'status': 'success', 'output': animation_result["output"]}

    except Exception as e:
        logging.error(f"Failed to create rigged and animated character: {e}")
        return {"status": "error", "output": str(e)}
    
@mcp.tool()
def initialize(args: dict) -> dict:
    global _image_cropper
    try:
        va_api_key = args.get("va_api_key")
        target_image_path = args.get("target_image_path")
        meshy_api_key = args.get("meshy_api_key") or os.getenv("MESHY_API_KEY")
        save_dir = args.get("output_dir") + "/assets"
        previous_assets_dir = args.get("assets_dir")
        
        if not previous_assets_dir and target_image_path:
            target_dir = os.path.dirname(target_image_path)
            previous_assets_dir = os.path.join(target_dir, "assets")
        
        if va_api_key and target_image_path:
            _image_cropper = ImageCropper(va_api_key, target_image_path)
        if meshy_api_key:
            global _meshy_api
            _meshy_api = MeshyAPI(meshy_api_key, save_dir, previous_assets_dir)
        return {"status": "success", "output": {"text": ["Meshy initialize completed"], "tool_configs": tool_configs}}
    
    except Exception as e:
        return {"status": "error", "output": {"text": [str(e)]}}

@mcp.tool()
def meshy_get_better_object(object_name: str, reference_type: str, object_description: str = None, rig_and_animate: bool = False, action_description: str = None) -> dict:
    try:
        global _meshy_api
        previous_asset = _meshy_api.check_previous_asset(object_name, is_animated=rig_and_animate, is_rigged=rig_and_animate)
        if previous_asset:
            logging.info(f"[Meshy] Using previous static asset from image: {previous_asset}")
            return {'status': 'success', 'output': {'text': [f"Successfully generated static asset, downloaded to: {previous_asset}"]}}
        
        if reference_type == "text":
            description = (object_description or object_name or "").strip()
            if not description:
                return {"status": "error", "output": {"text": ["object_description or object_name must be provided"]}}
            static_result = download_meshy_asset(object_name=object_name, description=description)
            
        elif reference_type == "image":
            if _image_cropper is None:
                return {"status": "error", "output": {"text": ["ImageCropper not initialized. Call initialize with va_api_key and target_image_path."]}}
            crop_resp = _image_cropper.crop_image_by_text(object_name=object_name)
            try:
                bbox = crop_resp['data'][0][0]['bounding_box']
                from PIL import Image as _PILImage
                img = _PILImage.open(_image_cropper.target_image_path)
                x1, y1, x2, y2 = map(int, bbox)
                cropped = img.crop((x1, y1, x2, y2))
                local_image_path = os.path.join(_meshy_api.save_dir, f"cropped_{object_name}.png")
                cropped.save(local_image_path)
            except Exception as e:
                return {"status": "error", "output": {"text": [f"Cropping failed: {e}"]}}
            if not os.path.exists(local_image_path):
                return {"status": "error", "output": {"text": [f"Image file not found: {local_image_path}"]}}
            static_result = download_meshy_asset_from_image(object_name=object_name, image_path=local_image_path)
        
        if static_result.get('status') != 'success':
            return {"status": "error", "output": {"text": [static_result.get('output', {}).get('text', ['Failed to generate static asset'])]}}
        if not rig_and_animate:
            return {"status": "success", "output": {"text": ["Successfully generated static asset, downloaded to: " + static_result.get('output', {}).get('path', '')]}}
        
        model_url = static_result.get('model_url', None)
        dynamic_result = create_rigged_and_animated_character(model_url=model_url, action_description=action_description, object_name=object_name)
        if dynamic_result.get('status') == 'success':
            return {"status": "success", "output": {"text": ["Successfully generated dynamic asset, downloaded to: " + dynamic_result.get('output', {}).get('path', '')]}}
        else:
            return {"status": "error", "output": {"text": [dynamic_result.get('output', {}).get('text', ['Failed to generate dynamic asset'])]}}
    
    except Exception as e:
        return {"status": "error", "output": {"text": [str(e)]}}

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        logging.info("Running Meshy tools test (rig_and_animate)...")
        meshy_api_key = os.getenv("MESHY_API_KEY")
        va_api_key = os.getenv("VA_API_KEY")
        save_dir = "test/meshy/assets"
        previous_assets_dir = "data/static_scene/christmas1/assets"

        init_payload = {
            "meshy_api_key": meshy_api_key,
            "va_api_key": va_api_key,
            "save_dir": save_dir,
            "previous_assets_dir": previous_assets_dir,
        }
        logging.info("[test] initialize(...) with:", init_payload)
        init_res = initialize(init_payload)
        logging.info("[test:init]", json.dumps(init_res, ensure_ascii=False))
        if init_res.get("status") != "success":
            logging.info("Initialization failed; aborting test.")
            return

        logging.info("\n[test] Text reference: humanoid, rig_and_animate=True")
        text_res = meshy_get_better_object(
            object_name="cat",
            reference_type="text",
            object_description="Realistic domestic short-hair bicolor cat with gray and white coat similar to a tuxedo pattern: gray cap over head and ears, gray back and tail, white face blaze, chest, belly, and legs. Medium adult size with short fur. Blender-compatible quadruped rig suitable for walk cycle. Include natural materials with slight fur roughness and subtle color variation.",
            rig_and_animate=True,
            action_description="walk",
        )
        logging.info(json.dumps(text_res, ensure_ascii=False, indent=2))
    else:
        mcp.run()
        
    # "{\"object_name\":\"cat\",\"reference_type\":\"text\",\"object_description\":\"Realistic domestic short-hair bicolor cat with gray and white coat similar to a tuxedo pattern: gray cap over head and ears, gray back and tail, white face blaze, chest, belly, and legs. Medium adult size with short fur. Blender-compatible quadruped rig suitable for walk cycle. Include natural materials with slight fur roughness and subtle color variation.\",\"rig_and_animate\":true,\"action_description\":\"walk\"}"

if __name__ == "__main__":
    main()