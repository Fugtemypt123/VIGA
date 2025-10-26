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
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
        }
        self.save_dir = save_dir
        self.previous_assets_dir = previous_assets_dir
        os.makedirs(self.save_dir, exist_ok=True)
        if self.previous_assets_dir:
            os.makedirs(self.previous_assets_dir, exist_ok=True)
        # with open('logs/meshy.log', 'w') as f:
        #     f.write(f"MeshyAPI initialized with save_dir: {self.save_dir} and previous_assets_dir: {self.previous_assets_dir}\n")

    def normalize_name(self, name: str) -> str:
        """
        Normalize name for fuzzy matching: remove case, spaces, underscores, singular/plural differences
        """
        if not name:
            return ""
        
        # Convert to lowercase
        normalized = name.lower()
        
        # Remove spaces and underscores
        normalized = re.sub(r'[\s_]+', '', normalized)
        
        # Handle singular/plural differences
        # Remove common plural suffixes
        plural_endings = ['s', 'es', 'ies', 'ves']
        for ending in plural_endings:
            if normalized.endswith(ending) and len(normalized) > len(ending) + 1:
                # Special case: plural forms ending with y
                if ending == 'ies' and normalized.endswith('ies'):
                    normalized = normalized[:-3] + 'y'
                # Special case: plural forms ending with f/fe
                elif ending == 'ves' and (normalized.endswith('ves') and 
                                         (normalized[-4] == 'f' or normalized[-4] == 'e')):
                    normalized = normalized[:-3] + 'f'
                else:
                    normalized = normalized[:-len(ending)]
                break
        
        return normalized

    def find_matching_files(self, target_name: str, extensions: List[str], prefix: str = "") -> List[str]:
        """
        Find matching files in previous_assets directory
        
        Args:
            target_name: Target object name
            extensions: File extension list
            prefix: File name prefix (e.g., "animated_", "rigged_")
            
        Returns:
            List[str]: List of matching file paths
        """
        if not self.previous_assets_dir or not os.path.exists(self.previous_assets_dir):
            return []
        
        target_normalized = self.normalize_name(target_name)
        matching_files = []
        
        try:
            for filename in os.listdir(self.previous_assets_dir):
                # Check file extension
                if not any(filename.lower().endswith(ext.lower()) for ext in extensions):
                    continue
                
                # Remove extension and prefix
                base_name = filename
                for ext in extensions:
                    if base_name.lower().endswith(ext.lower()):
                        base_name = base_name[:-len(ext)]
                        break
                
                # Remove prefix
                if prefix:
                    if base_name.lower().startswith(prefix.lower()):
                        base_name = base_name[len(prefix):]
                    else:
                        continue
                
                # Normalize and compare
                base_normalized = self.normalize_name(base_name)
                if base_normalized in target_normalized or target_normalized in base_normalized:
                    matching_files.append(os.path.join(self.previous_assets_dir, filename))
                    
        except OSError as e:
            logging.warning(f"Error reading previous_assets directory: {e}")
            
        return matching_files

    def check_previous_asset(self, object_name: str, is_animated: bool = False, is_rigged: bool = False) -> Optional[str]:
        """
        Check if corresponding asset file exists in previous_assets directory (supports fuzzy matching)
        
        Args:
            object_name: Object name
            is_animated: Whether it is an animation file
            is_rigged: Whether it is a rigged file
            
        Returns:
            str: If file is found, return file path; otherwise return None
        """
        if not self.previous_assets_dir:
            return None
            
        # Determine search parameters based on settings
        if is_animated:
            extensions = [".glb"]
            prefix = "animated_"
        elif is_rigged:
            extensions = [".fbx"]
            prefix = "rigged_"
        else:
            # Static files, try common extensions
            extensions = [".glb", ".gltf", ".fbx", ".obj", ".zip"]
            prefix = ""
        
        # Use fuzzy matching to find files
        matching_files = self.find_matching_files(object_name, extensions, prefix)
        
        if matching_files:
            # Return the first matching file
            matched_file = matching_files[0]
            logging.info(f"Found previous asset (fuzzy match): {matched_file}")
            return matched_file
        
        return None

    def create_text_to_3d_preview(self, prompt: str, **kwargs) -> str:
        """
        Create Text-to-3D preview task (no texture)
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
        # Some environments return {"result": "<id>"}, some return {"id": "<id>"}
        return data.get("result") or data.get("id")

    def poll_text_to_3d(self, task_id: str, interval_sec: float = 5.0, timeout_sec: int = 1800) -> dict:
        """
        Poll Text-to-3D task until completion
        Returns: Task JSON (containing status / model_urls etc.)
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
        Launch refine texture task based on preview
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

    def download_model_url(self, file_url: str, filename: str) -> str:
        """
        Download file from model_urls direct link to local
        """
        r = requests.get(file_url, stream=True)
        r.raise_for_status()
        output_path = os.path.join(self.save_dir, filename)
        with open(output_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        # with open('logs/meshy.log', 'a') as f:
        #     f.write(f"Downloaded {filename} from {file_url} to {output_path}\n")
        return output_path

    def create_image_to_3d_preview(self, image_path: str, prompt: str = None, **kwargs) -> str:
        """
        Create Image-to-3D preview task (no texture)
        
        Args:
            image_path: Input image path
            prompt: Optional text prompt
            **kwargs: Other parameters
            
        Returns: task_id (str)
        """
        url = f"{self.base_url}/openapi/v1/image-to-3d"
        
        # Prepare file upload
        with open(image_path, 'rb') as f:
            # Convert image to base64 format
            image_base64 = base64.b64encode(f.read()).decode('utf-8')
            files = {
                'image_url': f"data:image/png;base64,{image_base64}",
                'enable_pbr': True,
            }
            
            # Send request (note: JSON headers not used here because we need to upload files)
            headers = {
                "Authorization": f"Bearer {self.api_key}"
            }
            
            resp = requests.post(url, headers=headers, json=files)
            resp.raise_for_status()
            data = resp.json()
            return data.get("result") or data.get("id")

    def poll_image_to_3d(self, task_id: str, interval_sec: float = 5.0, timeout_sec: int = 1800) -> dict:
        """
        Poll Image-to-3D task until completion
        Returns: Task JSON (containing status / model_urls etc.)
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

    def create_rigging_task(self, model_url: str) -> str:
        """
        Create automatic rigging task
        
        Args:
            model_url: URL of 3D model that needs rigging
            
        Returns: task_id (str)
        """
        url = f"{self.base_url}/openapi/v1/rigging"
        payload = {
            "model_url": model_url,
        }
        resp = requests.post(url, headers=self.headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data.get("result") or data.get("id")

    def poll_rigging_task(self, task_id: str, interval_sec: float = 5.0, timeout_sec: int = 1800) -> dict:
        """
        Poll rigging task until completion
        
        Args:
            task_id: Task ID
            interval_sec: Polling interval (seconds)
            timeout_sec: Timeout time (seconds)
            
        Returns: Task JSON (containing status / result etc.)
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

    def create_animation_task(self, rig_task_id: str, action_description: str) -> str:
        """
        Create animation task
        
        Args:
            rig_task_id: ID of successfully completed rigging task
            action_id: Identifier of animation action to apply, default 92
            
        Returns: task_id (str)
        """
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
        """
        Poll animation task until completion
        
        Args:
            task_id: Task ID
            interval_sec: Polling interval (seconds)
            timeout_sec: Timeout time (seconds)
            
        Returns: Task JSON (containing status / result etc.)
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
    """Image cropping tool, supports intelligent cropping based on text description (Grounding DINO + SAM / YOLO / OpenAI fallback)"""

    def __init__(self, api_key: str, target_image_path: str):
        self.url = "https://api.va.landing.ai/v1/tools/agentic-object-detection"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
        }
        self.target_image_path = target_image_path

    # ---------------------------
    # External: cropping (signature and return structure preserved)
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

        # Check if static file already exists in previous_assets directory
        previous_asset = _meshy_api.check_previous_asset(object_name, is_animated=False, is_rigged=False)
        if previous_asset:
            print(f"[Meshy] Using previous static asset: {previous_asset}")
            return {'status': 'success', 'output': {'path': previous_asset, 'model_url': None, 'from_cache': True}}

        # 1) Create preview task
        print(f"[Meshy] Creating preview task for: {description}")
        preview_id = _meshy_api.create_text_to_3d_preview(description)

        # 2) Poll preview
        preview_task = _meshy_api.poll_text_to_3d(preview_id, interval_sec=5, timeout_sec=900)
        if preview_task.get("status") != "SUCCEEDED":
            return {"status": "error", "output": f"Preview failed: {preview_task.get('status')}"}
        final_task = preview_task

        # 3) refine (texture)
        print(f"[Meshy] Starting refine for preview task: {preview_id}")
        refine_id = _meshy_api.create_text_to_3d_refine(preview_id)
        refine_task = _meshy_api.poll_text_to_3d(refine_id, interval_sec=5, timeout_sec=1800)
        if refine_task.get("status") != "SUCCEEDED":
            return {"status": "error", "output": f"Refine failed: {refine_task.get('status')}"}
        final_task = refine_task

        # 4) Get download link from model_urls
        model_urls = (final_task or {}).get("model_urls", {}) or {}
        candidate_keys = ["glb", "fbx", "obj", "zip"]
        file_url = None
        for k in candidate_keys:
            if model_urls.get(k):
                file_url = model_urls[k]
                break
        if not file_url:
            return {"status": "error", "output": "No downloadable model_urls found"}
        
        # 5) Download model to local
        result_path = _meshy_api.download_model_url(file_url, f"{object_name}_0.glb")
        print(f"[Meshy] Downloading Meshy asset to: {result_path}")
        
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
        # Check if image file exists
        if not os.path.exists(image_path):
            return {"status": "error", "output": f"Image file not found: {image_path}"}

        global _meshy_api
        if _meshy_api is None:
            return {"status": "error", "output": "Meshy API not initialized"}

        # Check if static file already exists in previous_assets directory
        previous_asset = _meshy_api.check_previous_asset(object_name, is_animated=False, is_rigged=False)
        if previous_asset:
            print(f"[Meshy] Using previous static asset from image: {previous_asset}")
            return {'status': 'success', 'output': {'path': previous_asset, 'model_url': None, 'from_cache': True}}

        # 1) Create Image-to-3D preview task
        print(f"[Meshy] Creating Image-to-3D preview task for: {image_path}")
        if prompt:
            print(f"[Meshy] Using prompt: {prompt}")
        
        preview_id = _meshy_api.create_image_to_3d_preview(image_path, prompt)

        # 2) Poll preview
        preview_task = _meshy_api.poll_image_to_3d(preview_id, interval_sec=5, timeout_sec=900)
        if preview_task.get("status") != "SUCCEEDED":
            return {"status": "error", "output": f"Image-to-3D preview failed: {preview_task.get('status')}"}
        final_task = preview_task

        # 3) Get download link from model_urls
        model_urls = (final_task or {}).get("model_urls", {}) or {}
        candidate_keys = ["glb", "fbx", "obj", "zip"]
        file_url = None
        for k in candidate_keys:
            if model_urls.get(k):
                file_url = model_urls[k]
                break
        if not file_url:
            return {"status": "error", "output": "No downloadable model_urls found"}

        # 4) Download model to local persistent directory
        # Handle extensionless direct links: default .glb
        result_path = _meshy_api.download_model_url(file_url, f"{object_name}_0.glb")
        print(f"[Meshy] Downloading Image-to-3D model to: {result_path}")

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

        # Check if rigged file already exists in previous_assets directory
        previous_asset = _meshy_api.check_previous_asset(object_name, is_animated=False, is_rigged=True)
        if previous_asset:
            print(f"[Meshy] Using previous rigged asset: {previous_asset}")
            return {
                'status': 'success',
                'output': {'task_id': 'cached', 'path': previous_asset, 'model_url': None, 'from_cache': True}
            }

        # 1) Create rigging task
        print(f"[Meshy] Creating rigging task for: {model_url}")
        rig_task_id = _meshy_api.create_rigging_task(model_url=model_url)

        # 2) Poll rigging task
        rig_task = _meshy_api.poll_rigging_task(rig_task_id, interval_sec=5, timeout_sec=1800)
        if rig_task.get("status") != "SUCCEEDED":
            return {"status": "error", "output": f"Rigging failed: {rig_task.get('status')}"}

        # 3) Get rigged model download link from result
        result = rig_task.get("result", {})
        rigged_model_url = result.get("rigged_character_fbx_url")
        if not rigged_model_url:
            return {"status": "error", "output": "No rigged model URL found in result"}

        # 4) Download rigged model to local
        local_path = _meshy_api.download_model_url(rigged_model_url, f"{object_name}_1.fbx")
        print(f"[Meshy] Downloading rigged model to: {local_path}")

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

        # Check if animation file already exists in previous_assets directory
        previous_asset = _meshy_api.check_previous_asset(object_name, is_animated=True, is_rigged=False)
        if previous_asset:
            print(f"[Meshy] Using previous animated asset: {previous_asset}")
            return {
                'status': 'success',
                'output': {'task_id': 'cached', 'path': previous_asset, 'model_url': None, 'from_cache': True}
            }

        # 1) Create animation task
        print(f"[Meshy] Creating animation task for rig_task_id: {rig_task_id}")
        anim_task_id = _meshy_api.create_animation_task(rig_task_id=rig_task_id, action_description=action_description)

        # 2) Poll animation task
        anim_task = _meshy_api.poll_animation_task(anim_task_id, interval_sec=5, timeout_sec=1800)
        if anim_task.get("status") != "SUCCEEDED":
            return {"status": "error", "output": f"Animation failed: {anim_task.get('status')}"}

        # 3) Get animation file download link from result
        result = anim_task.get("result", {})
        animated_model_url = result.get("animation_glb_url")
        if not animated_model_url:
            return {"status": "error", "output": "No animated model URL found in result"}

        # 4) Download animated model to local
        local_path = _meshy_api.download_model_url(animated_model_url, f"{object_name}_2.glb")
        print(f"[Meshy] Downloading animated model to: {local_path}")

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
        
        # 1) First create rigged character
        rigging_result = create_rigged_character(model_url=model_url, object_name=object_name)
        if rigging_result.get("status") != "success":
            return rigging_result

        # 2) Then create animation
        rig_task_id = rigging_result["output"]["task_id"]
        # If rigging result comes from cache, we need to check if there is already complete animation cache
        if rigging_result["output"].get("from_cache"):
            # Check if animation cache already exists
            animation_result = create_animated_character(rig_task_id="cached", action_description=action_description, object_name=object_name)
            if animation_result["output"].get("from_cache"):
                return animation_result
            # If no animation cache, need to regenerate rigging (because we need real rig_task_id)
            if not model_url:
                return {"status": "error", "output": "Cannot create animation without model_url when rigging is cached"}
            # Recreate rigging to get real task_id
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
        
        # If previous_assets_dir is not specified, default to assets folder in same directory as target_image_path
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
            print(f"[Meshy] Using previous static asset from image: {previous_asset}")
            return {'status': 'success', 'output': {'text': [f"Successfully generated static asset, downloaded to: {previous_asset}"]}}
        
        if reference_type == "text":
            description = (object_description or object_name or "").strip()
            if not description:
                return {"status": "error", "output": {"text": ["object_description or object_name must be provided"]}}
            static_result = download_meshy_asset(object_name=object_name, description=description)
            
        elif reference_type == "image":
            # crop from target image via ImageCropper
            if _image_cropper is None:
                return {"status": "error", "output": {"text": ["ImageCropper not initialized. Call initialize with va_api_key and target_image_path."]}}
            crop_resp = _image_cropper.crop_image_by_text(object_name=object_name)
            # Expecting {'data': [[{'bounding_box': [x1,y1,x2,y2], ...}]]}
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
    # Test entry similar to investigator.py
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        print("Running Meshy tools test (rig_and_animate)...")
        # Read basic args from env for simplicity
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
        print("[test] initialize(...) with:", init_payload)
        init_res = initialize(init_payload)
        print("[test:init]", json.dumps(init_res, ensure_ascii=False))
        if init_res.get("status") != "success":
            print("Initialization failed; aborting test.")
            return

        # Text reference test
        print("\n[test] Text reference: humanoid, rig_and_animate=True")
        text_res = meshy_get_better_object(
            object_name="Snow man",
            reference_type="text",
            object_description="stylized humanoid character",
            rig_and_animate=False,
            action_description="walk",
        )
        print(json.dumps(text_res, ensure_ascii=False, indent=2))
    else:
        # Normal MCP server mode
        mcp.run()

if __name__ == "__main__":
    main()