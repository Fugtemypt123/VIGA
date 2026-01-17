"""Meshy MCP Server for 3D Asset Generation.

Provides tools for generating high-quality 3D assets using Meshy API,
including text-to-3D, image-to-3D, rigging, and animation capabilities.
"""

import base64
import json
import logging
import os
import re
import sys
import time
from typing import Dict, List, Optional

import requests
from mcp.server.fastmcp import FastMCP

# Tool configuration for agent
tool_configs: List[Dict[str, object]] = [
    {
        "type": "function",
        "function": {
            "name": "get_better_object",
            "description": "Generate high-quality 3D assets, download them locally, and provide their paths for later use. The textures, materials, and finishes of these objects are already high-quality with fine-grained detail; please do not repaint them. If you do, you will need to re-import the object.",
            "parameters": {
                "type": "object",
                "properties": {
                    "thought": {"type": "string", "description": "Think about the object you want to download. Consider the overall description of the scene and the detailed plan for scene construction."},
                    "object_name": {"type": "string", "description": "The name of the object to download. For example, 'chair', 'table', 'lamp', etc."},
                    "reference_type": {"type": "string", "enum": ["text", "image"], "description": 'The type of generation reference. If the target 3D asset in the reference image is clear and unobstructed, use reference_type=\"image\". Otherwise, use reference_type=\"text\".'},
                    "object_description": {"type": "string", "description": "If you use reference_type=\"text\", you must provide a detailed description of the object to download."},
                    "rig_and_animate": {"type": "boolean", "description": "Whether to rig and animate the downloaded asset. True for dynamic scene, False for static scene"},
                    "action_description": {"type": "string", "description": "If you use rig_and_animate=True, you must provide a description of the action to apply to the downloaded asset. Only input verbs here, e.g. walk, run, jump, etc."}
                },
                "required": ["thought", "object_name"]
            }
        }
    }
]


# Create MCP instance
mcp = FastMCP("meshy-executor")

# Global instances
_image_cropper: Optional["ImageCropper"] = None
_meshy_api: Optional["MeshyAPI"] = None


class MeshyAPI:
    """Meshy API client for 3D asset generation.

    Handles text-to-3D generation, image-to-3D generation, rigging,
    animation, polling, and downloading of 3D assets.

    Attributes:
        api_key: Meshy API key.
        base_url: Meshy API base URL.
        headers: Request headers with authorization.
        save_dir: Directory for saving downloaded assets.
        previous_assets_dir: Directory to check for cached assets.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        save_dir: Optional[str] = None,
        previous_assets_dir: Optional[str] = None
    ) -> None:
        """Initialize the Meshy API client.

        Args:
            api_key: Meshy API key. Falls back to MESHY_API_KEY env var.
            save_dir: Directory for saving downloaded assets.
            previous_assets_dir: Directory to check for cached assets.

        Raises:
            ValueError: If no API key is provided.
        """
        self.api_key = api_key or os.getenv("MESHY_API_KEY")
        if not self.api_key:
            raise ValueError("Meshy API key is required. Set MESHY_API_KEY environment variable or pass api_key parameter.")
        self.base_url = "https://api.meshy.ai"
        self.headers = {"Authorization": f"Bearer {self.api_key}"}
        self.save_dir = previous_assets_dir
        self.previous_assets_dir = previous_assets_dir
        os.makedirs(self.save_dir, exist_ok=True)
        if self.previous_assets_dir:
            os.makedirs(self.previous_assets_dir, exist_ok=True)
        with open(f'{self.save_dir}/meshy.log', 'w') as f:
            f.write(f"MeshyAPI initialized with save_dir: {self.save_dir} and previous_assets_dir: {self.previous_assets_dir}\n")

    def normalize_name(self, name: str) -> str:
        """Normalize object name for fuzzy matching.

        Converts to lowercase, removes spaces/underscores, and handles
        common plural endings.

        Args:
            name: Object name to normalize.

        Returns:
            Normalized name string.
        """
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
        """Find files matching target name with fuzzy matching.

        Args:
            target_name: Name to search for.
            extensions: List of file extensions to match (e.g., ['.glb']).
            prefix: Optional filename prefix filter.

        Returns:
            List of matching file paths.
        """
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
        """Check for previously cached asset matching the object name.

        Args:
            object_name: Name of the object to find.
            is_animated: Whether to search for animated assets.
            is_rigged: Whether to search for rigged assets.

        Returns:
            Path to cached asset if found, None otherwise.
        """
        if not self.previous_assets_dir:
            return None
        extensions = [".glb"]
        prefix = ""
        if is_animated:
            prefix = "animated_"
        elif is_rigged:
            prefix = "rigged_"
        matching_files = self.find_matching_files(object_name, extensions, prefix)
        if matching_files:
            matched_file = matching_files[0]
            logging.info(f"Found previous asset (fuzzy match): {matched_file}")
            return matched_file
        return None

    def create_text_to_3d_preview(self, prompt: str, **kwargs: object) -> str:
        """Create a text-to-3D preview task.

        Args:
            prompt: Text description of the 3D model to generate.
            **kwargs: Additional parameters for the API request.

        Returns:
            Task ID for the preview generation.
        """
        url = f"{self.base_url}/openapi/v2/text-to-3d"
        payload = {"mode": "preview", "prompt": prompt[:600]}
        payload.update(kwargs or {})
        resp = requests.post(url, headers=self.headers, data=json.dumps(payload))
        resp.raise_for_status()
        data = resp.json()
        return data.get("result") or data.get("id")

    def poll_text_to_3d(self, task_id: str, interval_sec: float = 5.0, timeout_sec: int = 1800) -> Dict[str, object]:
        """Poll text-to-3D task until completion.

        Args:
            task_id: ID of the task to poll.
            interval_sec: Seconds between poll attempts.
            timeout_sec: Maximum seconds to wait before timeout.

        Returns:
            Task result dictionary with status and model URLs.

        Raises:
            TimeoutError: If task doesn't complete within timeout.
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

    def create_text_to_3d_refine(self, preview_task_id: str, **kwargs: object) -> str:
        """Create a text-to-3D refine task from a preview.

        Args:
            preview_task_id: ID of the completed preview task.
            **kwargs: Additional parameters for the API request.

        Returns:
            Task ID for the refine generation.
        """
        url = f"{self.base_url}/openapi/v2/text-to-3d"
        payload = {"mode": "refine", "preview_task_id": preview_task_id}
        payload.update(kwargs or {})
        resp = requests.post(url, headers=self.headers, data=json.dumps(payload))
        resp.raise_for_status()
        data = resp.json()
        return data.get("result") or data.get("id")

    def download_model_url(self, file_url: str, file_name: str) -> str:
        """Download a 3D model from URL to local storage.

        Args:
            file_url: URL of the model file to download.
            file_name: Local filename for the downloaded model.

        Returns:
            Path to the downloaded file.
        """
        r = requests.get(file_url, stream=True)
        r.raise_for_status()
        output_path = os.path.join(self.save_dir, file_name)
        with open(output_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        with open(f'{self.save_dir}/meshy.log', 'a') as f:
            f.write(f"Downloaded {file_name} from {file_url} to {output_path}\n")
        return output_path

    def create_image_to_3d_preview(self, image_path: str, prompt: Optional[str] = None, **kwargs: object) -> str:
        """Create an image-to-3D task from an input image.

        Args:
            image_path: Path to the input image file.
            prompt: Optional text prompt to guide generation.
            **kwargs: Additional parameters for the API request.

        Returns:
            Task ID for the image-to-3D generation.
        """
        url = f"{self.base_url}/openapi/v1/image-to-3d"
        with open(image_path, 'rb') as f:
            image_base64 = base64.b64encode(f.read()).decode('utf-8')
            files = {'image_url': f"data:image/png;base64,{image_base64}", 'enable_pbr': True}
            resp = requests.post(url, headers=self.headers, json=files)
            resp.raise_for_status()
            data = resp.json()
            return data.get("result") or data.get("id")

    def poll_image_to_3d(self, task_id: str, interval_sec: float = 5.0, timeout_sec: int = 1800) -> Dict[str, object]:
        """Poll image-to-3D task until completion.

        Args:
            task_id: ID of the task to poll.
            interval_sec: Seconds between poll attempts.
            timeout_sec: Maximum seconds to wait before timeout.

        Returns:
            Task result dictionary with status and model URLs.

        Raises:
            TimeoutError: If task doesn't complete within timeout.
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
        """Create a rigging task for a 3D model.

        Args:
            model_url: URL of the 3D model to rig.

        Returns:
            Task ID for the rigging operation.
        """
        url = f"{self.base_url}/openapi/v1/rigging"
        payload = {"model_url": model_url}
        resp = requests.post(url, headers=self.headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data.get("result") or data.get("id")

    def poll_rigging_task(self, task_id: str, interval_sec: float = 5.0, timeout_sec: int = 1800) -> Dict[str, object]:
        """Poll rigging task until completion.

        Args:
            task_id: ID of the task to poll.
            interval_sec: Seconds between poll attempts.
            timeout_sec: Maximum seconds to wait before timeout.

        Returns:
            Task result dictionary with rigged model URL.

        Raises:
            TimeoutError: If task doesn't complete within timeout.
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
        """Create an animation task for a rigged model.

        Args:
            rig_task_id: ID of the completed rigging task.
            action_description: Description of the animation action.

        Returns:
            Task ID for the animation operation.
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

    def poll_animation_task(self, task_id: str, interval_sec: float = 5.0, timeout_sec: int = 1800) -> Dict[str, object]:
        """Poll animation task until completion.

        Args:
            task_id: ID of the task to poll.
            interval_sec: Seconds between poll attempts.
            timeout_sec: Maximum seconds to wait before timeout.

        Returns:
            Task result dictionary with animated model URL.

        Raises:
            TimeoutError: If task doesn't complete within timeout.
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
    """Image cropping utility using Landing AI object detection.

    Attributes:
        url: Landing AI API endpoint URL.
        headers: Request headers with authorization.
        target_image_path: Path to the target image for cropping.
    """

    def __init__(self, api_key: str, target_image_path: str) -> None:
        """Initialize the image cropper.

        Args:
            api_key: Landing AI API key.
            target_image_path: Path to the image to crop from.
        """
        self.url = "https://api.va.landing.ai/v1/tools/agentic-object-detection"
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.target_image_path = target_image_path

    def crop_image_by_text(self, object_name: str) -> Dict[str, object]:
        """Detect and get bounding box for an object by text description.

        Args:
            object_name: Name/description of the object to detect.

        Returns:
            API response with detected bounding boxes.
        """
        files = {"image": open(self.target_image_path, "rb")}
        data = {"prompts": object_name, "model": "agentic"}
        response = requests.post(self.url, files=files, data=data, headers=self.headers)
        return response.json()


def download_meshy_asset(object_name: str, description: str) -> Dict[str, object]:
    """Download a Meshy text-to-3D asset.

    Creates a preview, refines it, and downloads the final GLB model.

    Args:
        object_name: Name for the downloaded asset file.
        description: Text description of the 3D model to generate.

    Returns:
        Dictionary with status and output containing path and model URL.
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
        with open(f'{_meshy_api.save_dir}/meshy.log', 'a') as f:
            f.write(f"Preview ID: {preview_id}\n")

        preview_task = _meshy_api.poll_text_to_3d(preview_id, interval_sec=5, timeout_sec=900)
        if preview_task.get("status") != "SUCCEEDED":
            return {"status": "error", "output": f"Preview failed: {preview_task.get('status')}"}
        final_task = preview_task

        logging.info(f"[Meshy] Starting refine for preview task: {preview_id}")
        refine_id = _meshy_api.create_text_to_3d_refine(preview_id)
        with open(f'{_meshy_api.save_dir}/meshy.log', 'a') as f:
            f.write(f"Refine ID: {refine_id}\n")
            
        refine_task = _meshy_api.poll_text_to_3d(refine_id, interval_sec=5, timeout_sec=1800)
        if refine_task.get("status") != "SUCCEEDED":
            return {"status": "error", "output": f"Refine failed: {refine_task.get('status')}"}
        final_task = refine_task

        model_urls = (final_task or {}).get("model_urls", {}) or {}
        candidate_keys = ["glb"]
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


def download_meshy_asset_from_image(object_name: str, image_path: str, prompt: Optional[str] = None) -> Dict[str, object]:
    """Download a Meshy image-to-3D asset.

    Generates a 3D model from an input image and downloads the GLB file.

    Args:
        object_name: Name for the downloaded asset file.
        image_path: Path to the input image.
        prompt: Optional text prompt to guide generation.

    Returns:
        Dictionary with status and output containing path and model URL.
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
        with open(f'{_meshy_api.save_dir}/meshy.log', 'a') as f:
            f.write(f"Preview ID: {preview_id}\n")

        preview_task = _meshy_api.poll_image_to_3d(preview_id, interval_sec=5, timeout_sec=900)
        if preview_task.get("status") != "SUCCEEDED":
            return {"status": "error", "output": f"Image-to-3D preview failed: {preview_task.get('status')}"}
        final_task = preview_task

        model_urls = (final_task or {}).get("model_urls", {}) or {}
        candidate_keys = ["glb"]
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


def create_rigged_character(model_url: str, object_name: str) -> Dict[str, object]:
    """Create a rigged character from a 3D model.

    Submits a rigging task and downloads the rigged GLB model.

    Args:
        model_url: URL of the 3D model to rig.
        object_name: Name for the downloaded asset file.

    Returns:
        Dictionary with status and output containing task ID, path, and URL.
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
        with open(f'{_meshy_api.save_dir}/meshy.log', 'a') as f:
            f.write(f"Rig task ID: {rig_task_id}\n")

        rig_task = _meshy_api.poll_rigging_task(rig_task_id, interval_sec=5, timeout_sec=1800)
        if rig_task.get("status") != "SUCCEEDED":
            return {"status": "error", "output": f"Rigging failed: {rig_task.get('status')}"}

        result = rig_task.get("result", {})
        rigged_model_url = result.get("rigged_character_glb_url")
        if not rigged_model_url:
            return {"status": "error", "output": "No rigged model URL found in result"}

        local_path = _meshy_api.download_model_url(rigged_model_url, f"rigged_{object_name}.glb")
        logging.info(f"[Meshy] Downloading rigged model to: {local_path}")

        return {
            'status': 'success',
            'output': {'task_id': rig_task_id, 'path': local_path, 'model_url': rigged_model_url}
        }

    except Exception as e:
        logging.error(f"Failed to create rigged character: {e}")
        return {"status": "error", "output": str(e)}


def create_animated_character(rig_task_id: str, action_description: str, object_name: str) -> Dict[str, object]:
    """Create an animated character from a rigged model.

    Submits an animation task and downloads the animated GLB model.

    Args:
        rig_task_id: ID of the completed rigging task.
        action_description: Description of the animation action.
        object_name: Name for the downloaded asset file.

    Returns:
        Dictionary with status and output containing task ID, path, and URL.
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
        with open(f'{_meshy_api.save_dir}/meshy.log', 'a') as f:
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


def create_rigged_and_animated_character(model_url: str, action_description: str, object_name: str) -> Dict[str, object]:
    """Create a rigged and animated character in one workflow.

    Combines rigging and animation tasks to produce a fully animated
    character model.

    Args:
        model_url: URL of the 3D model to process.
        action_description: Description of the animation action.
        object_name: Name for the downloaded asset file.

    Returns:
        Dictionary with status and output from the animation task.
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
        return animation_result

    except Exception as e:
        logging.error(f"Failed to create rigged and animated character: {e}")
        return {"status": "error", "output": str(e)}
    
@mcp.tool()
def initialize(args: Dict[str, object]) -> Dict[str, object]:
    """Initialize the Meshy asset generation tool.

    Args:
        args: Configuration dictionary with 'meshy_api_key', 'output_dir',
            and optionally 'va_api_key', 'target_image_path', 'assets_dir'.

    Returns:
        Dictionary with status and tool configurations on success.
    """
    global _image_cropper
    global _meshy_api
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
            _meshy_api = MeshyAPI(meshy_api_key, save_dir, previous_assets_dir)
        return {
            "status": "success",
            "output": {"text": ["Meshy initialize completed"], "tool_configs": tool_configs}
        }

    except Exception as e:
        return {"status": "error", "output": {"text": [str(e)]}}
    
@mcp.tool()
def get_better_object(
    thought: str,
    object_name: str,
    reference_type: str = 'text',
    object_description: Optional[str] = None,
    rig_and_animate: bool = False,
    action_description: Optional[str] = None
) -> Dict[str, object]:
    """Generate and download a high-quality 3D asset.

    Args:
        thought: Reasoning about the object to download.
        object_name: Name of the object (e.g., 'chair', 'table').
        reference_type: 'text' for description-based or 'image' for image-based.
        object_description: Detailed description if reference_type is 'text'.
        rig_and_animate: Whether to add rigging and animation.
        action_description: Action verb for animation (e.g., 'walk', 'run').

    Returns:
        Dictionary with status and path to downloaded asset.
    """
    # thought is used by the model for reasoning but not in execution
    _ = thought

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
        
        model_url = static_result.get('output', {}).get('model_url', None)
        dynamic_result = create_rigged_and_animated_character(model_url=model_url, action_description=action_description, object_name=object_name)
        if dynamic_result.get('status') == 'success':
            return {"status": "success", "output": {"text": ["Successfully generated dynamic asset, downloaded to: " + dynamic_result.get('output', {}).get('path', '')]}}
        else:
            return {"status": "error", "output": {"text": [dynamic_result.get('output', {}).get('text', ['Failed to generate dynamic asset'])]}}
    
    except Exception as e:
        return {"status": "error", "output": {"text": [str(e)]}}

def main() -> None:
    """Run the MCP server or execute test mode."""
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        meshy_api_key = os.getenv("MESHY_API_KEY")
        va_api_key = os.getenv("VA_API_KEY")
        save_dir = "output/test/meshy"
        previous_assets_dir = "data/static_scene/christmas1/assets"

        init_payload = {
            "meshy_api_key": meshy_api_key,
            "va_api_key": va_api_key,
            "output_dir": save_dir,
            "assets_dir": previous_assets_dir,
            "target_image_path": "data/static_scene/christmas1/target.png",
        }
        result = initialize(init_payload)
        print("initialize result:", result)

        result = get_better_object(
            thought="Testing cat generation",
            object_name="cat",
            reference_type="text",
            object_description="Realistic domestic cat with gray and white coat.",
            rig_and_animate=True,
            action_description="walk",
        )
        print("download_object result:", result)
    else:
        mcp.run()


if __name__ == "__main__":
    main()