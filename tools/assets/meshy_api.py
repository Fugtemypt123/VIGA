"""Meshy API client for 3D asset generation.

Provides classes for interacting with the Meshy API for text-to-3D,
image-to-3D, rigging, and animation capabilities.
"""

import base64
import json
import logging
import os
import re
import time
from typing import Dict, List, Optional

import requests


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
