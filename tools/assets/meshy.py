"""Meshy MCP Server for 3D Asset Generation.

Provides tools for generating high-quality 3D assets using Meshy API,
including text-to-3D, image-to-3D, rigging, and animation capabilities.
"""

import logging
import os
import sys
from typing import Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from tools.assets.meshy_api import ImageCropper, MeshyAPI

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
_image_cropper: Optional[ImageCropper] = None
_meshy_api: Optional[MeshyAPI] = None


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
