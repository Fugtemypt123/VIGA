import os
import json
import logging
from typing import Optional, Dict, Any
from .meshy import add_meshy_asset, add_meshy_asset_from_image
from openai import OpenAI
from PIL import Image

def crop_image(image_path: str, object_name: str) -> str:
    """
    Crop image
    """
    import requests

    url = "https://api.va.landing.ai/v1/tools/agentic-object-detection"
    files = {
        "image": open(image_path, "rb")
    }
    data = {
        "prompts": f"{object_name}",
        "model": "agentic"
     }
    headers = {
        "Authorization": "Basic " + os.getenv("VA_API_KEY")
    }
    response = requests.post(url, files=files, data=data, headers=headers)
    return response.json()

class AssetGenerator:
    """3D Asset Generator, supports generating assets from text and images"""
    
    def __init__(self, blender_path: str, client: OpenAI, model: str = None):
        """
        Initialize asset generator
        
        Args:
            blender_path: Blender file path
            client: OpenAI client
            model: OpenAI model
        """
        self.blender_path = blender_path
        self.client = client
        self.model = model
        if not self.client:
            raise ValueError("OpenAI client is required. Pass client parameter.")
        if not self.model:
            raise ValueError("OpenAI model is required. Pass model parameter.")
    
    def ask_vlm_for_object_description(self, object_name: str, context_image_path: str = None) -> str:
        """
        Ask VLM for detailed text description of the object
        
        Args:
            object_name: Object name
            context_image_path: Context image path (optional)
            
        Returns:
            str: Detailed object description
        """
        system_prompt = f"""You are an expert in fine-grained image description. Now I give you a picture and the name of an object in it. Please describe the object as detailed and accurate as possible (including shape, color, material and other attributes). Output in the following format: 'Object description: [your description]'"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [{"type": "image_url", "image_url": {"url": context_image_path}}, {"type": "text", "text": f"Object name: {object_name}"}]}
            ]
        )
        response = response.choices[0].message.content
        if f'Object description:' in response:
            response = response.split(f'Object description:')[1].strip()
        else:
            raise ValueError("Response is not a valid object description")
        return response
    
    def generate_asset_from_text(self, object_name: str, location: str = "0,0,0", scale: float = 1.0) -> Dict[str, Any]:
        """
        Generate 3D asset from text
        
        Args:
            object_name: Object name
            location: Asset position "x,y,z"
            scale: Scale factor
            
        Returns:
            dict: Generation result
        """
        try:
            # Get detailed description
            description = self.ask_vlm_for_object_description(object_name)
            print(f"[AssetGenerator] Generating text-to-3D asset for '{object_name}': {description}")
            
            # Call functions in meshy.py
            result = add_meshy_asset(
                description=description,
                blender_path=self.blender_path,
                location=location,
                scale=scale,
                api_key=self.api_key,
                refine=True,
                save_dir="output/demo/assets",
                filename=f"text_{object_name}"
            )
            
            return {
                "type": "text_to_3d",
                "object_name": object_name,
                "description": description,
                "result": result
            }
            
        except Exception as e:
            logging.error(f"Failed to generate text asset: {e}")
            return {
                "type": "text_to_3d",
                "object_name": object_name,
                "status": "error",
                "error": str(e)
            }
    
    def generate_asset_from_image(self, object_name: str, image_path: str, location: str = "0,0,0", scale: float = 1.0) -> Dict[str, Any]:
        """
        Generate 3D asset from image
        
        Args:
            object_name: Object name
            image_path: Input image path
            location: Asset position "x,y,z"
            scale: Scale factor
            
        Returns:
            dict: Generation result
        """
        try:
            # Get detailed description as prompt
            cropped_image_bbox = crop_image(image_path, object_name)
            # crop the image with PIL
            cropped_image = Image.open(image_path).crop(cropped_image_bbox)
            # Save the cropped image to a temporary file
            # TODO: see the output format here
            cropped_image_path = f"output/demo/assets/cropped_images/{object_name}.png"
            cropped_image.save(cropped_image_path)
            print(f"[AssetGenerator] Generating image-to-3D asset for '{object_name}' from image: {image_path}")
            
            # Call functions in meshy.py
            result = add_meshy_asset_from_image(
                image_path=cropped_image_path,
                blender_path=self.blender_path,
                location=location,
                scale=scale,
                prompt=f"A 3D model of {object_name}",
                api_key=self.api_key,
                refine=True,
                save_dir="output/demo/assets",
                filename=f"image_{object_name}"
            )
            
            return {
                "type": "image_to_3d",
                "object_name": object_name,
                "image_path": image_path,
                "description": f"A 3D model of {object_name}",
                "result": result
            }
            
        except Exception as e:
            logging.error(f"Failed to generate image asset: {e}")
            return {
                "type": "image_to_3d",
                "object_name": object_name,
                "status": "error",
                "error": str(e)
            }
    
    def generate_both_assets(self, object_name: str, image_path: str = None, location: str = "0,0,0", scale: float = 1.0) -> Dict[str, Any]:
        """
        Generate both text and image 3D assets simultaneously
        
        Args:
            object_name: Object name
            image_path: Input image path (optional)
            location: Asset position "x,y,z"
            scale: Scale factor
            
        Returns:
            dict: Contains both asset generation results
        """
        try:
            print(f"[AssetGenerator] Generating both text and image assets for '{object_name}'")
            
            # Generate text asset
            text_result = self.generate_asset_from_text(object_name, location, scale)
            
            # Generate image asset (if image path provided)
            image_result = None
            if image_path and os.path.exists(image_path):
                # Adjust position for image asset (avoid overlap)
                image_location = f"{float(location.split(',')[0]) + 2},{location.split(',')[1]},{location.split(',')[2]}"
                image_result = self.generate_asset_from_image(object_name, image_path, image_location, scale)
            else:
                print(f"[AssetGenerator] Warning: Image path not provided or file not found: {image_path}")
                image_result = {
                    "type": "image_to_3d",
                    "object_name": object_name,
                    "status": "skipped",
                    "message": "Image path not provided or file not found"
                }
            
            return {
                "object_name": object_name,
                "text_asset": text_result,
                "image_asset": image_result,
                "status": "success" if text_result.get("result", {}).get("status") == "success" else "partial"
            }
            
        except Exception as e:
            logging.error(f"Failed to generate both assets: {e}")
            return {
                "object_name": object_name,
                "status": "error",
                "error": str(e)
            }
    
    def get_asset_summary(self, generation_result: Dict[str, Any]) -> str:
        """
        Get summary of asset generation results
        
        Args:
            generation_result: Asset generation result
            
        Returns:
            str: Result summary
        """
        try:
            object_name = generation_result.get("object_name", "Unknown")
            text_asset = generation_result.get("text_asset", {})
            image_asset = generation_result.get("image_asset", {})
            
            summary_parts = [f"Asset generation for '{object_name}':"]
            
            # Text asset result
            if text_asset.get("type") == "text_to_3d":
                text_result = text_asset.get("result", {})
                if text_result.get("status") == "success":
                    summary_parts.append(f"  ✓ Text-to-3D: {text_result.get('message', 'Success')}")
                else:
                    summary_parts.append(f"  ✗ Text-to-3D: {text_result.get('error', 'Failed')}")
            
            # Image asset result
            if image_asset.get("type") == "image_to_3d":
                if image_asset.get("status") == "skipped":
                    summary_parts.append(f"  ⏭️ Image-to-3D: {image_asset.get('message', 'Skipped')}")
                else:
                    image_result = image_asset.get("result", {})
                    if image_result.get("status") == "success":
                        summary_parts.append(f"  ✓ Image-to-3D: {image_result.get('message', 'Success')}")
                    else:
                        summary_parts.append(f"  ✗ Image-to-3D: {image_result.get('error', 'Failed')}")
            
            return "\n".join(summary_parts)
            
        except Exception as e:
            return f"Failed to generate summary: {e}"
