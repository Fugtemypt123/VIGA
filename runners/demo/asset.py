import os
import json
import logging
from typing import Optional, Dict, Any
from .meshy import add_meshy_asset, add_meshy_asset_from_image

class AssetGenerator:
    """3D Asset Generator, supports generating assets from text and images"""
    
    def __init__(self, blender_path: str, api_key: str = None):
        """
        Initialize asset generator
        
        Args:
            blender_path: Blender file path
            api_key: Meshy API key (optional, defaults to environment variable)
        """
        self.blender_path = blender_path
        self.api_key = api_key or os.getenv("MESHY_API_KEY")
        if not self.api_key:
            raise ValueError("Meshy API key is required. Set MESHY_API_KEY environment variable or pass api_key parameter.")
    
    def ask_vlm_for_object_description(self, object_name: str, context_image_path: str = None) -> str:
        """
        Ask VLM for detailed text description of the object
        
        Args:
            object_name: Object name
            context_image_path: Context image path (optional)
            
        Returns:
            str: Detailed object description
        """
        try:
            # This should call VLM API, now using simple templates first
            # In actual implementation, you need to call OpenAI or other VLM services
            
            # Simple description templates (should use VLM generation in actual application)
            description_templates = {
                "chair": "A comfortable wooden chair with a high backrest and armrests, suitable for dining or office use",
                "table": "A sturdy wooden dining table with four legs and a smooth surface, perfect for family meals",
                "lamp": "A modern table lamp with a metal base and fabric shade, providing warm ambient lighting",
                "sofa": "A plush three-seater sofa with soft cushions and elegant fabric upholstery",
                "bookshelf": "A tall wooden bookshelf with multiple shelves for storing books and decorative items",
                "coffee_table": "A low wooden coffee table with a glass top, perfect for placing drinks and magazines",
                "bed": "A comfortable double bed with a wooden headboard and soft mattress",
                "desk": "A spacious wooden desk with drawers and a smooth work surface for studying or working",
                "television": "A modern flat-screen television with a sleek black frame and remote control",
                "plant": "A healthy green houseplant in a ceramic pot, adding natural beauty to the room",
                "clock": "A classic wall clock with Roman numerals and brass hands",
                "vase": "An elegant ceramic vase with a smooth finish, perfect for displaying flowers",
                "painting": "A beautiful framed artwork with vibrant colors and artistic composition",
                "mirror": "A large rectangular mirror with a wooden frame, perfect for a bedroom or hallway",
                "rug": "A soft area rug with geometric patterns and warm colors",
                "curtain": "Heavy fabric curtains with elegant drapes and curtain rods",
                "pillow": "A soft decorative pillow with colorful fabric and comfortable filling",
                "candle": "An aromatic scented candle in a decorative holder with a warm flame",
                "book": "A hardcover book with an interesting cover and pages ready to be read",
                "cup": "A ceramic coffee cup with a handle, perfect for morning coffee or tea"
            }
            
            # If object name is in templates, use template
            if object_name.lower() in description_templates:
                return description_templates[object_name.lower()]
            
            # Otherwise generate generic description
            return f"A detailed 3D model of a {object_name}, with realistic textures and proper proportions"
            
        except Exception as e:
            logging.error(f"Failed to get object description: {e}")
            return f"A 3D model of {object_name}"
    
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
                save_dir="output/demo/assets/text",
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
            description = self.ask_vlm_for_object_description(object_name)
            print(f"[AssetGenerator] Generating image-to-3D asset for '{object_name}' from image: {image_path}")
            
            # Call functions in meshy.py
            result = add_meshy_asset_from_image(
                image_path=image_path,
                blender_path=self.blender_path,
                location=location,
                scale=scale,
                prompt=description,
                api_key=self.api_key,
                refine=True,
                save_dir="output/demo/assets/image",
                filename=f"image_{object_name}"
            )
            
            return {
                "type": "image_to_3d",
                "object_name": object_name,
                "image_path": image_path,
                "description": description,
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
