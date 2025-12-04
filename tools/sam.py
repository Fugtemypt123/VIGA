# sam.py
import os
import sys
import logging
import numpy as np
from PIL import Image
from typing import Optional
from mcp.server.fastmcp import FastMCP

# Add paths for sam3 and sam3d imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "utils", "sam3"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "utils", "sam3d", "notebook"))

# tool_configs for agent (only the function w/ @mcp.tool)
tool_configs = [
    {
        "type": "function",
        "function": {
            "name": "extract_3d_object",
            "description": "Extract a 3D object from an image using SAM3 for mask extraction and SAM-3D for 3D reconstruction. Returns the GLB file path and position information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "object_name": {
                        "type": "string",
                        "description": "The name of the object to extract from the image (e.g., 'chair', 'table', 'lamp', etc.)"
                    }
                },
                "required": ["object_name"]
            }
        }
    }
]

mcp = FastMCP("sam-executor")
_sam3_model = None
_sam3_processor = None
_sam3d_inference = None
_target_image_path = None
_output_dir = None


class SAM3Wrapper:
    """Wrapper for SAM3 model and processor"""
    def __init__(self):
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        
        logging.info("[SAM3] Loading SAM3 model...")
        self.model = build_sam3_image_model()
        self.processor = Sam3Processor(self.model)
        logging.info("[SAM3] SAM3 model loaded successfully")
    
    def extract_mask(self, image_path: str, object_name: str) -> Optional[np.ndarray]:
        """
        Extract mask for the specified object from the image
        
        Args:
            image_path: Path to the image
            object_name: Name of the object to extract
            
        Returns:
            numpy array mask (boolean or uint8), or None if failed
        """
        try:
            # Load image
            image = Image.open(image_path)
            
            # Set image in processor
            inference_state = self.processor.set_image(image)
            
            # Prompt with text
            output = self.processor.set_text_prompt(state=inference_state, prompt=object_name)
            
            # Get masks, boxes, and scores
            masks = output["masks"]
            boxes = output["boxes"]
            scores = output["scores"]
            
            if masks is None or len(masks) == 0:
                logging.warning(f"[SAM3] No masks found for object: {object_name}")
                return None
            
            # Select the mask with highest score
            best_idx = np.argmax(scores) if len(scores) > 0 else 0
            mask = masks[best_idx]
            
            # Convert mask to numpy array if it's a tensor
            if hasattr(mask, 'cpu'):
                mask = mask.cpu().numpy()
            elif hasattr(mask, 'numpy'):
                mask = mask.numpy()
            
            # Ensure mask is boolean
            if mask.dtype != bool:
                mask = mask > 0.5
            
            logging.info(f"[SAM3] Successfully extracted mask for {object_name} with score {scores[best_idx]}")
            return mask.astype(np.uint8) * 255
            
        except Exception as e:
            logging.error(f"[SAM3] Failed to extract mask: {e}")
            return None


class SAM3DWrapper:
    """Wrapper for SAM-3D inference"""
    def __init__(self, config_path: str):
        from inference import Inference
        
        logging.info("[SAM-3D] Loading SAM-3D model...")
        self.inference = Inference(config_path, compile=False)
        logging.info("[SAM-3D] SAM-3D model loaded successfully")
    
    def reconstruct_3d(self, image_path: str, mask: np.ndarray, seed: int = 42) -> Optional[dict]:
        """
        Reconstruct 3D object from image and mask
        
        Args:
            image_path: Path to the image
            mask: Mask array (numpy array, uint8, 0-255)
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary with 'glb', 'translation', 'rotation', 'scale', or None if failed
        """
        try:
            # Load image
            image = Image.open(image_path)
            image_array = np.array(image)
            
            # Ensure mask is uint8 and in 0-255 range
            if mask.dtype != np.uint8:
                if mask.max() <= 1.0:
                    mask = (mask * 255).astype(np.uint8)
                else:
                    mask = mask.astype(np.uint8)
            
            # Ensure mask is 2D (H, W)
            if len(mask.shape) == 3:
                if mask.shape[2] == 1:
                    mask = mask[:, :, 0]
                else:
                    # Take first channel or convert to grayscale
                    mask = mask[:, :, 0]
            
            # Ensure mask is the right shape
            if mask.shape[:2] != image_array.shape[:2]:
                # Resize mask to match image
                from PIL import Image as PILImage
                mask_pil = PILImage.fromarray(mask, mode='L')
                mask_pil = mask_pil.resize((image_array.shape[1], image_array.shape[0]), resample=PILImage.NEAREST)
                mask = np.array(mask_pil)
            
            # Ensure mask is boolean-like (0 or 255)
            mask = (mask > 127).astype(np.uint8) * 255
            
            # Run inference - Inference.__call__ will merge mask to RGBA internally
            output = self.inference(image_array, mask, seed=seed)
            
            # Extract results
            result = {
                "translation": None,
                "rotation": None,
                "scale": None,
                "glb": None
            }
            
            # Get translation, rotation, scale
            if "translation" in output:
                translation = output["translation"]
                if hasattr(translation, 'cpu'):
                    translation = translation.cpu().numpy()
                elif hasattr(translation, 'numpy'):
                    translation = translation.numpy()
                # Handle tensor shape: might be (1, 3) or (3,)
                if isinstance(translation, np.ndarray):
                    translation = translation.flatten()
                    if len(translation) == 3:
                        result["translation"] = translation.tolist()
                    else:
                        result["translation"] = translation.tolist() if translation.size > 0 else None
                else:
                    result["translation"] = translation
            
            if "rotation" in output:
                rotation = output["rotation"]
                if hasattr(rotation, 'cpu'):
                    rotation = rotation.cpu().numpy()
                elif hasattr(rotation, 'numpy'):
                    rotation = rotation.numpy()
                # Handle quaternion: might be (1, 4) or (4,)
                if isinstance(rotation, np.ndarray):
                    rotation = rotation.flatten()
                    if len(rotation) == 4:
                        result["rotation"] = rotation.tolist()
                    else:
                        result["rotation"] = rotation.tolist() if rotation.size > 0 else None
                else:
                    result["rotation"] = rotation
            
            if "scale" in output:
                scale = output["scale"]
                if hasattr(scale, 'cpu'):
                    scale = scale.cpu().numpy()
                elif hasattr(scale, 'numpy'):
                    scale = scale.numpy()
                # Handle scale: might be scalar or (1, 3) or (3,)
                if isinstance(scale, np.ndarray):
                    scale = scale.flatten()
                    if len(scale) == 1:
                        result["scale"] = float(scale[0])
                    elif len(scale) == 3:
                        result["scale"] = scale.tolist()
                    else:
                        result["scale"] = scale.tolist() if scale.size > 0 else None
                else:
                    result["scale"] = float(scale) if scale is not None else None
            
            # Get GLB mesh if available
            if "glb" in output and output["glb"] is not None:
                result["glb"] = output["glb"]
            
            logging.info("[SAM-3D] Successfully reconstructed 3D object")
            return result
            
        except Exception as e:
            logging.error(f"[SAM-3D] Failed to reconstruct 3D object: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return None


@mcp.tool()
def initialize(args: dict) -> dict:
    """
    Initialize SAM3 and SAM-3D models
    
    Args:
        args: Dictionary containing:
            - target_image_path: Path to the target image
            - output_dir: Directory to save output files
            - sam3d_config_path: (Optional) Path to SAM-3D config file
    """
    global _sam3_model, _sam3_processor, _sam3d_inference, _target_image_path, _output_dir
    
    try:
        target_image_path = args.get("target_image_path")
        output_dir = args.get("output_dir")
        sam3d_config_path = args.get("sam3d_config_path")
        
        if not target_image_path:
            return {"status": "error", "output": {"text": ["target_image_path is required"]}}
        
        if not os.path.exists(target_image_path):
            return {"status": "error", "output": {"text": [f"Target image not found: {target_image_path}"]}}
        
        if not output_dir:
            return {"status": "error", "output": {"text": ["output_dir is required"]}}
        
        # Set default SAM-3D config path if not provided
        if not sam3d_config_path:
            sam3d_config_path = os.path.join(
                os.path.dirname(__file__), "..", "utils", "sam3d", "checkpoints", "hf", "pipeline.yaml"
            )
        
        if not os.path.exists(sam3d_config_path):
            return {"status": "error", "output": {"text": [f"SAM-3D config not found: {sam3d_config_path}"]}}
        
        # Store paths
        _target_image_path = target_image_path
        _output_dir = output_dir
        os.makedirs(_output_dir, exist_ok=True)
        
        # Initialize SAM3
        logging.info("[SAM] Initializing SAM3...")
        _sam3_model = SAM3Wrapper()
        
        # Initialize SAM-3D
        logging.info("[SAM] Initializing SAM-3D...")
        _sam3d_inference = SAM3DWrapper(sam3d_config_path)
        
        logging.info("[SAM] Initialization completed successfully")
        return {"status": "success", "output": {"text": ["SAM initialize completed"], "tool_configs": tool_configs}}
    
    except Exception as e:
        logging.error(f"[SAM] Initialization failed: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {"status": "error", "output": {"text": [str(e)]}}


@mcp.tool()
def extract_3d_object(object_name: str) -> dict:
    """
    Extract 3D object from the target image using SAM3 and SAM-3D
    
    Args:
        object_name: Name of the object to extract
        
    Returns:
        Dictionary with status and output containing glb_path and position information
    """
    global _sam3_model, _sam3d_inference, _target_image_path, _output_dir
    
    try:
        if _sam3_model is None or _sam3d_inference is None:
            return {"status": "error", "output": {"text": ["SAM models not initialized. Call initialize first."]}}
        
        if _target_image_path is None:
            return {"status": "error", "output": {"text": ["Target image path not set. Call initialize first."]}}
        
        if _output_dir is None:
            return {"status": "error", "output": {"text": ["Output directory not set. Call initialize first."]}}
        
        # Step 1: Extract mask using SAM3
        logging.info(f"[SAM] Extracting mask for object: {object_name}")
        mask = _sam3_model.extract_mask(_target_image_path, object_name)
        
        if mask is None:
            return {"status": "error", "output": {"text": [f"Failed to extract mask for object: {object_name}"]}}
        
        # Step 2: Reconstruct 3D using SAM-3D
        logging.info(f"[SAM] Reconstructing 3D object: {object_name}")
        result = _sam3d_inference.reconstruct_3d(_target_image_path, mask, seed=42)
        
        if result is None:
            return {"status": "error", "output": {"text": [f"Failed to reconstruct 3D object: {object_name}"]}}
        
        # Step 3: Save GLB file if available
        glb_path = None
        if result.get("glb") is not None:
            glb_path = os.path.join(_output_dir, f"{object_name}.glb")
            try:
                # Export GLB file
                glb_obj = result["glb"]
                if hasattr(glb_obj, 'export'):
                    glb_obj.export(glb_path)
                    logging.info(f"[SAM] Saved GLB file to: {glb_path}")
                elif hasattr(glb_obj, 'vertices') and hasattr(glb_obj, 'faces'):
                    # Try using trimesh if it's a mesh-like object
                    try:
                        import trimesh
                        mesh = trimesh.Trimesh(vertices=glb_obj.vertices, faces=glb_obj.faces)
                        mesh.export(glb_path)
                        logging.info(f"[SAM] Saved GLB file to: {glb_path}")
                    except Exception as e2:
                        logging.warning(f"[SAM] Failed to export with trimesh: {e2}")
                        glb_path = None
                else:
                    logging.warning(f"[SAM] GLB object type: {type(glb_obj)}, does not have export method")
                    glb_path = None
            except Exception as e:
                logging.error(f"[SAM] Failed to save GLB file: {e}")
                import traceback
                logging.error(traceback.format_exc())
                glb_path = None
        
        # Prepare output
        output_text = [f"Successfully extracted 3D object: {object_name}"]
        if glb_path:
            output_text.append(f"GLB file saved to: {glb_path}")
        
        output_data = {
            "glb_path": glb_path,
            "translation": result.get("translation"),
            "rotation": result.get("rotation"),
            "scale": result.get("scale")
        }
        
        return {
            "status": "success",
            "output": {
                "text": output_text,
                "data": output_data
            }
        }
    
    except Exception as e:
        logging.error(f"[SAM] Failed to extract 3D object: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {"status": "error", "output": {"text": [str(e)]}}


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        # Test mode
        target_image_path = "data/static_scene/christmas/target.png"
        output_dir = "output/test/sam"
        sam3d_config_path = os.path.join(
            os.path.dirname(__file__), "..", "utils", "sam3d", "checkpoints", "hf", "pipeline.yaml"
        )
        
        init_payload = {
            "target_image_path": target_image_path,
            "output_dir": output_dir,
            "sam3d_config_path": sam3d_config_path
        }
        result = initialize(init_payload)
        print("initialize result:", result)
        
        if result.get("status") == "success":
            result = extract_3d_object(object_name="chair")
            print("extract_3d_object result:", result)
    else:
        mcp.run()


if __name__ == "__main__":
    main()

