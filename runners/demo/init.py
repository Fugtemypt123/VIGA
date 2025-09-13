import os
import bpy
import json
import time
from pathlib import Path
import logging
from openai import OpenAI
import subprocess

system_prompt = """You are a 3D scene designer. You are good at initializing the scene's background, walls, lights, and other information. Now I will give you a picture that describes the scene, and ask you to write a Blender code to initialize various scene information. Note: Be very careful when initializing your scene and camera coordinates so they match the image settings as closely as possible."""

def initialize_3d_scene_from_image(client: OpenAI, model: str, image_path: str, output_dir: str = "output/demo") -> dict:
    """
    Initialize a 3D scene from an input image
    
    Args:
        image_path: Input image path
        output_dir: Output directory
        
    Returns:
        dict: Dictionary containing scene information
    """
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate scene filename
        timestamp = int(time.time())
        image_name = Path(image_path).stem
        scene_name = f"scene_{image_name}_{timestamp}"
        blender_file_path = os.path.join(output_dir, f"{scene_name}.blend")
        
        # Create new Blender scene
        bpy.ops.wm.read_homefile(app_template="")
        bpy.ops.wm.save_mainfile(filepath=blender_file_path)
        
        # Generate init code
        code_response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": 'Initial Image:'},
                    {"type": "image_url", "image_url": {"url": image_path}}
                ]}
            ]
        )
        
        # parse init code
        init_code = code_response.choices[0].message.content
        if '```python' in init_code:
            init_code = init_code.split('```python')[1].split('```')[0].strip()
            code_path = os.path.join(output_dir, f"{scene_name}.py")
            with open(code_path, 'w', encoding='utf-8') as f:
                f.write(init_code)
        else:
            raise ValueError("Init code is not a valid Python code")
        
        # execute init_code
        cmd = [
            'utils/blender/infinigen/blender/blender',
            "--background", blender_file_path,
            "--python", code_path
        ]
        subprocess.run(cmd, check=True)
        
        # Create scene info file
        scene_info = {
            "scene_name": scene_name,
            "blender_file_path": blender_file_path,
            "code_path": code_path,
            "source_image": image_path,
            "created_at": timestamp,
            "objects": [],
            "target_objects": []  # Will be populated in subsequent loops
        }
        
        # Save scene info
        info_file_path = os.path.join(output_dir, f"{scene_name}_info.json")
        with open(info_file_path, 'w', encoding='utf-8') as f:
            json.dump(scene_info, f, indent=2, ensure_ascii=False)
        
        print(f"✓ 3D scene initialization completed: {scene_name}")
        print(f"  - Blender file: {blender_file_path}")
        print(f"  - Scene info: {info_file_path}")
        
        return {
            "status": "success",
            "message": f"3D scene '{scene_name}' initialized successfully",
            "scene_name": scene_name,
            "blender_file_path": blender_file_path,
            "scene_info_path": info_file_path,
            "code_path": code_path,
        }
        
    except Exception as e:
        logging.error(f"Failed to initialize 3D scene: {e}")
        return {
            "status": "error",
            "error": str(e)
        }

def load_scene_info(scene_info_path: str) -> dict:
    """
    Load scene information
    
    Args:
        scene_info_path: Scene info file path
        
    Returns:
        dict: Scene information
    """
    try:
        with open(scene_info_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Failed to load scene info: {e}")
        return {}

def update_scene_info(scene_info_path: str, updates: dict) -> dict:
    """
    Update scene information
    
    Args:
        scene_info_path: Scene info file path
        updates: Information to update
        
    Returns:
        dict: Update result
    """
    try:
        # Load existing information
        scene_info = load_scene_info(scene_info_path)
        if not scene_info:
            return {"status": "error", "error": "Failed to load scene info"}
        
        # Update information
        scene_info.update(updates)
        
        # Save updated information
        with open(scene_info_path, 'w', encoding='utf-8') as f:
            json.dump(scene_info, f, indent=2, ensure_ascii=False)
        
        return {
            "status": "success",
            "message": "Scene info updated successfully",
            "scene_info": scene_info
        }
        
    except Exception as e:
        logging.error(f"Failed to update scene info: {e}")
        return {"status": "error", "error": str(e)}
