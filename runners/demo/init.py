import os
import bpy
import json
import time
from pathlib import Path
import logging

def initialize_3d_scene_from_image(image_path: str, output_dir: str = "output/demo/scenes") -> dict:
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
        
        # Create new Blender scene
        bpy.ops.wm.read_homefile(app_template="")
        
        # Generate scene filename
        timestamp = int(time.time())
        image_name = Path(image_path).stem
        scene_name = f"scene_{image_name}_{timestamp}"
        blender_file_path = os.path.join(output_dir, f"{scene_name}.blend")
        
        # Set basic scene parameters
        scene = bpy.context.scene
        scene.render.resolution_x = 1024
        scene.render.resolution_y = 1024
        scene.render.engine = 'CYCLES'
        
        # Create basic camera
        bpy.ops.object.camera_add(location=(5, -5, 3))
        camera = bpy.context.active_object
        camera.name = "Camera1"
        camera.rotation_euler = (1.1, 0, 0.785)
        scene.camera = camera
        
        # Create basic lighting
        bpy.ops.object.light_add(type='SUN', location=(5, 5, 10))
        sun_light = bpy.context.active_object
        sun_light.name = "Sun"
        sun_light.data.energy = 3.0
        
        # Add ambient light
        bpy.ops.object.light_add(type='AREA', location=(-5, -5, 5))
        area_light = bpy.context.active_object
        area_light.name = "AreaLight"
        area_light.data.energy = 2.0
        area_light.data.size = 10
        
        # Create a simple plane as ground
        bpy.ops.mesh.primitive_plane_add(size=20, location=(0, 0, 0))
        ground = bpy.context.active_object
        ground.name = "Ground"
        
        # Save initial scene
        bpy.ops.wm.save_mainfile(filepath=blender_file_path)
        
        # Create scene info file
        scene_info = {
            "scene_name": scene_name,
            "blender_file_path": blender_file_path,
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
            "scene_info": scene_info
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
