import os
import sys
import shutil

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import bpy
import json
import time
from pathlib import Path
import logging
from openai import OpenAI
import subprocess
from utils.blender.get_scene_info import get_scene_info, add_assets
from agents.utils import get_image_base64

system_prompt = """You are a 3D spatial perception assistant, skilled at inferring the three-dimensional structure of an image. I will now give you an image depicting a 3D scene, the scene's initial coordinate settings (do not output these objects), and a list of objects to be inferred (you will simply output the objects in the list in the code). Based on this information, infer the position of each object in the list.

For example, if the coordinate range of a wall is min(-2, 2, 0) and max(2, 2, 3), then the coordinates of an object hanging on this wall should fall within this range (i.e., x-2 to 2, y-2, and z-0 to 3). If the object is in the center of the wall, its coordinates should be (0, 2, 1.5).

First, output a reasoning process for each object, for example, "Object A is in the center of the wall, and the wall's coordinate range is xxx, so its coordinates are yyy." Then, output the code according to the bpy code format, for example:
```python
import bpy

bpy.data.objects['object_1'].location = (0,0,0)
bpy.data.objects['object_2'].location = (0,1,0)
```"""

def initialize_3d_scene_from_image(client: OpenAI, model: str, task_name: str, image_path: str, blender_path: str) -> dict:
    """
    Initialize a 3D scene from an input image
    
    Args:
        image_path: Input image path
        blender_path: Blender file path
        
    Returns:
        dict: Dictionary containing scene information
    """
    # 将image转换为base64
    image_base64 = get_image_base64(image_path)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "text", "text": 'Object list: ' + str(add_assets[task_name])},
            {"type": "text", "text": 'Initial image:'},
            {"type": "image_url", "image_url": {"url": image_base64}},
            {"type": "text", "text": 'Initial scene information (Do NOT include these objects in the output.): ' + get_scene_info(task_name, blender_path)}
        ]}
    ]
    # Generate init code
    code_response = client.chat.completions.create(model=model, messages=messages)
    
    messages.append({"role": "assistant", "content": code_response.choices[0].message.content})
    
    with open(os.path.join(os.path.dirname(blender_path), f"init_code.json"), 'w', encoding='utf-8') as f:
        json.dump(messages, f, indent=2, ensure_ascii=False)
    
    code_path = os.path.dirname(blender_path) + f"/start.py"
    init_code = code_response.choices[0].message.content
    if '```python' in init_code:
        init_code = init_code.split('```python')[1].split('```')[0].strip()
        with open(code_path, 'w', encoding='utf-8') as f:
            f.write(init_code)
    else:
        raise ValueError("Init code is not a valid Python code")
    
    return {
        "status": "success",
        "message": f"3D scene initialized successfully",
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

if __name__ == "__main__":
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    task_name = 'level4-2'
    map_task = {
        'level4-1': 'christmas1',
        'level4-2': 'meeting2',
        'level4-3': 'outdoor3',
    }
    initialize_3d_scene_from_image(client, "o4-mini", "level4-2", f"data/blendergym_hard/level4/{map_task[task_name]}/renders/goal/visprompt1.png", f"data/blendergym_hard/level4/{map_task[task_name]}/blender_file.blend")