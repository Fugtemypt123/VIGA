# blender_server.py
import math
import os
import sys
from pathlib import Path
import logging
from mcp.server.fastmcp import FastMCP
import subprocess
import json
import traceback
from typing import Optional, Dict, Any

# tool config for agent (only the function w/ @mcp.tools)
tool_configs = [
    {
        "type": "function",
        "function": {
            "name": "initialize_viewpoint",
            "description": "Adds a viewpoint to observe the listed objects. The viewpoints are added to the four corners of the bounding box of the listed objects. This tool returns the positions and rotations of the four viewpoint cameras, as well as the rendered images of the four cameras. You can use these information to set the camera to a good initial position and orientation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "object_names": {
                        "type": "array", 
                        "description": "The names of the objects to observe. Objects must exist in the scene (you can check the scene information to see if they exist). If you want to observe the whole scene, you can pass an empty list.",
                        "items": {
                            "type": "string",
                            "description": "The name of the object to observe."
                        }
                    }
                },
                "required": ["object_names"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_camera",
            "description": "Set the current active camera to the given location and rotation",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "array", 
                        "description": "The location of the camera (in world coordinates)",
                        "items": {
                            "type": "number",
                            "description": "The location of the camera (in world coordinates)"
                        }
                    },
                    "rotation_euler": {
                        "type": "array", 
                        "description": "The rotation of the camera (in euler angles)",
                        "items": {
                            "type": "number",
                            "description": "The rotation of the camera (in euler angles)"
                        }
                    }
                },
                "required": ["location", "rotation_euler"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "investigate",
            "description": "Investigate the scene by the current camera. You can zoom, move, and focus on the object you want to investigate.",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {"type": "string", "choices": ["zoom", "move", "focus"], "description": "The operation to perform."},
                    "object_name": {"type": "string", "description": "If the operation is focus, you need to provide the name of the object to focus on. The object must exist in the scene."},
                    "direction": {"type": "string", "choices": ["up", "down", "left", "right", "in", "out"], "description": "If the operation is move or zoom, you need to provide the direction to move or zoom."}
                },
                "required": ["operation"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_object_visibility",
            "description": "Set the visibility of the objects in the scene. You can decide to show or hide the objects. You do not need to mention all the objects here, the objects you do not metioned will keep their original visibility.",
            "parameters": {
                "type": "object",
                "properties": {
                    "show_object_list": {
                        "type": "array", 
                        "description": "The names of the objects to show. Objects must exist in the scene.",
                        "items": {
                            "type": "string",
                            "description": "The name of the object to show."
                        }
                    },
                    "hide_object_list": {
                        "type": "array", 
                        "description": "The names of the objects to hide. Objects must exist in the scene.",
                        "items": {
                            "type": "string",
                            "description": "The name of the object to hide."
                        }
                    }
                },
                "required": ["show_object_list", "hide_object_list"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_keyframe",
            "description": "Set the scene to a specific frame number for observation",
            "parameters": {
                "type": "object",
                "properties": {
                    "frame_number": {"type": "integer", "description": "The specific frame number to set the scene to."}
                },
                "required": ["frame_number"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_scene_info",
            "description": "Get the scene information",
        }
    }
]

# Create global MCP instance
mcp = FastMCP("scene-server")

# Global tool instance
_investigator = None

# ======================
# Camera investigator (Fixed: save path first then load)
# ======================

# Local lightweight Executor for running Blender with our verifier script
class Executor:
    def __init__(self,
                 blender_command: str,
                 blender_file: str,
                 blender_script: str,
                 script_save: str,
                 render_save: str,
                 blender_save: Optional[str] = None,
                 gpu_devices: Optional[str] = None):
        self.blender_command = blender_command
        self.blender_file = blender_file
        self.blender_script = blender_script
        self.script_path = Path(script_save)
        self.render_path = Path(render_save)
        self.blender_save = blender_save
        self.gpu_devices = gpu_devices
        self.count = 0

        self.script_path.mkdir(parents=True, exist_ok=True)
        self.render_path.mkdir(parents=True, exist_ok=True)

    def next_run_dir(self) -> Path:
        self.count += 1
        run_dir = self.render_path / f"{self.count}"
        run_dir.mkdir(parents=True, exist_ok=True)
        # Clean old images if any
        for p in run_dir.glob("*"):
            try:
                p.unlink()
            except Exception:
                pass
        return run_dir

    def _execute_blender(self, code_file: Path, run_dir: Path) -> Dict[str, Any]:
        cmd = [
            self.blender_command,
            "--background", self.blender_file,
            "--python", self.blender_script,
            "--", str(code_file), str(run_dir)
        ]
        if self.blender_save:
            cmd.append(self.blender_save)

        env = os.environ.copy()
        if self.gpu_devices:
            env["CUDA_VISIBLE_DEVICES"] = self.gpu_devices

        try:
            # Propagate render directory to scripts
            env["RENDER_DIR"] = str(run_dir)
            proc = subprocess.run(" ".join(cmd), shell=True, check=True, capture_output=True, text=True, env=env)
            imgs = sorted([str(p) for p in run_dir.glob("*") if p.suffix.lower() in [".png", ".jpg", ".jpeg"]])
            return {"status": "success", "output": {"image": imgs, "text": [proc.stdout]}}
        except subprocess.CalledProcessError as e:
            logging.error(f"Blender failed: {e.stderr}")
            return {"status": "error", "output": {"text": [e.stderr or e.stdout]}}

    def execute(self, full_code: str) -> Dict[str, Any]:
        run_dir = self.next_run_dir()
        code_file = self.script_path / f"{self.count}.py"
        with open(code_file, "w") as f:
            f.write(full_code)
        return self._execute_blender(code_file, run_dir)

class Investigator3D:
    def __init__(self, save_dir: str, blender_path: str, blender_command: str, blender_script: str, gpu_devices: str):
        self.blender_file = blender_path
        self.blender_command = blender_command
        self.base = Path(save_dir)
        self.base.mkdir(parents=True, exist_ok=True)
        
        # Create executor
        self.executor = Executor(
            blender_command=blender_command,
            blender_file=blender_path,
            blender_script=blender_script,  # Default script
            script_save=str(self.base / "scripts"),
            render_save=str(self.base / "renders"),
            blender_save=str(self.base / "current_scene.blend"),
            gpu_devices=gpu_devices
        )
        
        # State variables
        self.target = None
        self.radius = 5.0
        self.theta = 0.0
        self.phi = 0.0
        self.count = 0
        self.scene_info_cache = None

    def _generate_scene_info_script(self) -> str:
        """Generate script to get scene information"""
        return f'''import bpy
import json
import sys

# Get scene information
scene_info = {{"objects": [], "materials": [], "lights": [], "cameras": []}}

for obj in bpy.data.objects:
    if obj.type == 'CAMERA' or obj.type == 'LIGHT':
        continue
    scene_info["objects"].append({{
        "name": obj.name, 
        "type": obj.type,
        "location": list(obj.matrix_world.translation),
        "rotation": list(obj.rotation_euler),
        "scale": list(obj.scale),
        "visible": not (obj.hide_viewport or obj.hide_render)
    }})

for mat in bpy.data.materials:
    scene_info["materials"].append({{
        "name": mat.name,
        "use_nodes": mat.use_nodes,
        "diffuse_color": list(mat.diffuse_color),
    }})

for light in [o for o in bpy.data.objects if o.type == 'LIGHT']:
    scene_info["lights"].append({{
        "name": light.name,
        "type": light.data.type,
        "energy": light.data.energy,
        "color": list(light.data.color),
        "location": list(light.matrix_world.translation),
        "rotation": list(light.rotation_euler)
    }})

for cam in [o for o in bpy.data.objects if o.type == 'CAMERA']:
    scene = bpy.context.scene
    scene_info["cameras"].append({{
        "name": cam.name,
        "lens": cam.data.lens,
        "location": list(cam.matrix_world.translation),
        "rotation": list(cam.rotation_euler),
        "is_active": cam == scene.camera,
    }})

# Save to file for retrieval
with open("{self.base}/tmp/scene_info.json", "w") as f:
    json.dump(scene_info, f)

print("Scene info extracted successfully")
'''

    def _generate_render_script(self) -> str:
        """Generate script to render current scene once into RENDER_DIR/1.png"""
        return '''import bpy, os

render_dir = os.environ.get("RENDER_DIR", "/tmp")

# Basic render settings
bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.render.image_settings.file_format = 'PNG'
bpy.context.scene.render.resolution_x = 512
bpy.context.scene.render.resolution_y = 512

# Single render
bpy.context.scene.render.filepath = os.path.join(render_dir, "1.png")
bpy.ops.render.render(write_still=True)

print("Render completed to", bpy.context.scene.render.filepath)
'''

    def _generate_camera_focus_script(self, object_name: str) -> str:
        """Generate script to focus camera on object"""
        return f'''import bpy, os
import math

# Get target object
target_obj = bpy.data.objects.get('{object_name}')
if not target_obj:
    raise ValueError(f"Object '{object_name}' not found")

# Get camera
camera = bpy.context.scene.camera
if not camera:
    # Find first camera
    cameras = [obj for obj in bpy.data.objects if obj.type == 'CAMERA']
    if cameras:
        camera = cameras[0]
        bpy.context.scene.camera = camera
    else:
        raise ValueError("No camera found in scene")

# Calculate camera position relative to target
target_pos = target_obj.matrix_world.translation
camera_pos = camera.matrix_world.translation
distance = (camera_pos - target_pos).length

# Set up track-to constraint
constraint = None
for c in camera.constraints:
    if c.type == 'TRACK_TO':
        constraint = c
        break

if not constraint:
    constraint = camera.constraints.new('TRACK_TO')

constraint.target = target_obj
constraint.track_axis = 'TRACK_NEGATIVE_Z'
constraint.up_axis = 'UP_Y'

# Render after focus
render_dir = os.environ.get("RENDER_DIR", "/tmp")
bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.render.image_settings.file_format = 'PNG'
bpy.context.scene.render.resolution_x = 512
bpy.context.scene.render.resolution_y = 512
bpy.context.scene.render.filepath = os.path.join(render_dir, "1.png")
bpy.ops.render.render(write_still=True)

print(f"Camera focused on object '{object_name}' and rendered")
'''

    def _generate_camera_set_script(self, location: list, rotation_euler: list) -> str:
        """Generate script to set camera position and rotation"""
        return f'''import bpy, os

# Get camera
camera = bpy.context.scene.camera
if not camera:
    # Find first camera
    cameras = [obj for obj in bpy.data.objects if obj.type == 'CAMERA']
    if cameras:
        camera = cameras[0]
        bpy.context.scene.camera = camera
    else:
        raise ValueError("No camera found in scene")

# Set camera location and rotation
camera.location = {location}
camera.rotation_euler = {rotation_euler}

# Render after setting camera
render_dir = os.environ.get("RENDER_DIR", "/tmp")
bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.render.image_settings.file_format = 'PNG'
bpy.context.scene.render.resolution_x = 512
bpy.context.scene.render.resolution_y = 512
bpy.context.scene.render.filepath = os.path.join(render_dir, "1.png")
bpy.ops.render.render(write_still=True)

print(f"Camera set to location {location} and rotation {rotation_euler} and rendered")
'''

    def _generate_visibility_script(self, show_objects: list, hide_objects: list) -> str:
        """Generate script to set object visibility and render once"""
        return f'''import bpy, os

show_list = {show_objects}
hide_list = {hide_objects}

# Apply visibility changes
for obj in bpy.data.objects:
    if obj.name in hide_list:
        obj.hide_viewport = True
        obj.hide_render = True
    if obj.name in show_list:
        obj.hide_viewport = False
        obj.hide_render = False

# Render after visibility update
render_dir = os.environ.get("RENDER_DIR", "/tmp")
bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.render.image_settings.file_format = 'PNG'
bpy.context.scene.render.resolution_x = 512
bpy.context.scene.render.resolution_y = 512
bpy.context.scene.render.filepath = os.path.join(render_dir, "1.png")
bpy.ops.render.render(write_still=True)

print("Visibility updated and rendered: show", show_list, ", hide", hide_list)
'''

    def _generate_camera_move_script(self, target_obj_name: str, radius: float, theta: float, phi: float) -> str:
        """Generate script to move camera around target object"""
        return f'''import bpy, os
import math

# Get target object
target_obj = bpy.data.objects.get('{target_obj_name}')
if not target_obj:
    raise ValueError(f"Target object '{target_obj_name}' not found")

# Get camera
camera = bpy.context.scene.camera
if not camera:
    cameras = [obj for obj in bpy.data.objects if obj.type == 'CAMERA']
    if cameras:
        camera = cameras[0]
        bpy.context.scene.camera = camera

# Calculate new camera position
target_pos = target_obj.matrix_world.translation
x = {radius} * math.cos({phi}) * math.cos({theta})
y = {radius} * math.cos({phi}) * math.sin({theta})
z = {radius} * math.sin({phi})

new_pos = (target_pos.x + x, target_pos.y + y, target_pos.z + z)
camera.matrix_world.translation = new_pos

# Render after moving
render_dir = os.environ.get("RENDER_DIR", "/tmp")
bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.render.image_settings.file_format = 'PNG'
bpy.context.scene.render.resolution_x = 512
bpy.context.scene.render.resolution_y = 512
bpy.context.scene.render.filepath = os.path.join(render_dir, "1.png")
bpy.ops.render.render(write_still=True)

print("Camera moved to position and rendered:", new_pos)
'''

    def _generate_keyframe_script(self, frame_number: int) -> str:
        """Generate script to set frame number"""
        return f'''import bpy, os

scene = bpy.context.scene
current_frame = scene.frame_current

# Ensure frame number is within valid range
target_frame = max(scene.frame_start, min(scene.frame_end, {frame_number}))
scene.frame_set(target_frame)

# Render after frame change
render_dir = os.environ.get("RENDER_DIR", "/tmp")
bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.render.image_settings.file_format = 'PNG'
bpy.context.scene.render.resolution_x = 512
bpy.context.scene.render.resolution_y = 512
bpy.context.scene.render.filepath = os.path.join(render_dir, "1.png")
bpy.ops.render.render(write_still=True)

print("Changed to frame", target_frame, "(was", current_frame, ") and rendered")
'''

    def _generate_viewpoint_script(self, object_names: list) -> str:
        """Generate script to initialize viewpoints around objects"""
        return f'''import bpy, os
import math
from mathutils import Vector

object_names = {object_names}
objects = []

# Find objects to observe
if not object_names:
    # If empty list, observe all mesh objects
    for obj in bpy.data.objects:
        if obj.type == 'MESH' and obj.name not in ['Ground', 'Plane']:
            objects.append(obj)
else:
    # Find specified objects by name
    for obj_name in object_names:
        obj = bpy.data.objects.get(obj_name)
        if obj:
            objects.append(obj)

if not objects:
    raise ValueError("No valid objects found")

# Calculate bounding box
min_x = min_y = min_z = float('inf')
max_x = max_y = max_z = float('-inf')

for obj in objects:
    bbox_corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    for corner in bbox_corners:
        min_x = min(min_x, corner.x)
        min_y = min(min_y, corner.y)
        min_z = min(min_z, corner.z)
        max_x = max(max_x, corner.x)
        max_y = max(max_y, corner.y)
        max_z = max(max_z, corner.z)

center_x = (min_x + max_x) / 2
center_y = (min_y + max_y) / 2
center_z = (min_z + max_z) / 2

size_x = max_x - min_x
size_y = max_y - min_y
size_z = max_z - min_z
max_size = max(size_x, size_y, size_z)
margin = max_size * 0.5

camera_positions = [
    (center_x - margin, center_y - margin, center_z + margin),
    (center_x + margin, center_y - margin, center_z + margin),
    (center_x - margin, center_y + margin, center_z + margin),
    (center_x + margin, center_y + margin, center_z + margin)
]

# Get camera
camera = bpy.context.scene.camera
if not camera:
    cameras = [obj for obj in bpy.data.objects if obj.type == 'CAMERA']
    if cameras:
        camera = cameras[0]
        bpy.context.scene.camera = camera

# Store original position
original_location = camera.location.copy()
original_rotation = camera.rotation_euler.copy()
camera_infos = []

# Set up viewpoints and render each
render_dir = os.environ.get("RENDER_DIR", "/tmp")
for i, pos in enumerate(camera_positions):
    camera.location = pos
    camera.rotation_euler = (math.radians(60), 0, math.radians(45))
    
    # Look at center
    direction = Vector((center_x, center_y, center_z)) - camera.location
    camera.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
    # Render per viewpoint
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.resolution_x = 512
    bpy.context.scene.render.resolution_y = 512
    bpy.context.scene.render.filepath = os.path.join(render_dir, str(i+1)+".png")
    bpy.ops.render.render(write_still=True)
    
    camera_infos.append({{"camera_parameters": {{"location": list(camera.location), "rotation": list(camera.rotation_euler)}}, "image_path": bpy.context.scene.render.filepath}})
    
with open("{self.base}/tmp/camera_info.json", "w") as f:
    json.dump(camera_infos, f)
    
# Restore original position
camera.location = original_location
camera.rotation_euler = original_rotation

print("Viewpoints initialized and rendered for", len(objects), "objects")
'''

    def _execute_script(self, script_code: str, description: str = "") -> dict:
        """Execute a blender script and return results"""
        try:
            result = self.executor.execute(full_code=script_code)
            
            # Update blender_background to the saved blend file
            if result.get("status") == "success":
                if self.executor.blender_save:
                    self.executor.blender_file = self.executor.blender_save
                
            return result
        except Exception as e:
            logging.error(f"Script execution failed: {e}")
            return {"status": "error", "output": {"text": [str(e)]}}

    def _render(self):
        """Render current scene and return image path and camera parameters"""
        render_script = self._generate_render_script()
        result = self._execute_script(render_script, "Render current scene")
        
        if result.get("status") == "success":
            # The executor handles the actual rendering and returns image paths
            images = result.get("output", {}).get("image", [])
            if images:
                return {
                    "image_path": images[0] if isinstance(images, list) else images,
                    "camera_parameters": "Camera parameters extracted from render"
                }
        return {"image_path": None, "camera_parameters": "Render failed"}
    
    def get_info(self) -> dict:
        """Get scene information by executing a script"""
        try:
            # Use cached info if available
            if self.scene_info_cache:
                return self.scene_info_cache
                
            os.makedirs(f"{self.base}/tmp", exist_ok=True)
            script = self._generate_scene_info_script()
            result = self._execute_script(script, "Extract scene information")
            
            # Try to read the scene info from the temporary file
            try:
                import json
                with open(f"{self.base}/tmp/scene_info.json", "r") as f:
                    scene_info = json.load(f)
                    self.scene_info_cache = scene_info
                    return scene_info
            except Exception:
                pass
                
            # Fallback: return empty info if file reading fails
            return {"objects": [], "materials": [], "lights": [], "cameras": []}
        except Exception as e:
            logging.error(f"scene info error: {e}")
            return {}

    def focus_on_object(self, object_name: str) -> dict:
        """Focus camera on a specific object"""
        self.target = object_name  # Store object name instead of object reference
        
        # Generate and execute focus script
        focus_script = self._generate_camera_focus_script(object_name)
        result = self._execute_script(focus_script, f"Focus camera on object {object_name}")
        
        if result.get("status") == "success":
            # default orbit params
            self.radius = 5.0
            self.theta = 0.0
            self.phi = 0.0
            images = result.get("output", {}).get("image", [])
            return {
                "image_path": images[0] if images else None,
                "camera_parameters": "Focus completed"
            }
        else:
            raise ValueError(f"Failed to focus on object {object_name}")

    def zoom(self, direction: str) -> dict:
        """Zoom camera in or out"""
        if direction == 'in':
            self.radius = max(1, self.radius-3)
        elif direction == 'out':
            self.radius += 3
        return self._update_and_render()

    def move_camera(self, direction: str) -> dict:
        """Move camera around target object"""
        if not self.target:
            raise ValueError("No target object set. Call focus first.")
            
        step = self.radius
        theta_step = step / (self.radius*math.cos(self.phi)) if math.cos(self.phi) != 0 else 0.1
        phi_step = step / self.radius
        
        if direction=='up': 
            self.phi = min(math.pi/2-0.1, self.phi+phi_step)
        elif direction=='down': 
            self.phi = max(-math.pi/2+0.1, self.phi-phi_step)
        elif direction=='left': 
            self.theta -= theta_step
        elif direction=='right': 
            self.theta += theta_step
            
        return self._update_and_render()

    def _update_and_render(self) -> dict:
        """Update camera position and render"""
        if not self.target:
            return self._render()
            
        # Generate script to move camera
        move_script = self._generate_camera_move_script(self.target, self.radius, self.theta, self.phi)
        result = self._execute_script(move_script, f"Move camera around {self.target}")
        
        if result.get("status") == "success":
            images = result.get("output", {}).get("image", [])
            return {"image_path": images[0] if images else None, "camera_parameters": "Move completed"}
        else:
            return {"image_path": None, "camera_parameters": "Camera move failed"}

    def set_camera(self, location: list, rotation_euler: list) -> dict:
        """Set camera position and rotation"""
        script = self._generate_camera_set_script(location, rotation_euler)
        return self._execute_script(script, f"Set camera to location {location} and rotation {rotation_euler}")

    def initialize_viewpoint(self, object_names: list) -> dict:
        """Initialize viewpoints around specified objects"""
        try:
            script = self._generate_viewpoint_script(object_names)
            _ = self._execute_script(script, f"Initialize viewpoints for objects: {object_names}")
            result = {"status": "success", "output": {"image": [], "text": []}}
            with open(f"{self.base}/tmp/camera_info.json", "r") as f:
                camera_infos = json.load(f)
                for i, camera_info in enumerate(camera_infos):
                    result['output']['image'].append(camera_info['image_path'])
                    result['output']['text'].append(f"[Viewpoint {i+1}] Camera parameters: {camera_info['camera_parameters']}")
            return result
        except Exception as e:
            return {'status': 'error', 'output': {'text': [str(e)]}}

    def set_keyframe(self, frame_number: int) -> dict:
        """Set scene to a specific frame"""
        try:
            script = self._generate_keyframe_script(frame_number)
            result = self._execute_script(script, f"Set frame to {frame_number}")
            return result
        except Exception as e:
            return {'status': 'error', 'output': {'text': [str(e)]}}

@mcp.tool()
def initialize(args: dict) -> dict:
    """
    Initialize 3D scene investigation tool.
    """
    global _investigator
    try:
        save_dir = args.get("output_dir") + "/investigator/"
        blender_script = os.path.dirname(args.get("blender_script")) + "/verifier_script.py"
        _investigator = Investigator3D(save_dir, str(args.get("blender_file")), str(args.get("blender_command")), blender_script, str(args.get("gpu_devices")))
        return {"status": "success", "output": {"text": ["Investigator3D initialized successfully"], "tool_configs": tool_configs}}
    except Exception as e:
        return {"status": "error", "output": {"text": [str(e)]}}

def focus(object_name: str) -> dict:
    global _investigator
    if _investigator is None:
        return {"status": "error", "output": {"text": ["Investigator3D not initialized. Call initialize_investigator first."]}}

    try:
        # Object existence check is now handled in the script execution
        result = _investigator.focus_on_object(object_name)
        return {
            "status": "success", 
            "output": {"image": [result["image_path"]], "text": [f"Camera parameters: {result['camera_parameters']}"]}
        }
    except Exception as e:
        logging.error(f"Focus failed: {e}")
        return {"status": "error", "output": {"text": [str(e)]}}

def zoom(direction: str) -> dict:
    global _investigator
    if _investigator is None:
        return {"status": "error", "output": {"text": ["Investigator3D not initialized. Call initialize_investigator first."]}}

    try:
        # Check if there is a target object
        if _investigator.target is None:
            return {"status": "error", "output": {"text": ["No target object set. Call focus first."]}}

        result = _investigator.zoom(direction)
        return {
            "status": "success", 
            "output": {'image': [result["image_path"]], 'text': [f"Camera parameters: {result['camera_parameters']}"]}
        }
    except Exception as e:
        logging.error(f"Zoom failed: {e}")
        return {"status": "error", "output": {"text": [str(e)]}}

def move(direction: str) -> dict:
    global _investigator
    if _investigator is None:
        return {"status": "error", "output": {"text": ["Investigator3D not initialized. Call initialize_investigator first."]}}

    try:
        # Check if there is a target object
        if _investigator.target is None:
            return {"status": "error", "output": {"text": ["No target object set. Call focus first."]}}

        result = _investigator.move_camera(direction)
        return {
            "status": "success", 
            "output": {'image': [result["image_path"]], 'text': [f"Camera parameters: {result['camera_parameters']}"]}
        }
    except Exception as e:
        logging.error(f"Move failed: {e}")
        return {"status": "error", "output": {"text": [str(e)]}}

@mcp.tool()
def get_scene_info() -> dict:
    try:
        global _investigator
        if _investigator is None:
            return {"status": "error", "output": {"text": ["SceneInfo not initialized. Call initialize first."]}}
        info = _investigator.get_info()
        return {"status": "success", "output": {"text": [str(info)], "json": [info]}}
    except Exception as e:
        logging.error(f"Failed to get scene info: {e}")
        return {"status": "error", "output": {"text": [str(e)]}}

@mcp.tool()
def initialize_viewpoint(object_names: list) -> dict:
    global _investigator
    if _investigator is None:
        return {"status": "error", "output": {"text": ["Investigator3D not initialized. Call initialize_investigator first."]}}
    try:
        result = _investigator.initialize_viewpoint(object_names)
        return result
    except Exception as e:
        logging.error(f"Add viewpoint failed: {e}")
        return {"status": "error", "output": {"text": [str(e)]}}

@mcp.tool()
def investigate(operation: str, object_name: str = None, direction: str = None) -> dict:
    if operation == "focus":
        if not object_name:
            return {"status": "error", "output": {"text": ["object_name is required for focus"]}}
        return focus(object_name=object_name)
    elif operation == "zoom":
        if direction not in ("in", "out"):
            return {"status": "error", "output": {"text": ["direction must be 'in' or 'out' for zoom"]}}
        return zoom(direction=direction)
    elif operation == "move":
        if direction not in ("up", "down", "left", "right"):
            return {"status": "error", "output": {"text": ["direction must be one of up/down/left/right for move"]}}
        return move(direction=direction)
    else:
        return {"status": "error", "output": {"text": [f"Unknown operation: {operation}"]}}

@mcp.tool()
def set_object_visibility(show_object_list: list = None, hide_object_list: list = None) -> dict:
    global _investigator
    if _investigator is None:
        return {"status": "error", "output": {"text": ["Investigator3D not initialized. Call initialize_investigator first."]}}
    try:
        show_object_list = show_object_list or []
        hide_object_list = hide_object_list or []
        
        # Generate and execute visibility script
        script = _investigator._generate_visibility_script(show_object_list, hide_object_list)
        result = _investigator._execute_script(script, f"Set visibility: show {show_object_list}, hide {hide_object_list}")
        
        if result.get("status") == "success":
            render_result = _investigator._render()
            return {"status": "success", "output": {'image': [render_result["image_path"]], 'text': [f"Camera parameters: {render_result['camera_parameters']}"]}}
        else:
            return result
    except Exception as e:
        logging.error(f"set_object_visibility failed: {e}")
        return {"status": "error", "output": {"text": [str(e)]}}

@mcp.tool()
def set_keyframe(frame_number: int) -> dict:
    global _investigator
    if _investigator is None:
        return {"status": "error", "output": {"text": ["Investigator3D not initialized. Call initialize_investigator first."]}}
    try:
        return _investigator.set_keyframe(frame_number)
    except Exception as e:
        logging.error(f"Set keyframe failed: {e}")
        return {"status": "error", "output": {"text": [str(e)]}, "status": "error"}
    
@mcp.tool()
def set_camera(location: list, rotation_euler: list) -> dict:
    global _investigator
    if _investigator is None:
        return {"status": "error", "output": {"text": ["Investigator3D not initialized. Call initialize_investigator first."]}}
    try:
        _investigator.set_camera(location, rotation_euler)
        return {"status": "success", "output": {"text": ["Successfully set the camera with the given location and rotation"]}}
    except Exception as e:
        logging.error(f"set_camera failed: {e}")
        return {"status": "error", "output": {"text": [str(e)]}}
    
@mcp.tool()
def reload_scene() -> dict:
    global _investigator
    if _investigator is None:
        return {"status": "error", "output": {"text": ["Investigator3D not initialized. Call initialize first."]}}

    # Reload the blender_background file
    _investigator.executor.blender_file = _investigator.blender_file
    return {"status": "success", "output": {"text": ["Scene reloaded successfully"]}}

# ======================
# Entry point and testing
# ======================

def main():
    # Check if script is run directly (for testing)
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        print("Running investigator tools test...")
        test_tools()
    else:
        # Run MCP server normally
        mcp.run()

def test_tools():
    """Test all investigator tool functions (read environment variable configuration)"""
    print("=" * 50)
    print("Testing Scene Tools")
    print("=" * 50)

    # Set test paths (read from environment variables)
    blender_file = os.getenv("BLENDER_FILE", "output/static_scene/20251018_012341/christmas1/blender_file.blend")
    test_save_dir = os.getenv("THOUGHT_SAVE", "output/test/investigator/")
    blender_command = os.getenv("BLENDER_COMMAND", "utils/blender/infinigen/blender/blender")
    blender_script = os.getenv("BLENDER_SCRIPT", "data/static_scene/verifier_script.py")
    gpu_devices = os.getenv("GPU_DEVICES", "0,1,2,3,4,5,6,7")
    
    # Check if blender file exists
    if not os.path.exists(blender_file):
        print(f"⚠ Blender file not found: {blender_file}")
        print("Skipping all tests.")
        return

    print(f"✓ Using blender file: {blender_file}")

    # Test 1: Initialize investigation tool
    print("\n1. Testing initialize...")
    args = {"output_dir": test_save_dir, "blender_file": blender_file, "blender_command": blender_command, "blender_script": blender_script, "gpu_devices": gpu_devices}
    result = initialize(args)
    print(f"Result: {result}")
        
    # Test 2: Get scene info
    print("\n2. Testing get_scene_info...")
    scene_info = get_scene_info()
    print(f"Result: {scene_info}")
    
    # Test 3: Reload scene
    print("\n3. Testing reload_scene...")
    reload_result = reload_scene()
    print(f"Result: {reload_result}")
    
    object_names = [obj['name'] for obj in scene_info['output']['json'][0]['objects']]
    print(f"Object names: {object_names}")
        
    # Test 4: Initialize viewpoint
    print("\n4. Testing initialize_viewpoint...")
    viewpoint_result = initialize_viewpoint(object_names=object_names)
    print(f"Result: {viewpoint_result}")
    raise NotImplementedError("Not implemented")

    # Test 5: Focus, zoom, move, set_keyframe if objects exist
    if object_names:
        first_object = object_names[0]
        print(f"\n5. Testing camera operations with object: {first_object}")
        
        # Test focus
        print("\n5.1. Testing focus...")
        focus_result = focus(object_name=first_object)
        print(f"Result: {focus_result}")

        # Test zoom
        print("\n5.2. Testing zoom...")
        zoom_result = zoom(direction="in")
        print(f"Result: {zoom_result}")

        # Test move
        print("\n5.3. Testing move...")
        move_result = move(direction="up")
        print(f"Result: {move_result}")

        # Test set_keyframe
        print("\n5.4. Testing set_keyframe...")
        keyframe_result = set_keyframe(frame_number=1)
        print(f"Result: {keyframe_result}")
    else:
        print("\n5. Skipping camera operations - no objects found in scene")

    print("\n" + "=" * 50)
    print("Test completed!")
    print("=" * 50)
    print(f"\nTest files saved to: {test_save_dir}")
    print("\nTo run the MCP server normally:")
    print("python tools/investigator.py")
    print("\nTo run tests:")
    print("BLENDER_FILE=/path/to.blend THOUGHT_SAVE=output/test/scene_test python tools/investigator.py --test")


if __name__ == "__main__":
    main()
    
{"objects": [{"name": "Ceiling", "type": "MESH", "location": [0.0, 0.0, 2.700000047683716], "rotation": [0.0, 0.0, 0.0], "scale": [2.5, 2.200000047683716, 1.0], "visible": True}, {"name": "Cone", "type": "MESH", "location": [1.899999976158142, -0.699999988079071, 1.0499999523162842], "rotation": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0], "visible": True}, {"name": "Cone.001", "type": "MESH", "location": [-1.9299999475479126, 0.3499999940395355, 0.949999988079071], "rotation": [0.0, 0.0, 0.13962633907794952], "scale": [1.0, 1.0, 1.0], "visible": True}, {"name": "Cone.002", "type": "MESH", "location": [0.8999999761581421, 2.180000066757202, 1.7000000476837158], "rotation": [1.5707963705062866, 0.0, 0.0], "scale": [1.0, 1.0, 1.0], "visible": True}, {"name": "Cone.003", "type": "MESH", "location": [1.059999942779541, 2.180000066757202, 1.7000000476837158], "rotation": [1.5707963705062866, 0.0, 0.0], "scale": [1.0, 1.0, 1.0], "visible": True}, {"name": "Cone.004", "type": "MESH", "location": [2.490000009536743, 1.4199999570846558, 1.100000023841858], "rotation": [0.0, 1.5707963705062866, 0.0], "scale": [1.0, 1.0, 1.0], "visible": True}, {"name": "Cone.005", "type": "MESH", "location": [2.490000009536743, 1.9500000476837158, 1.25], "rotation": [0.0, 1.5707963705062866, 0.0], "scale": [1.0, 1.0, 1.0], "visible": True}, {"name": "Cone.006", "type": "MESH", "location": [1.350000023841858, 1.899999976158142, 1.1699999570846558], "rotation": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0], "visible": True}, {"name": "Cone.007", "type": "MESH", "location": [1.850000023841858, 1.899999976158142, 1.1699999570846558], "rotation": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0], "visible": True}, {"name": "Cone.008", "type": "MESH", "location": [0.029999999329447746, -0.11999999731779099, 0.5899999737739563], "rotation": [0.0, 1.5707963705062866, 0.0], "scale": [1.0, 1.0, 1.0], "visible": True}, {"name": "CornerCap", "type": "MESH", "location": [2.4700000286102295, 2.1700000762939453, 1.350000023841858], "rotation": [0.0, 0.0, 0.0], "scale": [0.07000000029802322, 0.07000000029802322, 2.700000047683716], "visible": True}, {"name": "Cube", "type": "MESH", "location": [1.2999999523162842, 1.600000023841858, 0.5], "rotation": [0.0, 0.0, 0.0], "scale": [0.11999999731779099, 0.30000001192092896, 0.5], "visible": True}, {"name": "Cube.001", "type": "MESH", "location": [1.899999976158142, 1.600000023841858, 0.5], "rotation": [0.0, 0.0, 0.0], "scale": [0.11999999731779099, 0.30000001192092896, 0.5], "visible": True}, {"name": "Cube.002", "type": "MESH", "location": [1.600000023841858, 1.600000023841858, 1.0499999523162842], "rotation": [0.0, 0.0, 0.0], "scale": [0.5, 0.30000001192092896, 0.07999999821186066], "visible": True}, {"name": "Cube.003", "type": "MESH", "location": [1.600000023841858, 1.2999999523162842, 0.07999999821186066], "rotation": [0.0, 0.0, 0.0], "scale": [0.6000000238418579, 0.25, 0.07999999821186066], "visible": True}, {"name": "Cube.004", "type": "MESH", "location": [1.600000023841858, 1.6200000047683716, 0.550000011920929], "rotation": [0.0, 0.0, 0.0], "scale": [0.36000001430511475, 0.11999999731779099, 0.2800000011920929], "visible": True}, {"name": "Cube.005", "type": "MESH", "location": [1.899999976158142, 0.20000000298023224, 0.4000000059604645], "rotation": [0.0, 0.0, 0.0], "scale": [1.0, 0.44999998807907104, 0.4000000059604645], "visible": True}, {"name": "Cube.006", "type": "MESH", "location": [1.7000000476837158, 0.3499999940395355, 0.6200000047683716], "rotation": [0.0, 0.0, 0.0], "scale": [0.22499999403953552, 0.14000000059604645, 0.05000000074505806], "visible": True}, {"name": "Cube.007", "type": "MESH", "location": [1.899999976158142, -0.699999988079071, 0.2750000059604645], "rotation": [0.0, 0.0, 0.0], "scale": [0.25, 0.25, 0.2750000059604645], "visible": True}, {"name": "Cube.008", "type": "MESH", "location": [0.20000000298023224, -0.10000000149011612, 0.4399999976158142], "rotation": [0.0, 0.0, 0.0], "scale": [0.5, 0.30000001192092896, 0.03999999910593033], "visible": True}, {"name": "Cube.009", "type": "MESH", "location": [-2.2799999713897705, 0.44999998807907104, 0.1899999976158142], "rotation": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0], "visible": True}, {"name": "Cube.010", "type": "MESH", "location": [-2.2799999713897705, 0.44999998807907104, 0.3799999952316284], "rotation": [0.0, 0.0, 0.0], "scale": [0.019999999552965164, 0.1899999976158142, 0.004999999888241291], "visible": True}, {"name": "Cube.011", "type": "MESH", "location": [-2.2799999713897705, 0.44999998807907104, 0.3799999952316284], "rotation": [0.0, 0.0, 0.0], "scale": [0.1899999976158142, 0.019999999552965164, 0.004999999888241291], "visible": True}, {"name": "Cube.012", "type": "MESH", "location": [-1.7300000190734863, 0.20000000298023224, 0.1599999964237213], "rotation": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0], "visible": True}, {"name": "Cube.013", "type": "MESH", "location": [-1.7300000190734863, 0.20000000298023224, 0.3199999928474426], "rotation": [0.0, 0.0, 0.0], "scale": [0.019999999552965164, 0.1599999964237213, 0.004000000189989805], "visible": True}, {"name": "Cube.014", "type": "MESH", "location": [-1.7300000190734863, 0.20000000298023224, 0.3199999928474426], "rotation": [0.0, 0.0, 0.0], "scale": [0.1599999964237213, 0.019999999552965164, 0.004000000189989805], "visible": True}, {"name": "Cube.015", "type": "MESH", "location": [-1.600000023841858, 2.190000057220459, 1.350000023841858], "rotation": [0.0, 0.0, 0.0], "scale": [0.2800000011920929, 0.009999999776482582, 0.2199999988079071], "visible": True}, {"name": "Cube.016", "type": "MESH", "location": [-1.600000023841858, 2.194999933242798, 1.350000023841858], "rotation": [0.0, 0.0, 0.0], "scale": [0.25, 0.0020000000949949026, 0.1899999976158142], "visible": True}, {"name": "Cube.017", "type": "MESH", "location": [2.497999906539917, 0.8999999761581421, 1.5], "rotation": [0.0, 0.0, 5.235987663269043], "scale": [0.0020000000949949026, 0.10999999940395355, 0.004999999888241291], "visible": True}, {"name": "Cube.018", "type": "MESH", "location": [2.497999906539917, 0.8999999761581421, 1.5], "rotation": [0.0, 0.0, 1.0471975803375244], "scale": [0.0020000000949949026, 0.14000000059604645, 0.004999999888241291], "visible": True}, {"name": "Cube.019", "type": "MESH", "location": [1.350000023841858, 1.2999999523162842, 0.9200000166893005], "rotation": [0.0, 0.0, 0.0], "scale": [0.05999999865889549, 0.009999999776482582, 0.11999999731779099], "visible": True}, {"name": "Cube.020", "type": "MESH", "location": [1.350000023841858, 1.2999999523162842, 1.0199999809265137], "rotation": [0.0, 0.0, 0.0], "scale": [0.06499999761581421, 0.012000000104308128, 0.019999999552965164], "visible": True}, {"name": "Cube.021", "type": "MESH", "location": [1.600000023841858, 1.2999999523162842, 0.9200000166893005], "rotation": [0.0, 0.0, 0.0], "scale": [0.05999999865889549, 0.009999999776482582, 0.11999999731779099], "visible": True}, {"name": "Cube.022", "type": "MESH", "location": [1.600000023841858, 1.2999999523162842, 1.0199999809265137], "rotation": [0.0, 0.0, 0.0], "scale": [0.06499999761581421, 0.012000000104308128, 0.019999999552965164], "visible": True}, {"name": "Cube.023", "type": "MESH", "location": [1.850000023841858, 1.2999999523162842, 0.9200000166893005], "rotation": [0.0, 0.0, 0.0], "scale": [0.05999999865889549, 0.009999999776482582, 0.11999999731779099], "visible": True}, {"name": "Cube.024", "type": "MESH", "location": [1.850000023841858, 1.2999999523162842, 1.0199999809265137], "rotation": [0.0, 0.0, 0.0], "scale": [0.06499999761581421, 0.012000000104308128, 0.019999999552965164], "visible": True}, {"name": "Cylinder", "type": "MESH", "location": [0.20000000298023224, 0.0, 2.640000104904175], "rotation": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0], "visible": True}, {"name": "Cylinder.001", "type": "MESH", "location": [1.899999976158142, -0.699999988079071, 0.800000011920929], "rotation": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0], "visible": True}, {"name": "Cylinder.002", "type": "MESH", "location": [-0.05000000074505806, -0.30000001192092896, 0.1899999976158142], "rotation": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0], "visible": True}, {"name": "Cylinder.003", "type": "MESH", "location": [0.44999998807907104, -0.30000001192092896, 0.1899999976158142], "rotation": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0], "visible": True}, {"name": "Cylinder.004", "type": "MESH", "location": [-0.05000000074505806, 0.10000000149011612, 0.1899999976158142], "rotation": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0], "visible": True}, {"name": "Cylinder.005", "type": "MESH", "location": [0.44999998807907104, 0.10000000149011612, 0.1899999976158142], "rotation": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0], "visible": True}, {"name": "Cylinder.006", "type": "MESH", "location": [0.20000000298023224, -0.10000000149011612, 0.004000000189989805], "rotation": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0], "visible": True}, {"name": "Cylinder.007", "type": "MESH", "location": [0.20000000298023224, -0.10000000149011612, 0.0044999998062849045], "rotation": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0], "visible": True}, {"name": "Cylinder.008", "type": "MESH", "location": [0.20000000298023224, -0.10000000149011612, 0.004699999932199717], "rotation": [0.0, 0.0, 0.0], "scale": [0.8880000114440918, 0.8880000114440918, 1.0], "visible": True}, {"name": "Cylinder.009", "type": "MESH", "location": [-1.9299999475479126, 0.3499999940395355, 0.05000000074505806], "rotation": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0], "visible": True}, {"name": "Cylinder.010", "type": "MESH", "location": [-2.2799999713897705, 0.44999998807907104, 1.1200000047683716], "rotation": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0], "visible": True}, {"name": "Cylinder.011", "type": "MESH", "location": [2.490000009536743, 0.8999999761581421, 1.5], "rotation": [0.0, 1.5707963705062866, 0.0], "scale": [1.0, 1.0, 1.0], "visible": True}, {"name": "Cylinder.012", "type": "MESH", "location": [2.494999885559082, 0.8999999761581421, 1.5], "rotation": [0.0, 1.5707963705062866, 0.0], "scale": [1.0, 1.0, 1.0], "visible": True}, {"name": "Cylinder.013", "type": "MESH", "location": [2.484999895095825, 1.4199999570846558, 0.9800000190734863], "rotation": [0.0, 1.5707963705062866, 0.0], "scale": [1.0, 1.0, 1.0], "visible": True}, {"name": "Cylinder.014", "type": "MESH", "location": [2.484999895095825, 1.9500000476837158, 1.1299999952316284], "rotation": [0.0, 1.5707963705062866, 0.0], "scale": [1.0, 1.0, 1.0], "visible": True}, {"name": "Cylinder.015", "type": "MESH", "location": [1.350000023841858, 1.899999976158142, 1.0499999523162842], "rotation": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0], "visible": True}, {"name": "Cylinder.016", "type": "MESH", "location": [1.850000023841858, 1.899999976158142, 1.0499999523162842], "rotation": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0], "visible": True}, {"name": "Cylinder.017", "type": "MESH", "location": [1.600000023841858, 1.899999976158142, 1.149999976158142], "rotation": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0], "visible": True}, {"name": "Cylinder.018", "type": "MESH", "location": [0.07999999821186066, -0.07999999821186066, 0.5], "rotation": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0], "visible": True}, {"name": "Cylinder.019", "type": "MESH", "location": [0.25, -0.05000000074505806, 0.5], "rotation": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0], "visible": True}, {"name": "Cylinder.020", "type": "MESH", "location": [0.0, -0.11999999731779099, 0.6600000262260437], "rotation": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0], "visible": True}, {"name": "Cylinder.021", "type": "MESH", "location": [0.0, -0.11999999731779099, 0.6399999856948853], "rotation": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0], "visible": True}, {"name": "Floor", "type": "MESH", "location": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0], "scale": [2.5, 2.200000047683716, 1.0], "visible": True}, {"name": "Garland", "type": "CURVE", "location": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0], "visible": True}, {"name": "Icosphere", "type": "MESH", "location": [1.600000023841858, 1.899999976158142, 1.2200000286102295], "rotation": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0], "visible": True}, {"name": "Plane", "type": "MESH", "location": [0.0, 2.180000066757202, 1.350000023841858], "rotation": [1.5707963705062866, 0.0, 0.0], "scale": [2.5, 1.350000023841858, 1.0], "visible": True}, {"name": "Plane.001", "type": "MESH", "location": [2.4800000190734863, 0.0, 1.350000023841858], "rotation": [0.0, 1.5707963705062866, 0.0], "scale": [2.200000047683716, 1.350000023841858, 1.0], "visible": True}, {"name": "Plane.002", "type": "MESH", "location": [1.600000023841858, 1.6200000047683716, 0.44999998807907104], "rotation": [1.5707963705062866, 0.0, 0.0], "scale": [1.0, 1.0, 1.0], "visible": True}, {"name": "Sphere", "type": "MESH", "location": [-1.5115982294082642, 0.31339457631111145, 1.2000000476837158], "rotation": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0], "visible": True}, {"name": "Sphere.001", "type": "MESH", "location": [-1.6890978813171387, 0.6940438747406006, 1.2000000476837158], "rotation": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0], "visible": True}, {"name": "Sphere.002", "type": "MESH", "location": [-2.107499599456787, 0.7306492924690247, 1.2000000476837158], "rotation": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0], "visible": True}, {"name": "Sphere.003", "type": "MESH", "location": [-2.3484017848968506, 0.3866054117679596, 1.2000000476837158], "rotation": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0], "visible": True}, {"name": "Sphere.004", "type": "MESH", "location": [-2.1709020137786865, 0.0059561352245509624, 1.2000000476837158], "rotation": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0], "visible": True}, {"name": "Sphere.005", "type": "MESH", "location": [-1.752500295639038, -0.030649276450276375, 1.2000000476837158], "rotation": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0], "visible": True}, {"name": "Sphere.006", "type": "MESH", "location": [-1.4319026470184326, 0.3064221143722534, 1.3200000524520874], "rotation": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0], "visible": True}, {"name": "Sphere.007", "type": "MESH", "location": [-1.6432117223739624, 0.7595760226249695, 1.3200000524520874], "rotation": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0], "visible": True}, {"name": "Sphere.008", "type": "MESH", "location": [-2.1413090229034424, 0.8031538724899292, 1.3200000524520874], "rotation": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0], "visible": True}, {"name": "Sphere.009", "type": "MESH", "location": [-2.4280972480773926, 0.3935778737068176, 1.3200000524520874], "rotation": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0], "visible": True}, {"name": "Sphere.010", "type": "MESH", "location": [-2.2167880535125732, -0.05957602709531784, 1.3200000524520874], "rotation": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0], "visible": True}, {"name": "Sphere.011", "type": "MESH", "location": [-1.7186908721923828, -0.10315389931201935, 1.3200000524520874], "rotation": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0], "visible": True}, {"name": "Sphere.012", "type": "MESH", "location": [-1.4717503786087036, 0.30990836024284363, 1.4500000476837158], "rotation": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0], "visible": True}, {"name": "Sphere.013", "type": "MESH", "location": [-1.4826477766036987, 0.14139622449874878, 1.2599999904632568], "rotation": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0], "visible": True}, {"name": "Sphere.014", "type": "MESH", "location": [-1.4775670766830444, 0.47122901678085327, 1.3300000429153442], "rotation": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0], "visible": True}, {"name": "Sphere.015", "type": "MESH", "location": [-1.5296869277954102, 0.6627587676048279, 1.399999976158142], "rotation": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0], "visible": True}, {"name": "Sphere.016", "type": "MESH", "location": [0.9800000190734863, 2.197000026702881, 1.7899999618530273], "rotation": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0], "visible": True}, {"name": "Sphere.017", "type": "MESH", "location": [-2.0, 2.190000057220459, 2.4000000953674316], "rotation": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0], "visible": True}, {"name": "Sphere.018", "type": "MESH", "location": [-1.7423988580703735, 2.16595721244812, 2.374239921569824], "rotation": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0], "visible": True}, {"name": "Sphere.019", "type": "MESH", "location": [-1.484797716140747, 2.1419143676757812, 2.348479747772217], "rotation": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0], "visible": True}, {"name": "Sphere.020", "type": "MESH", "location": [-1.2271965742111206, 2.1178717613220215, 2.3227198123931885], "rotation": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0], "visible": True}, {"name": "Sphere.021", "type": "MESH", "location": [-0.9695954322814941, 2.0938289165496826, 2.296959638595581], "rotation": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0], "visible": True}, {"name": "Sphere.022", "type": "MESH", "location": [-0.5, 2.049999952316284, 2.25], "rotation": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0], "visible": True}, {"name": "Sphere.023", "type": "MESH", "location": [-0.24044868350028992, 2.049999952316284, 2.2347323894500732], "rotation": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0], "visible": True}, {"name": "Sphere.024", "type": "MESH", "location": [0.019102632999420166, 2.049999952316284, 2.2194645404815674], "rotation": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0], "visible": True}, {"name": "Sphere.025", "type": "MESH", "location": [0.27865391969680786, 2.049999952316284, 2.2041969299316406], "rotation": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0], "visible": True}, {"name": "Sphere.026", "type": "MESH", "location": [0.5382052659988403, 2.049999952316284, 2.1889290809631348], "rotation": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0], "visible": True}, {"name": "Sphere.027", "type": "MESH", "location": [0.797756552696228, 2.049999952316284, 2.173661470413208], "rotation": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0], "visible": True}, {"name": "Sphere.028", "type": "MESH", "location": [1.2000000476837158, 2.049999952316284, 2.1500000953674316], "rotation": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0], "visible": True}, {"name": "Sphere.029", "type": "MESH", "location": [1.451940894126892, 2.037402868270874, 2.2129852771759033], "rotation": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0], "visible": True}, {"name": "Sphere.030", "type": "MESH", "location": [1.7038817405700684, 2.024805784225464, 2.275970458984375], "rotation": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0], "visible": True}, {"name": "Sphere.031", "type": "MESH", "location": [-0.20000000298023224, 2.069999933242798, 2.259999990463257], "rotation": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0], "visible": True}, {"name": "Sphere.032", "type": "MESH", "location": [0.20000000298023224, 2.059999942779541, 2.2799999713897705], "rotation": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0], "visible": True}, {"name": "Sphere.033", "type": "MESH", "location": [-0.05000000074505806, 2.069999933242798, 2.2699999809265137], "rotation": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0], "visible": True}, {"name": "Sphere.034", "type": "MESH", "location": [0.05000000074505806, 2.059999942779541, 2.2799999713897705], "rotation": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0], "visible": True}, {"name": "Sphere.035", "type": "MESH", "location": [0.0, -0.11999999731779099, 0.47999998927116394], "rotation": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0], "visible": True}, {"name": "Sphere.036", "type": "MESH", "location": [0.0, -0.11999999731779099, 0.5899999737739563], "rotation": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0], "visible": True}, {"name": "StringCable", "type": "CURVE", "location": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0], "visible": True}, {"name": "Torus.004", "type": "MESH", "location": [-2.2799999713897705, 0.44999998807907104, 0.38999998569488525], "rotation": [0.0, -0.0, 0.0], "scale": [1.0, 1.0, 1.0], "visible": True}, {"name": "Torus.005", "type": "MESH", "location": [-2.2799999713897705, 0.44999998807907104, 1.2000000476837158], "rotation": [0.0, -0.0, 0.0], "scale": [1.0, 1.0, 1.0], "visible": True}, {"name": "Torus.006", "type": "MESH", "location": [0.9200000166893005, 2.194999933242798, 1.7899999618530273], "rotation": [0.0, 1.5707963705062866, 0.0], "scale": [1.0, 1.0, 1.0], "visible": True}, {"name": "Torus.007", "type": "MESH", "location": [1.0399999618530273, 2.194999933242798, 1.7899999618530273], "rotation": [0.0, 1.5707963705062866, 0.0], "scale": [1.0, 1.0, 1.0], "visible": True}, {"name": "Torus.008", "type": "MESH", "location": [0.12999999523162842, -0.07999999821186066, 0.5199999809265137], "rotation": [0.0, -0.0, 0.0], "scale": [1.0, 1.0, 1.0], "visible": True}, {"name": "Torus.009", "type": "MESH", "location": [0.30000001192092896, -0.05000000074505806, 0.5199999809265137], "rotation": [0.0, -0.0, 0.0], "scale": [1.0, 1.0, 1.0], "visible": True}, {"name": "Torus.015", "type": "MESH", "location": [-1.9299999475479126, 0.3499999940395355, 1.0499999523162842], "rotation": [0.20943951606750488, -0.0, 0.13617655634880066], "scale": [1.0, 1.0, 1.0], "visible": True}, {"name": "Torus.016", "type": "MESH", "location": [-1.9299999475479126, 0.3499999940395355, 1.2999999523162842], "rotation": [0.1745329201221466, -0.0, 0.12574441730976105], "scale": [1.0, 1.0, 1.0], "visible": True}, {"name": "Torus.017", "type": "MESH", "location": [-1.9299999475479126, 0.3499999940395355, 1.5499999523162842], "rotation": [0.24434609711170197, -0.0, 0.423566609621048], "scale": [1.0, 1.0, 1.0], "visible": True}, {"name": "TreeStar", "type": "MESH", "location": [-1.9299999475479126, 0.3499999940395355, 1.8600000143051147], "rotation": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0], "visible": True}], "materials": [{"name": "Mat_Art", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_Art.002", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_Bauble_0.001", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_Bauble_0.002", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_Bauble_1.001", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_Bauble_1.002", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_Bauble_2.001", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_Bauble_2.002", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_Bell", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_Bell.002", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_Bow", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_Bow.003", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_Bow.004", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_Bow.005", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_Brick", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_Brick.002", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_Bulb_0", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_Bulb_0.002", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_Bulb_1", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_Bulb_1.002", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_Bulb_2", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_Bulb_2.002", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_Bulb_3", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_Bulb_3.002", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_Cable", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_Cable.002", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_Candle", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_Candle.002", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_CandleFlame", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_CandleFlame.002", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_CandyCane.001", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_CandyCane.002", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_Carrot", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_Carrot.002", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_Ceiling", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_Ceiling.002", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_ClockFace", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_ClockFace.002", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_ClockHand", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_ClockHand.002", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_ClockRim", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_ClockRim.002", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_Cushion.001", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_Cushion.002", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_Fire", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_Fire.002", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_Firebox.001", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_Firebox.002", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_Fixture", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_Fixture.001", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_Floor", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_Floor.002", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_Frame", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_Frame.002", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_Garland", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_Garland.002", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_GiftGreen", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_GiftGreen.002", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_GiftRed", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_GiftRed.002", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_Gold.001", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_Gold.002", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_Hat", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_Hat.002", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_LampShade", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_LampShade.002", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_MiniTree", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_MiniTree.001", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_MiniTree.004", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_MiniTree.005", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_MiniTreeTop", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_MiniTreeTop.001", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_MiniTreeTop.004", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_MiniTreeTop.005", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_Mug", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_Mug.001", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_Mug.004", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_Mug.005", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_Pot", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_Pot.001", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_Pot.004", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_Pot.005", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_PotTop", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_PotTop.001", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_PotTop.004", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_PotTop.005", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_Ribbon.002", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_Ribbon.003", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_Ribbon.004", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_Ribbon.005", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_Ribbon2.002", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_Ribbon2.003", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_Ribbon2.004", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_Ribbon2.005", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_RugCream", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_RugCream.002", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_RugRed", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_RugRed.002", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_RugRed2", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_RugRed2.002", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_Snow", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_Snow.002", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_Sofa", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_Sofa.002", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_StockingRed", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_StockingRed.002", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_StockingWhite", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_StockingWhite.002", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_Table", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_Table.002", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_Tree", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_Tree.002", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_TreeBase", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_TreeBase.002", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_Wall_Red", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_Wall_Red.002", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_Wall_Wood", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}, {"name": "Mat_Wall_Wood.002", "use_nodes": True, "diffuse_color": [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0]}], "lights": [{"name": "CeilingArea", "type": "AREA", "energy": 180.0, "color": [1.0, 0.7300000190734863, 0.5], "location": [0.20000000298023224, 0.0, 2.619999885559082], "rotation": [0.0, 0.0, 0.0]}, {"name": "FillArea", "type": "AREA", "energy": 210.0, "color": [1.0, 0.7799999713897705, 0.5799999833106995], "location": [-1.100000023841858, -1.5499999523162842, 1.5499999523162842], "rotation": [1.4485357999801636, -5.419956039531826e-08, -0.6790053844451904]}, {"name": "FirePoint", "type": "POINT", "energy": 900.0, "color": [1.0, 0.5, 0.20000000298023224], "location": [1.600000023841858, 1.5499999523162842, 0.550000011920929], "rotation": [0.0, 0.0, 0.0]}, {"name": "LeftWallFill", "type": "AREA", "energy": 120.0, "color": [1.0, 0.7799999713897705, 0.5799999833106995], "location": [-0.20000000298023224, 2.0999999046325684, 1.600000023841858], "rotation": [1.1502618789672852, -1.0831571017888564e-07, -2.6779448986053467]}, {"name": "WallBounce", "type": "AREA", "energy": 110.0, "color": [1.0, 0.7799999713897705, 0.5799999833106995], "location": [-2.4000000953674316, 0.10000000149011612, 1.7999999523162842], "rotation": [1.4104933738708496, 4.532386554956247e-08, -1.3258177042007446]}], "cameras": [{"name": "Camera", "lens": 32.0, "location": [-2.559999942779541, -1.659999966621399, 1.5], "rotation": [1.4976924657821655, -3.6556357940753514e-10, -0.8738518357276917], "is_active": True}]}