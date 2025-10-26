# blender_server.py
import math
import os
import sys
from pathlib import Path
import logging
from mcp.server.fastmcp import FastMCP
import subprocess
import json
import shutil
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
            "name": "set_visibility",
            "description": "Set the visibility of the objects in the scene. You can decide to show or hide the objects. You do not need to mention all the objects here, the objects you do not metioned will keep their original visibility.",
            "parameters": {
                "type": "object",
                "properties": {
                    "show_objects": {
                        "type": "array", 
                        "description": "The names of the objects to show. Objects must exist in the scene.",
                        "items": {
                            "type": "string",
                            "description": "The name of the object to show."
                        }
                    },
                    "hide_objects": {
                        "type": "array", 
                        "description": "The names of the objects to hide. Objects must exist in the scene.",
                        "items": {
                            "type": "string",
                            "description": "The name of the object to hide."
                        }
                    }
                },
                "required": ["show_objects", "hide_objects"]
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
        self.base = os.path.dirname(script_save)
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
            # If no image output
            if not os.path.exists(f"{self.base}/tmp/camera_info.json"):
                return {"status": "success", "output": {"text": [proc.stdout]}}
            # If image output
            with open(f"{self.base}/tmp/camera_info.json", "r") as f:
                camera_info = json.load(f)
            return {"status": "success", "output": {"image": imgs, "text": [proc.stdout], "camera_parameters": camera_info}}
        except subprocess.CalledProcessError as e:
            logging.error(f"Blender failed: {e.stderr}")
            return {"status": "error", "output": {"text": [e.stderr or e.stdout]}}

    def execute(self, full_code: str) -> Dict[str, Any]:
        run_dir = self.next_run_dir()
        code_file = self.script_path / f"{self.count}.py"
        with open(code_file, "w") as f:
            f.write(full_code)
        result = self._execute_blender(code_file, run_dir)
        # check: if run_dir is empty, remove run_dir
        if not os.listdir(run_dir):
            shutil.rmtree(run_dir)
            self.count -= 1
        return result

class Investigator3D:
    def __init__(self, save_dir: str, blender_path: str, blender_command: str, blender_script: str, gpu_devices: str):
        self.blender_file = blender_path
        self.blender_command = blender_command
        self.base = Path(save_dir)
        self.base.mkdir(parents=True, exist_ok=True)
        self.tmp_dir = self.base / "tmp"
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
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
        "location": [round(x, 2) for x in obj.matrix_world.translation],
        "rotation": [round(x, 2) for x in obj.rotation_euler],
        "scale": [round(x, 2) for x in obj.scale],
        "visible": not (obj.hide_viewport or obj.hide_render)
    }})

for mat in bpy.data.materials:
    scene_info["materials"].append({{
        "name": mat.name,
        "use_nodes": mat.use_nodes,
        "diffuse_color": [round(x, 2) for x in mat.diffuse_color],
    }})

for light in [o for o in bpy.data.objects if o.type == 'LIGHT']:
    scene_info["lights"].append({{
        "name": light.name,
        "type": light.data.type,
        "energy": light.data.energy,
        "color": [round(x, 2) for x in light.data.color],
        "location": [round(x, 2) for x in light.matrix_world.translation],
        "rotation": [round(x, 2) for x in light.rotation_euler]
    }})

for cam in [o for o in bpy.data.objects if o.type == 'CAMERA']:
    scene = bpy.context.scene
    scene_info["cameras"].append({{
        "name": cam.name,
        "lens": cam.data.lens,
        "location": [round(x, 2) for x in cam.matrix_world.translation],
        "rotation": [round(x, 2) for x in cam.rotation_euler],
        "is_active": cam == scene.camera,
    }})

# Save to file for retrieval
with open("{self.base}/tmp/scene_info.json", "w") as f:
    json.dump(scene_info, f)

print("Scene info extracted successfully")
'''

    def _generate_render_script(self) -> str:
        """Generate script to render current scene once into RENDER_DIR/output.png"""
        return '''import bpy, os

render_dir = os.environ.get("RENDER_DIR", "/tmp")

# Basic render settings
bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.render.image_settings.file_format = 'PNG'
bpy.context.scene.render.resolution_x = 512
bpy.context.scene.render.resolution_y = 512

# Single render
bpy.context.scene.render.filepath = os.path.join(render_dir, "output.png")
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
bpy.context.scene.render.filepath = os.path.join(render_dir, "output.png")
bpy.ops.render.render(write_still=True)

# update camera info
camera_info = [{{
    "location": list(camera.location),
    "rotation": list(camera.rotation_euler)
}}]
rotate_info = {{
    "radius": distance,
    "theta": math.atan2(*(camera_pos[i] - target_pos[i] for i in (1,0))),
    "phi": math.asin((camera_pos.z - target_pos.z)/distance)
}}

with open(f"{self.base}/tmp/camera_info.json", "w") as f:
    json.dump(camera_info, f)
with open(f"{self.base}/tmp/rotate_info.json", "w") as f:
    json.dump(rotate_info, f)

print("Camera focused on object and rendered")
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
bpy.context.scene.render.filepath = os.path.join(render_dir, "output.png")
bpy.ops.render.render(write_still=True)

camera_info = [{{
    "location": list(camera.location),
    "rotation": list(camera.rotation_euler)
}}]

with open(f"{self.base}/tmp/camera_info.json", "w") as f:
    json.dump(camera_info, f)

print("Camera set to location and rotation and rendered")
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
bpy.context.scene.render.filepath = os.path.join(render_dir, "output.png")
bpy.ops.render.render(write_still=True)

camera_info = [{{
    "location": list(camera.location),
    "rotation": list(camera.rotation_euler)
}}]

with open(f"{self.base}/tmp/camera_info.json", "w") as f:
    json.dump(camera_info, f)

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
bpy.context.scene.render.filepath = os.path.join(render_dir, "output.png")
bpy.ops.render.render(write_still=True)

camera_info = [{{
    "location": list(camera.location),
    "rotation": list(camera.rotation_euler)
}}]

with open(f"{self.base}/tmp/camera_info.json", "w") as f:
    json.dump(camera_info, f)

print("Camera moved to position and rendered")
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
bpy.context.scene.render.filepath = os.path.join(render_dir, "output.png")
bpy.ops.render.render(write_still=True)

camera_info = [{{
    "location": list(bpy.context.scene.camera.location),
    "rotation": list(bpy.context.scene.camera.rotation_euler)
}}]

with open(f"{self.base}/tmp/camera_info.json", "w") as f:
    json.dump(camera_info, f)

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
    
    camera_infos.append([{{
        "location": list(camera.location),
        "rotation": list(camera.rotation_euler)
    }}])
    
with open(f"{self.base}/tmp/camera_info.json", "w") as f:
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
                    # Update the verifier base file
                    self.executor.blender_file = self.executor.blender_save
                
            return result
        except Exception as e:
            logging.error(f"Script execution failed: {e}")
            return {"status": "error", "output": {"text": [str(e)]}}

    def _render(self):
        """Render current scene and return image path and camera parameters"""
        render_script = self._generate_render_script()
        return self._execute_script(render_script, "Render current scene")
    
    def get_info(self) -> dict:
        """Get scene information by executing a script"""
        try:
            # Use cached info if available
            if self.scene_info_cache:
                return {"status": "success", "output": {"text": [str(self.scene_info_cache)]}}
            script = self._generate_scene_info_script()
            result = self._execute_script(script, "Extract scene information")
            if result.get("status") == "success":
                with open(f"{self.base}/tmp/scene_info.json", "r") as f:
                    scene_info = json.load(f)
                    self.scene_info_cache = scene_info
                    return {"status": "success", "output": {"text": [str(scene_info)]}}
            else:
                return {"status": "error", "output": {"text": ["Failed to extract scene information"]}}
        except Exception as e:
            return {"status": "error", "output": {"text": [str(e)]}}

    def focus_on_object(self, object_name: str) -> dict:
        """Focus camera on a specific object"""
        self.target = object_name  # Store object name instead of object reference
        # Generate and execute focus script
        focus_script = self._generate_camera_focus_script(object_name)
        result = self._execute_script(focus_script, f"Focus camera on object {object_name}")
        with open(f"{self.base}/tmp/rotate_info.json", "r") as f:
            rotate_info = json.load(f)
            self.radius = rotate_info['radius']
            self.theta = rotate_info['theta']
            self.phi = rotate_info['phi']
        return result

    def zoom(self, direction: str) -> dict:
        """Zoom camera in or out"""
        if not self.target:
            return {"status": "error", "output": {"text": ["No target object set. Call focus first."]}}
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
        return self._execute_script(move_script, f"Move camera around {self.target}")

    def set_camera(self, location: list, rotation_euler: list) -> dict:
        """Set camera position and rotation"""
        script = self._generate_camera_set_script(location, rotation_euler)
        return self._execute_script(script, f"Set camera to location {location} and rotation {rotation_euler}")

    def initialize_viewpoint(self, object_names: list) -> dict:
        """Initialize viewpoints around specified objects"""
        script = self._generate_viewpoint_script(object_names)
        return self._execute_script(script, f"Initialize viewpoints for objects: {object_names}")

    def set_keyframe(self, frame_number: int) -> dict:
        """Set scene to a specific frame"""
        script = self._generate_keyframe_script(frame_number)
        return self._execute_script(script, f"Set frame to {frame_number}")
    
    def set_visibility(self, show_objects: list, hide_objects: list) -> dict:
        """Set visibility of objects"""
        script = self._generate_visibility_script(show_objects, hide_objects)
        return self._execute_script(script, f"Set visibility: show {show_objects}, hide {hide_objects}")

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
        return {"status": "success", "output": {"image": result["image"], "text": [f"Camera parameters: {[round(x, 2) for x in result['camera_parameters']]}"]}}
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
        return {"status": "success", "output": {"image": result["image"], "text": [f"Camera parameters: {[round(x, 2) for x in result['camera_parameters']]}"]}}
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
        return {"status": "success", "output": {"image": result["image"], "text": [f"Camera parameters: {[round(x, 2) for x in result['camera_parameters']]}"]}}
    except Exception as e:
        logging.error(f"Move failed: {e}")
        return {"status": "error", "output": {"text": [str(e)]}}

@mcp.tool()
def get_scene_info() -> dict:
    try:
        global _investigator
        if _investigator is None:
            return {"status": "error", "output": {"text": ["SceneInfo not initialized. Call initialize first."]}}
        return _investigator.get_info()
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
        result['text'] = [f"[Viewpoint {i+1}] Camera parameters: {[round(x, 2) for x in result['camera_parameters'][i]]}" for i in range(len(result['image']))]
        return {"status": "success", "output": {"image": result["image"], "text": result['text']}}
    except Exception as e:
        logging.error(f"Add viewpoint failed: {e}")
        return {"status": "error", "output": {"text": [str(e)]}}

@mcp.tool()
def investigate(operation: str, object_name: str = None, direction: str = None) -> dict:
    if operation == "focus":
        if not object_name:
            return {"status": "error", "output": {"text": ["object_name is required for focus"]}}
        result = focus(object_name=object_name)
        return {"status": "success", "output": {"image": result["image"], "text": [f"Camera parameters: {[round(x, 2) for x in result['camera_parameters']]}"]}}
    elif operation == "zoom":
        if direction not in ("in", "out"):
            return {"status": "error", "output": {"text": ["direction must be 'in' or 'out' for zoom"]}}
        result = zoom(direction=direction)
        return {"status": "success", "output": {"image": result["image"], "text": [f"Camera parameters: {[round(x, 2) for x in result['camera_parameters']]}"]}}
    elif operation == "move":
        if direction not in ("up", "down", "left", "right"):
            return {"status": "error", "output": {"text": ["direction must be one of up/down/left/right for move"]}}
        result = move(direction=direction)
        return {"status": "success", "output": {"image": result["image"], "text": [f"Camera parameters: {[round(x, 2) for x in result['camera_parameters']]}"]}}
    else:
        return {"status": "error", "output": {"text": [f"Unknown operation: {operation}"]}}

@mcp.tool()
def set_visibility(show_objects: list = None, hide_objects: list = None) -> dict:
    global _investigator
    if _investigator is None:
        return {"status": "error", "output": {"text": ["Investigator3D not initialized. Call initialize_investigator first."]}}
    try:
        result = _investigator.set_visibility(show_objects, hide_objects)
        return {"status": "success", "output": {"image": result["image"], "text": [f"Camera parameters: {[round(x, 2) for x in result['camera_parameters']]}"]}}
    except Exception as e:
        logging.error(f"set_visibility failed: {e}")
        return {"status": "error", "output": {"text": [str(e)]}}

@mcp.tool()
def set_keyframe(frame_number: int) -> dict:
    global _investigator
    if _investigator is None:
        return {"status": "error", "output": {"text": ["Investigator3D not initialized. Call initialize_investigator first."]}}
    try:
        result = _investigator.set_keyframe(frame_number)
        return {"status": "success", "output": {"image": result["image"], "text": [f"Camera parameters: {[round(x, 2) for x in result['camera_parameters']]}"]}}
    except Exception as e:
        logging.error(f"Set keyframe failed: {e}")
        return {"status": "error", "output": {"text": [str(e)]}, "status": "error"}
    
@mcp.tool()
def set_camera(location: list, rotation_euler: list) -> dict:
    global _investigator
    if _investigator is None:
        return {"status": "error", "output": {"text": ["Investigator3D not initialized. Call initialize_investigator first."]}}
    try:
        result = _investigator.set_camera(location, rotation_euler)
        return {"status": "success", "output": {"image": result["image"], "text": [f"Camera parameters: {[round(x, 2) for x in result['camera_parameters']]}"]}}
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
    test_save_dir = os.getenv("THOUGHT_SAVE", "output/test/")
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
    
    object_names = ['CornerCap']
    print(f"Object names: {object_names}")
        
    # Test 4: Initialize viewpoint
    print("\n4. Testing initialize_viewpoint...")
    viewpoint_result = initialize_viewpoint(object_names=object_names)
    print(f"Result: {viewpoint_result}")

    # Test 5: Focus, zoom, move, set_keyframe if objects exist
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
    move_result = move(direction="left")
    print(f"Result: {move_result}")
    
    # Test set_keyframe
    print("\n5.4. Testing set_keyframe...")
    keyframe_result = set_keyframe(frame_number=1)
    print(f"Result: {keyframe_result}")
    
    # Test set_visibility
    print("\n5.5. Testing set_visibility...")
    visibility_result = set_visibility(hide_objects=[first_object])
    print(f"Result: {visibility_result}")
    
    # Test 3: Reload scene
    print("\n3. Testing reload_scene...")
    reload_result = reload_scene()
    print(f"Result: {reload_result}")

    # Test focus
    print("\n5.1. Testing focus...")
    focus_result = focus(object_name=first_object)
    print(f"Result: {focus_result}")

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