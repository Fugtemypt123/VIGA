"""Blender Script Generators.

Generates Python scripts for Blender operations used by investigator and
exec_blender tools. Contains reusable script generation methods for scene
inspection, rendering, and camera manipulation.
"""

from typing import List


def generate_scene_info_script(output_path: str) -> str:
    """Generate script to extract scene information with bounding boxes.

    Args:
        output_path: Path where the JSON scene info will be saved.

    Returns:
        Blender Python script as a string.
    """
    return f'''import bpy
import json
import sys
from mathutils import Vector

# Get scene information
scene_info = {{"objects": [], "materials": [], "lights": [], "cameras": []}}

for obj in bpy.data.objects:
    if obj.type == 'CAMERA' or obj.type == 'LIGHT':
        continue
    
    # Calculate bounding box in world coordinates
    bbox = None
    if hasattr(obj, 'bound_box') and obj.bound_box:
        bbox_corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
        min_x = min(corner.x for corner in bbox_corners)
        min_y = min(corner.y for corner in bbox_corners)
        min_z = min(corner.z for corner in bbox_corners)
        max_x = max(corner.x for corner in bbox_corners)
        max_y = max(corner.y for corner in bbox_corners)
        max_z = max(corner.z for corner in bbox_corners)
        bbox = {{
            "min": [round(min_x, 2), round(min_y, 2), round(min_z, 2)],
            "max": [round(max_x, 2), round(max_y, 2), round(max_z, 2)],
            "center": [round((min_x + max_x) / 2, 2), round((min_y + max_y) / 2, 2), round((min_z + max_z) / 2, 2)],
            "size": [round(max_x - min_x, 2), round(max_y - min_y, 2), round(max_z - min_z, 2)]
        }}
    
    scene_info["objects"].append({{
        "name": obj.name, 
        "type": obj.type,
        "location": [round(x, 2) for x in obj.matrix_world.translation],
        "rotation": [round(x, 2) for x in obj.rotation_euler],
        "scale": [round(x, 2) for x in obj.scale],
        "visible": not (obj.hide_viewport or obj.hide_render),
        "bbox": bbox
    }})
    if len(scene_info["objects"]) >= 25:
        break

for mat in bpy.data.materials:
    scene_info["materials"].append({{
        "name": mat.name,
        "use_nodes": mat.use_nodes,
        "diffuse_color": [round(x, 2) for x in mat.diffuse_color],
    }})
    if len(scene_info["materials"]) >= 10:
        break
        
for light in [o for o in bpy.data.objects if o.type == 'LIGHT']:
    scene_info["lights"].append({{
        "name": light.name,
        "type": light.data.type,
        "energy": light.data.energy,
        "color": [round(x, 2) for x in light.data.color],
        "location": [round(x, 2) for x in light.matrix_world.translation],
        "rotation": [round(x, 2) for x in light.rotation_euler]
    }})
    if len(scene_info["lights"]) >= 5:
        break
        
for cam in [o for o in bpy.data.objects if o.type == 'CAMERA']:
    scene = bpy.context.scene
    scene_info["cameras"].append({{
        "name": cam.name,
        "lens": cam.data.lens,
        "location": [round(x, 2) for x in cam.matrix_world.translation],
        "rotation": [round(x, 2) for x in cam.rotation_euler],
        "is_active": cam == scene.camera,
    }})
    if len(scene_info["cameras"]) >= 3:
        break
        
# Save to file for retrieval
with open("{output_path}", "w") as f:
    json.dump(scene_info, f)

print("Scene info extracted successfully")
'''


def generate_render_script() -> str:
    """Generate script to render the current scene.

    Renders the scene to RENDER_DIR/output.png using Cycles engine
    at 512x512 resolution.

    Returns:
        Blender Python script as a string.
    """
    return '''import bpy
import os

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


def generate_camera_focus_script(object_name: str, base_path: str) -> str:
    """Generate script to focus camera on a specific object.

    Creates a track-to constraint to point the camera at the target object,
    renders the scene, and saves camera info to JSON.

    Args:
        object_name: Name of the Blender object to focus on.
        base_path: Base path for saving camera info JSON files.

    Returns:
        Blender Python script as a string.
    """
    return f'''import bpy
import json
import math
import os

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

with open(f"{base_path}/tmp/camera_info.json", "w") as f:
    json.dump(camera_info, f)
with open(f"{base_path}/tmp/rotate_info.json", "w") as f:
    json.dump(rotate_info, f)

print("Camera focused on object and rendered")
'''


def generate_camera_set_script(location: List[float], rotation_euler: List[float], base_path: str) -> str:
    """Generate script to set camera position and rotation.

    Sets the camera to a specific location and rotation, renders the scene,
    and saves camera info to JSON.

    Args:
        location: Camera location as [x, y, z] coordinates.
        rotation_euler: Camera rotation as [rx, ry, rz] Euler angles.
        base_path: Base path for saving camera info JSON files.

    Returns:
        Blender Python script as a string.
    """
    return f'''import bpy
import json
import os

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

with open(f"{base_path}/tmp/camera_info.json", "w") as f:
    json.dump(camera_info, f)

print("Camera set to location and rotation and rendered")
'''


def generate_visibility_script(show_objects: List[str], hide_objects: List[str], base_path: str) -> str:
    """Generate script to set object visibility and render.

    Sets visibility for specified objects, renders the scene, and saves
    camera info to JSON.

    Args:
        show_objects: List of object names to make visible.
        hide_objects: List of object names to hide.
        base_path: Base path for saving camera info JSON files.

    Returns:
        Blender Python script as a string.
    """
    return f'''import bpy
import json
import os

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
    "location": list(bpy.context.scene.camera.location),
    "rotation": list(bpy.context.scene.camera.rotation_euler)
}}]

with open(f"{base_path}/tmp/camera_info.json", "w") as f:
    json.dump(camera_info, f)

print("Visibility updated and rendered: show", show_list, ", hide", hide_list)
'''


def generate_camera_move_script(target_obj_name: str, radius: float, theta: float, phi: float, base_path: str) -> str:
    """Generate script to move camera around a target object.

    Positions the camera in spherical coordinates relative to the target
    object, renders the scene, and saves camera info to JSON.

    Args:
        target_obj_name: Name of the object to orbit around.
        radius: Distance from the target object.
        theta: Azimuth angle in radians.
        phi: Elevation angle in radians.
        base_path: Base path for saving camera info JSON files.

    Returns:
        Blender Python script as a string.
    """
    return f'''import bpy
import json
import math
import os

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

with open(f"{base_path}/tmp/camera_info.json", "w") as f:
    json.dump(camera_info, f)

print("Camera moved to position and rendered")
'''


def generate_keyframe_script(frame_number: int, base_path: str) -> str:
    """Generate script to set the current frame and render.

    Sets the timeline to a specific frame number, renders the scene,
    and saves camera info to JSON.

    Args:
        frame_number: Target frame number to set.
        base_path: Base path for saving camera info JSON files.

    Returns:
        Blender Python script as a string.
    """
    return f'''import bpy
import json
import os

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

with open(f"{base_path}/tmp/camera_info.json", "w") as f:
    json.dump(camera_info, f)

print("Changed to frame", target_frame, "(was", current_frame, ") and rendered")
'''


def generate_viewpoint_script(object_names: List[str], base_path: str) -> str:
    """Generate script to initialize viewpoints around objects.

    Creates four viewpoints around the bounding box of specified objects,
    renders from each viewpoint, and saves camera info to JSON.

    Args:
        object_names: List of object names to observe. If empty, observes
            all mesh objects except Ground and Plane.
        base_path: Base path for saving camera info JSON files.

    Returns:
        Blender Python script as a string.
    """
    return f'''import bpy
import json
import math
import os
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
    
    camera_infos.append({{
        "location": list(camera.location),
        "rotation": list(camera.rotation_euler)
    }})
    
with open(f"{base_path}/tmp/camera_info.json", "w") as f:
    json.dump(camera_infos, f)
    
# Restore original position
camera.location = original_location
camera.rotation_euler = original_rotation

print("Viewpoints initialized and rendered for", len(objects), "objects")
'''

