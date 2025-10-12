# blender_server.py
import bpy
import mathutils
import math
import os
import sys
from pathlib import Path
import logging
from mcp.server.fastmcp import FastMCP
import json
import copy

# tool config for agent (only the function w/ @mcp.tools)
tool_configs = [
    {
        "type": "function",
        "function": {
            "name": "init_viewpoint",
            "description": "Adds a viewpoint to observe the listed objects. The viewpoints are added to the four corners of the bounding box of the listed objects. This tool returns the positions and rotations of the four viewpoint cameras, as well as the rendered images of the four cameras. You would better call this tool first before you can call the other tools.",
            "parameters": {
                "type": "object",
                "properties": {
                    "object_names": {"type": "array", "description": "The names of the objects to observe. Objects must exist in the scene (you can check the scene information to see if they exist). If you want to observe the whole scene, you can pass an empty list."}
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
                    "location": {"type": "array", "description": "The location of the camera (in world coordinates)"},
                    "rotation_euler": {"type": "array", "description": "The rotation of the camera (in euler angles)"}
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
                    "show_object_list": {"type": "array", "description": "The names of the objects to show. Objects must exist in the scene."},
                    "hide_object_list": {"type": "array", "description": "The names of the objects to hide. Objects must exist in the scene."}
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
# Built-in tools
# ======================

class GetSceneInfo:
    def __init__(self, blender_path: str):
        bpy.ops.wm.open_mainfile(filepath=str(blender_path))

    def get_info(self) -> dict:
        try:
            scene_info = {"objects": [], "materials": [], "lights": [], "cameras": []}
            for obj in bpy.data.objects:
                if obj.type == 'CAMERA' or obj.type == 'LIGHT':
                    continue
                scene_info["objects"].append({
                    "name": obj.name, 
                    "type": obj.type,
                    "location": list(obj.matrix_world.translation),
                    "rotation": list(obj.rotation_euler),
                    "scale": list(obj.scale),
                    "visible": not (obj.hide_viewport or obj.hide_render)
                })

            for mat in bpy.data.materials:
                scene_info["materials"].append({
                    "name": mat.name,
                    "use_nodes": mat.use_nodes,
                    "diffuse_color": list(mat.diffuse_color),
                })

            for light in [o for o in bpy.data.objects if o.type == 'LIGHT']:
                scene_info["lights"].append({
                    "name": light.name,
                    "type": light.data.type,
                    "energy": light.data.energy,
                    "color": list(light.data.color),
                    "location": list(light.matrix_world.translation),
                    "rotation": list(light.rotation_euler)
                })

            for cam in [o for o in bpy.data.objects if o.type == 'CAMERA']:
                scene = bpy.context.scene
                scene_info["cameras"].append({
                    "name": cam.name,
                    "lens": cam.data.lens,
                    "location": list(cam.matrix_world.translation),
                    "rotation": list(cam.rotation_euler),
                    "is_active": cam == scene.camera,
                })

            return scene_info
        except Exception as e:
            logging.error(f"scene info error: {e}")
            return {}

# ======================
# Camera investigator (Fixed: save path first then load)
# ======================

class Investigator3D:
    def __init__(self, save_dir: str, blender_path: str):
        self.blender_path = blender_path          # Save path first
        self._load_blender_file()                 # Then load file
        self.base = Path(save_dir)
        self.base.mkdir(parents=True, exist_ok=True)
        # self.cam = self._get_or_create_cam()
        self.target = None
        self.radius = 5.0
        self.theta = 0.0
        self.phi = 0.0
        self.count = 0

    def _load_blender_file(self):
        # current_file = bpy.data.filepath
        # if current_file != self.blender_path:
        bpy.ops.wm.open_mainfile(filepath=str(self.blender_path))
        self.cam = self._get_or_create_cam()

    def _get_or_create_cam(self):
        # Use existing camera if available, otherwise create with fixed starting positions
        # Add a new camera in the scene
        bpy.ops.object.camera_add()
        cam = bpy.context.active_object
        cam.name = "InvestigatorCamera"
        cam.location = (0, 0, 5)
        cam.rotation_euler = (math.radians(60), 0, 0)
        for existing_cam in bpy.data.objects:
            if existing_cam.type == 'CAMERA':
                cam.location = existing_cam.location
                cam.rotation_euler = existing_cam.rotation_euler
                break
        return cam

    def _render(self):
        bpy.context.scene.camera = self.cam
        bpy.context.scene.render.engine = 'CYCLES'
        bpy.context.scene.render.filepath = str(self.base / f"{self.count+1}.png")
        bpy.ops.render.render(write_still=True)
        out = bpy.context.scene.render.filepath
        self.count += 1
        camera_parameters = str({"position": list(self.cam.matrix_world.translation), "rotation": list(self.cam.rotation_euler)})
        return {"image_path": out, "camera_parameters": camera_parameters}

    def focus_on_object(self, object_name: str) -> dict:
        obj = bpy.data.objects.get(object_name)
        if not obj:
            raise ValueError(f"{object_name} not found")
        self.target = obj
        # track-to
        constraint = None
        for c in self.cam.constraints:
            if c.type == 'TRACK_TO':
                constraint = c
                break
        if not constraint:
            constraint = self.cam.constraints.new('TRACK_TO')
        constraint.target = obj
        constraint.track_axis = 'TRACK_NEGATIVE_Z'
        constraint.up_axis = 'UP_Y'
        self.radius = (self.cam.matrix_world.translation - obj.matrix_world.translation).length
        self.theta = math.atan2(*(self.cam.matrix_world.translation[i] - obj.matrix_world.translation[i] for i in (1,0)))
        self.phi = math.asin((self.cam.matrix_world.translation.z - obj.matrix_world.translation.z)/self.radius)
        return self._render()

    def zoom(self, direction: str) -> dict:
        if direction == 'in':
            self.radius = max(1, self.radius-3)
        elif direction == 'out':
            self.radius += 3
        return self._update_and_render()

    def move_camera(self, direction: str) -> dict:
        step = self.radius
        theta_step = step / (self.radius*math.cos(self.phi))
        phi_step = step / self.radius
        if direction=='up': self.phi = min(math.pi/2-0.1, self.phi+phi_step)
        elif direction=='down': self.phi = max(-math.pi/2+0.1, self.phi-phi_step)
        elif direction=='left': self.theta -= theta_step
        elif direction=='right': self.theta += theta_step
        return self._update_and_render()

    def _update_and_render(self) -> dict:
        t = self.target.matrix_world.translation
        x = self.radius * math.cos(self.phi) * math.cos(self.theta)
        y = self.radius * math.cos(self.phi) * math.sin(self.theta)
        z = self.radius * math.sin(self.phi)
        self.cam.matrix_world.translation = (t.x+x, t.y+y, t.z+z)
        return self._render()

    def set_camera(self, location: list, rotation_euler: list) -> dict:
        self.cam.location = location
        self.cam.rotation_euler = rotation_euler

    def init_viewpoint(self, object_names: list) -> dict:
        try:
            objects = []
            
            # If empty list is passed, observe all mesh objects in the entire scene
            if not object_names:
                for obj in bpy.data.objects:
                    if obj.type == 'MESH' and obj.name not in ['Ground', 'Plane']:  # Exclude ground and other auxiliary objects
                        objects.append(obj)
                logging.info(f"Observing whole scene: found {len(objects)} mesh objects")
            else:
                # Find specified objects by name
                for obj_name in object_names:
                    obj = bpy.data.objects.get(obj_name)
                    if obj:
                        objects.append(obj)
                    else:
                        logging.warning(f"Object '{obj_name}' not found in scene")
            
            if not objects:
                raise ValueError("No valid objects found in the scene or provided list")

            min_x = min_y = min_z = float('inf')
            max_x = max_y = max_z = float('-inf')
            
            for obj in objects:
                bbox_corners = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
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
            center = mathutils.Vector((center_x, center_y, center_z))

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

            viewpoints = {'image': [], 'text': []}
            previous_cam_info = {'location': self.cam.location, 'rotation': self.cam.rotation_euler}
            
            for i, pos in enumerate(camera_positions):
                self.cam.location = pos
                self.cam.rotation_euler = (math.radians(60), 0, math.radians(45))
                
                direction = center - self.cam.location
                self.cam.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
                render_result = self._render()
                
                camera_parameters = str({"position": list(self.cam.location), "rotation": list(self.cam.rotation_euler)})
                
                viewpoints['image'].append(render_result['image_path'])
                viewpoints['text'].append(f"[Viewpoint {i+1}] Camera parameters: {camera_parameters}")
                
                logging.info(f"Viewpoint {i+1}: position={self.cam.location}, rotation={self.cam.rotation_euler}")
                
            # Restore original camera position
            self.cam.location = previous_cam_info['location']
            self.cam.rotation_euler = previous_cam_info['rotation']
            
            return {'status': 'success', 'output': viewpoints}
                
        except Exception as e:
            return {'status': 'error', 'output': {'text': [str(e)]}}

    def set_keyframe(self, frame_number: int) -> dict:
        try:
            scene = bpy.context.scene
            current_frame = scene.frame_current
            
            # Ensure frame number is within valid range
            target_frame = max(scene.frame_start, min(scene.frame_end, frame_number))
            scene.frame_set(target_frame)
            logging.info(f"Changed to frame {target_frame} (was {current_frame})")
            render_result = self._render()
            
            return {
                'status': 'success',
                'output': {'image': [render_result['image_path']], 'text': [f"Camera parameters: {render_result['camera_parameters']}"]}
            }
            
        except Exception as e:
            return {'status': 'error', 'output': {'text': [str(e)]}}

@mcp.tool()
def initialize(args: dict) -> dict:
    """
    Initialize 3D scene investigation tool.
    """
    global _investigator
    try:
        save_dir = args.get("thought_save") + "/investigator/" + str(args.get("round"))
        _investigator = Investigator3D(save_dir, str(args.get("blender_file")))
        return {"status": "success", "output": {"text": ["Investigator3D initialized successfully"], "tool_configs": tool_configs}}
    except Exception as e:
        return {"status": "error", "output": {"text": [str(e)]}}

def focus(object_name: str) -> dict:
    global _investigator
    if _investigator is None:
        return {"status": "error", "output": {"text": ["Investigator3D not initialized. Call initialize_investigator first."]}}

    try:
        # Check if target object exists
        obj = bpy.data.objects.get(object_name)
        if not obj:
            return {"status": "error", "output": {"text": [f"Object '{object_name}' not found in scene"]}}

        result = _investigator.focus_on_object(object_name)
        return {
            "status": "success", 
            "output": {"image": [result["image_path"]], "text": [f"Camera parameters: {result["camera_parameters"]}"]}
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
            "output": {'image': [result["image_path"]], 'text': [f"Camera parameters: {result["camera_parameters"]}"]}
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
            "output": {'image': [result["image_path"]], 'text': [f"Camera parameters: {result["camera_parameters"]}"]}
        }
    except Exception as e:
        logging.error(f"Move failed: {e}")
        return {"status": "error", "output": {"text": [str(e)]}}

@mcp.tool()
def get_scene_info() -> dict:
    try:
        global _investigator
        if _investigator is None:
            return {"status": "error", "output": {"text": ["Investigator3D not initialized. Call initialize_investigator first."]}}
        info = GetSceneInfo(_investigator.blender_path).get_info()
        return {"status": "success", "output": {"text": [str(info)]}}
    except Exception as e:
        logging.error(f"Failed to get scene info: {e}")
        return {"status": "error", "output": {"text": [str(e)]}}

@mcp.tool()
def init_viewpoint(object_names: list) -> dict:
    global _investigator
    if _investigator is None:
        return {"status": "error", "output": {"text": ["Investigator3D not initialized. Call initialize_investigator first."]}}
    try:
        result = _investigator.init_viewpoint(object_names)
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
        # Apply hide/show
        for obj in bpy.data.objects:
            if obj.name in hide_object_list:
                obj.hide_viewport = True
                obj.hide_render = True
            if obj.name in show_object_list:
                obj.hide_viewport = False
                obj.hide_render = False
        result = _investigator._render()
        return {"status": "success", "output": {'image': [result["image_path"]], 'text': [f"Camera parameters: {result["camera_parameters"]}"]}}
    except Exception as e:
        logging.error(f"set_object_visibility failed: {e}")
        return {"status": "error", "output": {"text": [str(e)]}}

@mcp.tool()
def set_keyframe(frame_number: int) -> dict:
    global _investigator
    if _investigator is None:
        return {"status": "error", "output": {"text": ["Investigator3D not initialized. Call initialize_investigator first."]}}
    try:
        result = _investigator.set_keyframe(frame_number)
        if result.get("status") == "success":
            return {"status": "success", "output": {"text": ["Successfully set the keyframe"], "image": [result.get("output", {}).get("image", [])], "text": [f"Camera parameters: {result.get("output", {}).get("camera_parameters", {})}"]}}
        else:
            return {"status": "error", "output": {"text": [result.get("output", {}).get("text", ["Failed to set the keyframe"])]}}
    except Exception as e:
        logging.error(f"Set keyframe failed: {e}")
        return {"status": "error", "output": {"text": [str(e)]}}
    
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
        mcp.run(transport="stdio")

def test_tools():
    """Test all investigator tool functions (read environment variable configuration)"""
    print("=" * 50)
    print("Testing Scene Tools")
    print("=" * 50)

    # Set test paths (read from environment variables)
    blender_file = os.getenv("BLENDER_FILE", "output/test/exec_blender/test.blend")
    test_save_dir = os.getenv("THOUGHT_SAVE", "output/test/investigator/")

    # Check if blender file exists
    if not os.path.exists(blender_file):
        print(f"⚠ Blender file not found: {blender_file}")
        print("Skipping all tests.")
        return

    print(f"✓ Using blender file: {blender_file}")

    # Test 1: Get scene information
    print("\n1. Testing get_scene_info...")
    try:
        result = get_scene_info(blender_file)
        
        with open('logs/investigator.log', 'w') as f:
            f.write(json.dumps(result, ensure_ascii=False, indent=2))
        
        if result.get("status") == "success":
            print("✓ get_scene_info passed")
            info = result.get("output", {})

            # Get first object name for subsequent tests
            objects = info.get("objects", [])
            if not objects:
                print("⚠ No objects found in scene, skipping camera tests")
                # Continue Meshy tests
                first_object = None
            else:
                first_object = objects[0]['name']
                print(f"  - Will focus on: {first_object}")
        else:
            print("✗ get_scene_info failed")
            first_object = None
    except Exception as e:
        print(f"✗ get_scene_info failed with exception: {e}")
        first_object = None

    # Test 2: Initialize investigation tool
    print("\n2. Testing initialize_investigator...")
    try:
        args = {"thought_save": test_save_dir, "blender_file": blender_file, "round": 0}
        result = initialize(args)
        if result.get("status") == "success":
            print("✓ initialize_investigator passed")
        else:
            print("✗ initialize_investigator failed: ", result.get("output"))
    except Exception as e:
        print(f"✗ initialize_investigator failed with exception: {e}")
        
    # Attempt rendering
    global _investigator
    result = _investigator._render()
    print(f"Result: {result}")
        
    # Test 6: Add viewpoint functionality (specified objects)
    print("\n6. Testing init_viewpoint with specific objects...")
    try:
        test_objects = [obj['name'] for obj in objects]
        print(f"Test objects: {test_objects}")
        result = init_viewpoint(object_names=test_objects)
        print(f"Result: {result}")
        if result.get("status") == "success":
            print("✓ init_viewpoint passed")
            viewpoints = result.get('output', {}).get('viewpoints', [])
            print(f"  - Generated {len(viewpoints)} viewpoints:")
            for vp in viewpoints:
                print(f"    Viewpoint {vp.get('view_index', 'N/A')}: {vp.get('image', 'N/A')}")
                print(f"      Position: {vp.get('position', 'N/A')}")
                print(f"      Rotation: {vp.get('rotation', 'N/A')}")
        else:
            print("✗ init_viewpoint failed")
    except Exception as e:
        print(f"✗ init_viewpoint failed with exception: {e}")
    
    # Test 7: Add viewpoint functionality (entire scene)
    print("\n7. Testing init_viewpoint with whole scene...")
    try:
        result = init_viewpoint(object_names=[])  # Empty list means observe entire scene
        print(f"Result: {result}")
        if result.get("status") == "success":
            print("✓ init_viewpoint (whole scene) passed")
            viewpoints = result.get('output', {}).get('viewpoints', [])
            print(f"  - Generated {len(viewpoints)} viewpoints for whole scene:")
            for vp in viewpoints:
                print(f"    Viewpoint {vp.get('view_index', 'N/A')}: {vp.get('image', 'N/A')}")
                print(f"      Position: {vp.get('position', 'N/A')}")
                print(f"      Rotation: {vp.get('rotation', 'N/A')}")
        else:
            print("✗ init_viewpoint (whole scene) failed")
    except Exception as e:
        print(f"✗ init_viewpoint (whole scene) failed with exception: {e}")

    if first_object:        
        # Test 3: Focus on object (if objects exist)
        print("\n3. Testing focus...")
        try:
            result = focus(object_name=first_object)
            print(f"Result: {result}")
            if result.get("status") == "success":
                print("✓ focus passed")
                print(f"  - Focused on: {first_object}")
                print(f"  - Image saved: {result.get('output', 'N/A')}")
            else:
                print("✗ focus failed")
        except Exception as e:
            print(f"✗ focus failed with exception: {e}")

        # Test 4: Zoom functionality
        print("\n4. Testing zoom...")
        try:
            result = zoom(direction="in")
            print(f"Result: {result}")
            if result.get("status") == "success":
                print("✓ zoom passed")
                print(f"  - Image saved: {result.get('output', 'N/A')}")
            else:
                print("✗ zoom failed")
        except Exception as e:
            print(f"✗ zoom failed with exception: {e}")

        # Test 5: Move functionality
        print("\n5. Testing move...")
        try:
            result = move(direction="up")
            print(f"Result: {result}")
            if result.get("status") == "success":
                print("✓ move passed")
                print(f"  - Image saved: {result.get('output', 'N/A')}")
            else:
                print("✗ move failed")
        except Exception as e:
            print(f"✗ move failed with exception: {e}")

        # Test 8: Set keyframe functionality
        print("\n8. Testing set_keyframe...")
        try:
            result = set_keyframe(frame_number=1)
            print(f"Result: {result}")
            if result.get("status") == "success":
                print("✓ set_keyframe passed")
                print(f"  - Image saved: {result.get('output', 'N/A')}")
                print(f"  - Frame changed from {result.get('output', {}).get('frame_info', {}).get('previous_frame', 'N/A')} to {result.get('output', {}).get('frame_info', {}).get('current_frame', 'N/A')}")
            else:
                print("✗ set_keyframe failed")
        except Exception as e:
            print(f"✗ set_keyframe failed with exception: {e}")

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
    
    


# ======================
# MCP Tools
# ======================


# @mcp.tool()
# def set_key_frame(target_frame: int, round: int = 0) -> dict:
#     """
#     Jump to a specific keyframe (absolute frame index) and render a view.
#     """
#     global _investigator
#     if _investigator is None:
#         return {"status": "error", "error": "Investigator3D not initialized. Call initialize_investigator first."}
#     try:
#         bpy.context.scene.frame_set(int(target_frame))
#         result = _investigator._render(round)
#         return {
#             "status": "success",
#             "image": result["image_path"],
#             "camera_position": result["camera_position"],
#             "keyframe_info": {"current_frame": int(target_frame)}
#         }
#     except Exception as e:
#         logging.error(f"set_key_frame failed: {e}")
#         return {"status": "error", "error": str(e)}



# def set_camera_starting_position(direction: str = "z", round: int = 0) -> dict:
#     """
#     Set the camera to a fixed starting position for 3D scene investigation.
    
#     Args:
#         direction: Starting camera direction - "z" (from above), "x" (from side), "y" (from front), or "bbox" (above bounding box)
#         round: Current round number for file organization
        
#     Returns:
#         dict: Status and camera position information
        
#     Example:
#         set_camera_starting_position(direction="z", round=1)
#         # Sets camera to look down from above the scene
        
#     Detailed Description:
#         This tool sets the camera to predefined starting positions to ensure consistent 
#         scene investigation. The available directions are:
#         - "z": Camera positioned at (0,0,5) looking down at 60 degrees
#         - "x": Camera positioned at (-5,0,2) looking from the side
#         - "y": Camera positioned at (0,-5,2) looking from the front  
#         - "bbox": Camera positioned at (0,0,10) looking down from above bounding box
#     """
#     global _investigator
#     if _investigator is None:
#         return {"status": "error", "output": "Investigator3D not initialized. Call initialize_investigator first."}
    
#     try:
#         _investigator._set_camera_to_position(_investigator.cam, direction)
#         result = _investigator._render(round)
#         return {
#             "status": "success",
#             "output": {'image': result["image_path"], 'camera_position': result["camera_position"]}
#         }
#     except Exception as e:
#         return {"status": "error", "output": str(e)}

# @mcp.tool()
# def setup_camera(view: str = "top", round: int = 0) -> dict:
#     """
#     Setup an observer camera to a canonical view.
    
#     Args:
#         view: One of ["top", "front", "side", "oblique"].
#         round: Current round number for file organization.
#     Returns:
#         dict: status, image path, camera position
#     """
#     mapping = {
#         "top": "z",
#         "front": "y",
#         "side": "x",
#         "oblique": "bbox",
#     }
#     direction = mapping.get(view, "z")
#     return set_camera_starting_position(direction=direction, round=round)
