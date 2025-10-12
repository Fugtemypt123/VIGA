# blender_executor_server.py
import os
import subprocess
import base64
import io
from typing import Optional
from pathlib import Path
from PIL import Image
import logging
from typing import Tuple, Dict
from mcp.server.fastmcp import FastMCP
import json

# tool config for agent
tool_configs = [
    {
        "type": "function",
        "function": {
            "name": "execute_and_evaluate",
            "description": "Execute code modifications and trigger verifier evaluation. This tool combines code execution with automatic verification. Always use this tool when you want to execute your code changes.\nReturns either:\n  (1) On error: detailed error information; or \n  (2) On success: a clear render (you must add a camera in your code) and further modification suggestions from a separate verifier agent.",
            "parameters": {
                "type": "object",
                "properties": {
                    "thought": {
                        "type": "string",
                        "description": "Analyze the current state and provide a clear plan for the required changes. Consider scene representation consistency and infinigen optimization opportunities."
                    },
                    "code_edition": {
                        "type": "string", 
                        "description": "Provide your code modifications in the following format:\n-: [lines to remove]\n+: [lines to add]\nFocus on scene consistency and use infinigen functions when appropriate."
                    },
                    "full_code": {
                        "type": "string",
                        "description": "Merge your code changes into the full code with proper formatting. Ensure consistent scene representation."
                    }
                },
                "required": ["thought", "code_edition", "full_code"]
            }
        }
    }
]

mcp = FastMCP("blender-executor")

# Global executor instance
_executor = None

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
        self.blend_path = blender_save
        self.gpu_devices = gpu_devices  # e.g.: "0,1" or "0"

        self.script_path.mkdir(parents=True, exist_ok=True)
        self.render_path.mkdir(parents=True, exist_ok=True)

    def _execute_blender(self, script_path: str, render_path: str) -> Tuple[bool, str, str]:
        cmd = [
            self.blender_command,
            "--background", self.blender_file,
            "--python", self.blender_script,
            "--", script_path, render_path
        ]
        if self.blend_path:
            cmd.append(self.blend_path)
        cmd_str = " ".join(cmd)
        
        # Set environment variables to control GPU devices
        env = os.environ.copy()
        if self.gpu_devices:
            env['CUDA_VISIBLE_DEVICES'] = self.gpu_devices
            logging.info(f"Setting CUDA_VISIBLE_DEVICES to: {self.gpu_devices}")
        
        try:
            proc = subprocess.run(cmd_str, shell=True, check=True, capture_output=True, text=True, env=env)
            out = proc.stdout
            err = proc.stderr
            if os.path.isdir(render_path):
                imgs = sorted([str(p) for p in Path(render_path).glob("*") if p.suffix in ['.png','.jpg']])
                if not imgs:
                    return False, "No images", out
                return True, imgs, out
            return True, out, err
        except subprocess.CalledProcessError as e:
            logging.error(f"Blender failed: {e}")
            return False, e.stderr, e.stdout

    def _encode_image(self, img_path: str) -> str:
        img = Image.open(img_path)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    def _parse_code(self, full_code: str) -> str:
        if full_code.startswith("```python") and full_code.endswith("```"):
            return full_code[len("```python"):-len("```")]
        return full_code

    def execute(self, thought: str, code_edition: str, full_code: str, round: int) -> Dict:
        code_file = self.script_path / f"{round}.py"
        render_file = self.render_path / f"{round}"
        code = self._parse_code(full_code)
        
        # File operations
        with open(code_file, "w") as f:
            f.write(code)
        os.makedirs(render_file, exist_ok=True)
        for img in os.listdir(render_file):
            os.remove(os.path.join(render_file, img))
            
        # Execute Blender
        success, stdout, stderr = self._execute_blender(code, str(render_file))
        if not success or not os.path.exists(render_file):
            return {"status": "error", "output": {"text": [stderr or stdout]}}
        return {"status": "success", "output": {"image": [stdout]}}



@mcp.tool()
def initialize(args: dict) -> dict:
    """
    Initialize Blender executor and set all necessary parameters.
    
    Args:
        args: Dictionary containing the following keys:
            - blender_command: Blender executable file path
            - blender_file: Blender file path
            - blender_script: Blender script path
            - script_save: Script save directory
            - render_save: Render result save directory
            - blender_save: Blender file save path (optional)
            - gpu_devices: GPU device ID, such as "0" or "0,1" (optional)
            - meshy_api_key: Meshy API key (optional)
            - va_api_key: VA API key (optional)
            - target_image_path: Target image path (optional)
    """
    global _executor
    global tool_configs
    try:
        _executor = Executor(
            blender_command=args.get("blender_command"),
            blender_file=args.get("blender_file"),
            blender_script=args.get("blender_script"),
            script_save=args.get("script_save"),
            render_save=args.get("render_save"),
            blender_save=args.get("blender_save"),
            gpu_devices=args.get("gpu_devices")
        )
        
        return {"status": "success", "output": {"text": ["Executor initialized successfully"], "tool_configs": tool_configs}}
    except Exception as e:
        return {"status": "error", "output": {"text": [str(e)]}}

@mcp.tool()
def execute_and_evaluate(thought: str, code_edition: str, full_code: str, round: int) -> dict:
    """
    Execute the passed Blender Python script code and return base64 encoded rendered image.
    Need to call initialize_executor first for initialization.
    """
    global _executor
    if _executor is None:
        return {"status": "error", "output": {"text": ["Executor not initialized. Call initialize_executor first."]}}
    try:
        result = _executor.execute(thought, code_edition, full_code, round)
        return result
    except Exception as e:
        return {"status": "error", "output": {"text": [str(e)]}}
    
def main():
    # If running this script directly, execute test
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        print("Running blender-executor tools test...")
        # Read args from environment for convenience
        args = {
            "blender_command": os.getenv("BLENDER_COMMAND", "utils/blender/infinigen/blender/blender"),
            "blender_file": os.getenv("BLENDER_FILE", "output/test/exec_blender/test.blend"),
            "blender_script": os.getenv("BLENDER_SCRIPT", "data/dynamic_scene/pipeline_render_script.py"),
            "script_save": os.getenv("SCRIPT_SAVE", "output/test/exec_blender/scripts"),
            "render_save": os.getenv("RENDER_SAVE", "output/test/exec_blender/renders"),
            "blender_save": os.getenv("BLENDER_SAVE", "output/test/exec_blender/test.blend"),
            "gpu_devices": os.getenv("GPU_DEVICES", None),
        }
        
        import bpy
        bpy.ops.wm.save_as_mainfile(filepath=args["blender_file"])
        print(f"Created blender file: {args['blender_file']}")
        
        print("[test] initialize(...) with:", json.dumps({k:v for k,v in args.items() if k!="gpu_devices"}, ensure_ascii=False))
        init_res = initialize(args)
        print("[test:init]", json.dumps(init_res, ensure_ascii=False))

        # Note: The new blender file has a default Camera at position around (7,-6,4), facing direction (0,0,0)
        sample_code = """import bpy
import math

# Clear objects in scene (excluding environment light, camera)
for obj in bpy.context.scene.objects:
    if obj.name not in ["Camera"]:
        obj.select_set(True)
bpy.ops.object.delete(use_global=False)

# Set camera position
bpy.context.scene.camera.location = (14, -12, 8)

# Basic ground
bpy.ops.mesh.primitive_plane_add(size=20, location=(0, 0, 0))
ground = bpy.context.active_object
ground.name = "Ground"

# Slope
bpy.ops.mesh.primitive_plane_add(size=10, location=(0, 0, 2))
slope = bpy.context.active_object
slope.name = "Slope"
slope.rotation_euler = (math.radians(25.0), 0.0, math.radians(20.0))  # Tilt angle (adjustable)

# Ball (active rigid body)
bpy.ops.mesh.primitive_uv_sphere_add(radius=0.5, location=(-3, -2, 4))
ball = bpy.context.active_object
ball.name = "Ball"

# Lighting
bpy.ops.object.light_add(type='SUN', location=(8, -8, 12))
sun = bpy.context.active_object
sun.data.energy = 3.0

# Environment light (world node)
bpy.context.scene.world.use_nodes = True
bg = bpy.context.scene.world.node_tree.nodes['Background']
bg.inputs[1].default_value = 1.0  # Intensity

# ===== Physics Settings =====

# Create/get rigid body world
if not bpy.context.scene.rigidbody_world:
    bpy.ops.rigidbody.world_add()

scene = bpy.context.scene
rw = scene.rigidbody_world

# Gravity (default -9.81 m/s^2)
scene.gravity = (0.0, 0.0, -9.81)

# Time step / substeps & iterations (compatible with different version fields)
# Generally: steps_per_second (old/common), or substeps_per_frame (newer)
if hasattr(rw, "steps_per_second"):
    rw.steps_per_second = 240
elif hasattr(rw, "substeps_per_frame"):
    # Number of substeps (substeps per frame). 10~20 is common; you can increase for better stability
    rw.substeps_per_frame = 10

# Solver iteration count (some versions on world, some on constraint settings, the above is usually available)
if hasattr(rw, "solver_iterations"):
    rw.solver_iterations = 20

# Frame range
scene.frame_start = 1
scene.frame_end = 40

# Add rigid body properties to objects
# Ground: passive
bpy.context.view_layer.objects.active = ground
bpy.ops.rigidbody.object_add()
ground.rigid_body.type = 'PASSIVE'
ground.rigid_body.friction = 0.8
ground.rigid_body.restitution = 0.0
ground.rigid_body.use_deactivation = False

# Slope: passive
bpy.context.view_layer.objects.active = slope
bpy.ops.rigidbody.object_add()
slope.rigid_body.type = 'PASSIVE'
slope.rigid_body.friction = 0.7
slope.rigid_body.restitution = 0.0
slope.rigid_body.use_deactivation = False

# Ball: active
bpy.context.view_layer.objects.active = ball
bpy.ops.rigidbody.object_add()
ball.rigid_body.type = 'ACTIVE'
ball.rigid_body.mass = 1.0
ball.rigid_body.friction = 0.5
ball.rigid_body.restitution = 0.1      # Slight elasticity
ball.rigid_body.collision_shape = 'SPHERE'
ball.rigid_body.use_deactivation = False
ball.rigid_body.linear_damping = 0.05  # Slight damping
ball.rigid_body.angular_damping = 0.05

# To ensure the ball starts from above the slope, fine-tune initial position (avoid initial intersection)
ball.location = (-3, -2, 5.0)

# (Optional) Bake rigid body cache, more stable for background rendering
bpy.ops.ptcache.free_bake_all()
bpy.ops.ptcache.bake_all(bake=True)

print("Scene ready: press Play to watch the ball roll down the slope.")"""
        exec_res = execute_and_evaluate(thought="", code_edition="", full_code=sample_code)
        print("[test:exec_script]", json.dumps(exec_res, ensure_ascii=False))
        
    else:
        # Run MCP service normally
        mcp.run(transport="stdio")

if __name__ == "__main__":
    main()


# static code:
# import bpy
# import math

# # Scene objects
# bpy.ops.mesh.primitive_plane_add(size=4, location=(0,0,0))
# bpy.ops.mesh.primitive_cube_add(size=1, location=(0,0,1))

# # **Add a light** (otherwise it will be dark)
# bpy.ops.object.light_add(type='SUN', location=(5,5,10))
# sun = bpy.context.active_object
# sun.data.energy = 3.0

# # Can also add some environment light (optional)
# bpy.context.scene.world.use_nodes = True
# bg = bpy.context.scene.world.node_tree.nodes['Background']
# bg.inputs[1].default_value = 1.0   # Intensity

# # First check if there are matching files in local assets directory
# if os.path.exists(assets_dir):
#     for asset_file in os.listdir(assets_dir):
#         # Fuzzy matching: remove spaces from object_name and asset_file, convert to lowercase, check if they contain each other
#         new_object_name = object_name.replace(" ", "")
#         new_asset_file = asset_file.replace(" ", "")
#         new_asset_file = new_asset_file.split(".")[0]
#         if new_object_name.lower() in new_asset_file.lower() or new_asset_file.lower() in new_object_name.lower():
#             if asset_file.endswith('.glb') or asset_file.endswith('.obj'):
#                 generate_result = {
#                     'status': 'success',
#                     'message': 'Local asset found',
#                     'object_name': object_name,
#                     'local_path': os.path.join(assets_dir, asset_file),
#                     'save_dir': save_dir
#                 }
#                 break
#         elif os.path.isdir(os.path.join(assets_dir, asset_file)):
#             for asset_file_ in os.listdir(os.path.join(assets_dir, asset_file)):
#                 if object_name.lower() in asset_file_.lower() or asset_file_.lower() in object_name.lower():
#                     if asset_file_.endswith('.glb') or asset_file_.endswith('.obj'):
#                         generate_result = {
#                             'status': 'success',
#                             'message': 'Local asset found',
#                             'object_name': object_name,
#                             'local_path': os.path.join(assets_dir, asset_file, asset_file_),
#                             'save_dir': save_dir
#                         }
#                         break
#             if generate_result is not None:
#                 break