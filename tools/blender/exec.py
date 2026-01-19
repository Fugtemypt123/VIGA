"""Blender Executor MCP Server for executing Blender Python scripts.

This module provides an MCP server that manages Blender script execution,
rendering, and scene manipulation. It supports tool calls from the Generator
agent to execute code, get scene information, and undo operations.
"""

import base64
import io
import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from mcp.server.fastmcp import FastMCP
from PIL import Image

from tools.blender.script_generators import generate_scene_info_script

# Tool configuration dictionaries for the Generator agent
execute_and_evaluate_tool: Dict[str, object] = {
    "type": "function",
    "function": {
        "name": "execute_and_evaluate",
        "description": "Execute blender python code and trigger verifier evaluation.\nReturns either:\n(1) On error: detailed error information; or \n(2) On success: a clear render (you must add a camera in your code) and further modification suggestions from a separate verifier agent.",
        "parameters": {
            "type": "object",
            "properties": {
                "thought": {
                    "type": "string",
                    "description": "Think step by step about the current scene and reason about what code to write next. Describe your reasoning process clearly."
                },
                "code_diff": {
                    "type": "string",
                    "description": "Before outputting the final code, precisely list the line-level edits you will make. Use this minimal diff-like format ONLY:\n\n-: [lines to remove]\n+: [lines to add]\n\nRules:\n1) Show only the smallest necessary edits (avoid unrelated changes).\n2) Keep ordering: list removals first, then additions.\n3) Do not include commentary here—only the edit blocks.\n4) If starting from scratch, use `-: []` and put all new lines under `+: [...]`.\n5) Every line is a literal code line (no markdown, no fences)."
                },
                "code": {
                    "type": "string",
                    "description": "Provide the COMPLETE, UPDATED Blender Python code AFTER applying the edits listed in `code_diff`. The full code must include both the modified lines and the unchanged lines to ensure a coherent, runnable script."
                }
            },
            "required": ["thought", "code_diff", "code"]
        }
    }
}

get_scene_info_tool: Dict[str, object] = {
    "type": "function",
    "function": {
        "name": "get_scene_info",
        "description": "Get the scene information including objects, materials, lights, and cameras. This tool provides detailed information about the current state of the Blender scene, which can be used to understand what objects exist and their properties.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
}

undo_last_step_tool: Dict[str, object] = {
    "type": "function",
    "function": {
        "name": "undo_last_step",
        "description": "If you believe that your last action did not improve the current state, but instead moved it further away from the target state, you can call this tool to undo the last action.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
}

mcp = FastMCP("blender-executor")

# Global executor instance
_executor: Optional["Executor"] = None

class Executor:
    """Manages Blender script execution and rendering.

    Handles the lifecycle of Blender script execution including file management,
    subprocess invocation, and result collection.

    Attributes:
        blender_command: Path to the Blender executable.
        blender_file: Path to the .blend file to operate on.
        blender_script: Path to the wrapper script that executes user code.
        script_path: Directory to save generated scripts.
        render_path: Directory to save rendered images.
        blender_save: Optional path to save the Blender state after execution.
        gpu_devices: Comma-separated GPU device IDs (e.g., "0,1").
        count: Counter for executed scripts.
    """

    def __init__(
        self,
        blender_command: str,
        blender_file: str,
        blender_script: str,
        script_save: str,
        render_save: str,
        blender_save: Optional[str] = None,
        gpu_devices: Optional[str] = None
    ) -> None:
        """Initialize the Blender executor.

        Args:
            blender_command: Path to Blender executable.
            blender_file: Path to .blend file.
            blender_script: Path to wrapper script.
            script_save: Directory to save scripts.
            render_save: Directory to save renders.
            blender_save: Optional path to save Blender state.
            gpu_devices: Optional GPU device IDs.
        """
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

    def _execute_blender(
        self, script_path: str, render_path: str = ''
    ) -> Tuple[bool, List[str], str, str]:
        """Execute a Blender script in background mode.

        Args:
            script_path: Path to the Python script to execute.
            render_path: Directory to save rendered images.

        Returns:
            Tuple of (success, image_paths, stdout, stderr).
        """
        cmd = [
            self.blender_command,
            "--background", self.blender_file,
            "--python", self.blender_script,
            "--", script_path, render_path
        ]
        if self.blender_save:
            cmd.append(self.blender_save)
        cmd_str = " ".join(cmd)
        
        # Set environment variables to control GPU devices
        env = os.environ.copy()
        if self.gpu_devices:
            env['CUDA_VISIBLE_DEVICES'] = self.gpu_devices
            logging.info(f"Setting CUDA_VISIBLE_DEVICES to: {self.gpu_devices}")
            
        # Ban blender audio error
        env['AL_LIB_LOGLEVEL'] = '0'
        
        try:
            proc = subprocess.run(cmd_str, shell=True, check=True, capture_output=True, text=True, env=env)
            out = proc.stdout
            err = proc.stderr
            if os.path.isdir(render_path):
                imgs = sorted([str(p) for p in Path(render_path).glob("*") if p.suffix in ['.png','.jpg']])
                if len(imgs) > 0:
                    return True, imgs, out, err
            return True, [], out, err
        except subprocess.CalledProcessError as e:
            logging.error(f"Blender failed: {e}")
            return False, [], e.stdout, e.stderr

    def _encode_image(self, img_path: str) -> str:
        """Encode an image file to base64 string."""
        img = Image.open(img_path)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    def _parse_code(self, full_code: str) -> str:
        """Strip markdown code fences from code if present."""
        if full_code.startswith("```python") and full_code.endswith("```"):
            return full_code[len("```python"):-len("```")]
        return full_code

    def _generate_scene_info_script(self) -> str:
        """Generate a script to extract scene information."""
        return generate_scene_info_script(str(self.render_path.parent / "tmp" / "scene_info.json"))

    def execute(self, code: str) -> Dict[str, object]:
        """Execute Blender code and return results.

        Args:
            code: Python code to execute in Blender.

        Returns:
            Dictionary with status and output (text, images, or errors).
        """
        self.count += 1
        code_file = self.script_path / f"{self.count}.py"
        render_file = self.render_path / f"{self.count}"
        code = self._parse_code(code)
        
        # File operations
        with open(code_file, "w") as f:
            f.write(code)
        os.makedirs(render_file, exist_ok=True)
        for img in os.listdir(render_file):
            os.remove(os.path.join(render_file, img))
            
        # Execute Blender
        success, imgs, stdout, stderr = self._execute_blender(str(code_file), str(render_file))
        # Check if render_file is empty or not exist
        if not success:
            os.rmdir(render_file)
            return {"status": "error", "output": {"text": ['Error: ' + (stderr + stdout)]}}
        elif len(os.listdir(render_file)) == 0:
            # copy blender save under render file
            if self.blender_save:
                shutil.copy(self.blender_save, render_file / "state.blend")
            return {"status": "success", "output": {"text": ['The code was executed, but no image was generated. Please check and make sure that:\n(1) you have added the camera in the code (just modify the camera pose and other information, do not render the image in the code).\n(2) You may need to handle errors in the code. The following is the return message for reference. Please check if there are any errors and fix them: ' + (stderr + stdout)]}}
        else:
            if self.blender_save:
                shutil.copy(self.blender_save, render_file / "state.blend")
            return {"status": "success", "output": {"image": imgs, "text": [f"Render from camera {x}" for x in range(len(imgs))], 'require_verifier': True}}

    def get_scene_info(self) -> Dict[str, object]:
        """Get scene information by executing a Blender script."""
        try:
            # Create tmp directory if it doesn't exist
            tmp_dir = self.render_path.parent / "tmp"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate and execute scene info script
            scene_info_script = self._generate_scene_info_script()
            self.count += 1
            code_file = self.script_path / f"{self.count}.py"
            
            with open(code_file, "w") as f:
                f.write(scene_info_script)
            
            # Execute Blender script
            success, imgs, stdout, stderr = self._execute_blender(str(code_file))
            
            if not success:
                return {"status": "error", "output": {"text": ['Error: ' + (stderr or stdout)]}}
            
            # Read scene info from file
            scene_info_path = tmp_dir / "scene_info.json"
            if scene_info_path.exists():
                with open(scene_info_path, "r") as f:
                    scene_info = json.load(f)
                    return {"status": "success", "output": {"text": [str(scene_info)]}}
            else:
                return {"status": "error", "output": {"text": ["Failed to extract scene information"]}}
                
        except Exception as e:
            return {"status": "error", "output": {"text": [str(e)]}}



@mcp.tool()
def initialize(args: Dict[str, object]) -> Dict[str, object]:
    """Initialize Blender executor and set all necessary parameters.

    Args:
        args: Dictionary containing configuration keys including blender_command,
              blender_file, blender_script, output_dir, blender_save, gpu_devices.
    """
    global _executor
    try:
        _executor = Executor(
            blender_command=args.get("blender_command"),
            blender_file=args.get("blender_file"),
            blender_script=args.get("blender_script"),
            script_save=args.get("output_dir") + "/scripts",
            render_save=args.get("output_dir") + "/renders",
            blender_save=args.get("blender_save"),
            gpu_devices=args.get("gpu_devices")
        )
        if 'blender' in args.get("mode"):
            tool_configs = [execute_and_evaluate_tool]
        else:
            tool_configs = [execute_and_evaluate_tool, get_scene_info_tool, undo_last_step_tool]
        return {"status": "success", "output": {"text": ["Executor initialized successfully"], "tool_configs": tool_configs}}
    except Exception as e:
        return {"status": "error", "output": {"text": [str(e)]}}

@mcp.tool()
def execute_and_evaluate(thought: str = '', code_diff: str = '', code: str = '') -> Dict[str, object]:
    """Execute Blender Python script and return rendered image."""
    global _executor
    if _executor is None:
        return {"status": "error", "output": {"text": ["Executor not initialized. Call initialize_executor first."]}}
    try:
        result = _executor.execute(code)
        return result
    except Exception as e:
        return {"status": "error", "output": {"text": [str(e)]}}

@mcp.tool()
def undo_last_step() -> Dict[str, object]:
    """Undo the last executed step by reverting to previous state."""
    global _executor
    if _executor is None:
        return {"status": "error", "output": {"text": ["Executor not initialized. Call initialize_executor first."]}}
    render_path = _executor.render_path / f"{_executor.count}"
    code_path = _executor.script_path / f"{_executor.count}.py"
    if os.path.exists(code_path):
        os.remove(code_path)
    if os.path.exists(render_path):
        shutil.rmtree(render_path)
    _executor.count -= 1
    render_path = _executor.render_path / f"{_executor.count}"
    if os.path.exists(render_path / "state.blend") and _executor.blender_save:
        shutil.copy(render_path / "state.blend", _executor.blender_save)
    # If the path do not exist, then last step is error, no need to undo
    return {"status": "success", "output": {"text": ["Last step undone successfully"]}}

@mcp.tool()
def get_scene_info() -> Dict[str, object]:
    """Get scene information including objects, materials, lights, and cameras."""
    global _executor
    if _executor is None:
        return {"status": "error", "output": {"text": ["Executor not initialized. Call initialize_executor first."]}}
    try:
        result = _executor.get_scene_info()
        return result
    except Exception as e:
        return {"status": "error", "output": {"text": [str(e)]}}
    
def main() -> None:
    """Run MCP server or execute test mode."""
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        print("Running blender-executor tools test...")
        # Read args from environment for convenience
        args = {
            "mode": "blenderstudio",
            "blender_command": os.getenv("BLENDER_COMMAND", "utils/infinigen/blender/blender"),
            "blender_file": os.getenv("BLENDER_FILE", "data/static_scene/christmas1/reasonable_init/christmas1_gt.blend"),
            "blender_script": os.getenv("BLENDER_SCRIPT", "data/static_scene/generator_script.py"),
            "output_dir": os.getenv("OUTPUT_DIR", "output/test/exec_blender"),
            "blender_save": os.getenv("BLENDER_SAVE", None),
            "gpu_devices": os.getenv("GPU_DEVICES", None),
        }
        
        print("[test] initialize(...) with:", json.dumps({k:v for k,v in args.items() if k!="gpu_devices"}, ensure_ascii=False))
        init_res = initialize(args)
        print("[test:init]", init_res)

        # Test get_scene_info
        scene_info_res = get_scene_info()
        print("[test:get_scene_info]", json.dumps(scene_info_res, ensure_ascii=False))
        raise NotImplementedError

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
        exec_res = execute_and_evaluate(thought="", full_code=sample_code)
        print("[test:exec_script]", json.dumps(exec_res, ensure_ascii=False))
        
    else:
        # Run MCP service normally
        mcp.run()

if __name__ == "__main__":
    main()