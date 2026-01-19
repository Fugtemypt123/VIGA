"""Investigator MCP Server for 3D Scene Analysis.

Provides tools for camera manipulation, scene inspection, and viewpoint
management in Blender scenes. Used by the Verifier agent to analyze
generated 3D content from multiple angles.
"""

import os
import sys
from typing import Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from tools.blender.investigator_core import Investigator3D

# Tool configuration for agent
tool_configs: List[Dict[str, object]] = [
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
                    "operation": {"type": "string", "enum": ["zoom", "move", "focus"], "description": "The operation to perform."},
                    "object_name": {"type": "string", "description": "If the operation is focus, you need to provide the name of the object to focus on. The object must exist in the scene."},
                    "direction": {"type": "string", "enum": ["up", "down", "left", "right", "in", "out"], "description": "If the operation is move or zoom, you need to provide the direction to move or zoom."}
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

# Create MCP instance
mcp = FastMCP("scene-server")

# Global investigator instance
_investigator: Optional[Investigator3D] = None


@mcp.tool()
def initialize(args: Dict[str, object]) -> Dict[str, object]:
    """Initialize the 3D scene investigation tool.

    Args:
        args: Configuration dictionary with 'output_dir', 'blender_file',
            'blender_command', 'blender_script', and 'gpu_devices' keys.

    Returns:
        Dictionary with status and tool configurations on success.
    """
    global _investigator
    try:
        save_dir = args.get("output_dir") + "/investigator/"
        blender_script = os.path.dirname(args.get("blender_script")) + "/verifier_script.py"
        _investigator = Investigator3D(
            save_dir,
            str(args.get("blender_file")),
            str(args.get("blender_command")),
            blender_script,
            str(args.get("gpu_devices"))
        )
        return {
            "status": "success",
            "output": {
                "text": ["Investigator3D initialized successfully"],
                "tool_configs": tool_configs
            }
        }
    except Exception as e:
        return {"status": "error", "output": {"text": [str(e)]}}


@mcp.tool()
def get_scene_info() -> Dict[str, object]:
    """Get information about the current scene.

    Returns:
        Dictionary with scene objects, materials, lights, and cameras.
    """
    global _investigator
    if _investigator is None:
        return {"status": "error", "output": {"text": ["Not initialized. Call initialize first."]}}
    return _investigator.get_info()


def focus(object_name: str) -> Dict[str, object]:
    """Focus camera on a specific object.

    Args:
        object_name: Name of the object to focus on.

    Returns:
        Dictionary with status and rendered image.
    """
    global _investigator
    if _investigator is None:
        return {"status": "error", "output": {"text": ["Not initialized. Call initialize first."]}}
    return _investigator.focus_on_object(object_name)


def zoom(direction: str) -> Dict[str, object]:
    """Zoom camera in or out.

    Args:
        direction: Either 'in' or 'out'.

    Returns:
        Dictionary with status and rendered image.
    """
    global _investigator
    if _investigator is None:
        return {"status": "error", "output": {"text": ["Not initialized. Call initialize first."]}}
    return _investigator.zoom(direction)


def move(direction: str) -> Dict[str, object]:
    """Move camera around target object.

    Args:
        direction: One of 'up', 'down', 'left', 'right'.

    Returns:
        Dictionary with status and rendered image.
    """
    global _investigator
    if _investigator is None:
        return {"status": "error", "output": {"text": ["Not initialized. Call initialize first."]}}
    return _investigator.move_camera(direction)


@mcp.tool()
def initialize_viewpoint(object_names: List[str] = []) -> Dict[str, object]:
    """Initialize viewpoints around specified objects.

    Args:
        object_names: List of object names to observe. Empty for all objects.

    Returns:
        Dictionary with camera positions and rendered images.
    """
    global _investigator
    if _investigator is None:
        return {"status": "error", "output": {"text": ["Not initialized. Call initialize first."]}}
    return _investigator.initialize_viewpoint(object_names)


@mcp.tool()
def investigate(
    operation: str = '',
    object_name: str = '',
    direction: str = ''
) -> Dict[str, object]:
    """Investigate the scene with camera operations.

    Args:
        operation: One of 'focus', 'zoom', or 'move'.
        object_name: Required for 'focus' operation.
        direction: Required for 'zoom' (in/out) or 'move' (up/down/left/right).

    Returns:
        Dictionary with status and rendered image.
    """
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
            return {"status": "error", "output": {"text": ["direction must be up/down/left/right for move"]}}
        return move(direction=direction)
    else:
        return {"status": "error", "output": {"text": [f"Unknown operation: {operation}"]}}


@mcp.tool()
def set_visibility(
    show_objects: List[str] = [],
    hide_objects: List[str] = []
) -> Dict[str, object]:
    """Set visibility of objects in the scene.

    Args:
        show_objects: List of object names to show.
        hide_objects: List of object names to hide.

    Returns:
        Dictionary with status and rendered image.
    """
    global _investigator
    if _investigator is None:
        return {"status": "error", "output": {"text": ["Not initialized. Call initialize first."]}}
    return _investigator.set_visibility(show_objects, hide_objects)


@mcp.tool()
def set_keyframe(frame_number: int = 1) -> Dict[str, object]:
    """Set the scene to a specific frame number.

    Args:
        frame_number: The frame number to set.

    Returns:
        Dictionary with status and rendered image.
    """
    global _investigator
    if _investigator is None:
        return {"status": "error", "output": {"text": ["Not initialized. Call initialize first."]}}
    return _investigator.set_keyframe(frame_number)


@mcp.tool()
def set_camera(
    location: List[float] = [0, 0, 0],
    rotation_euler: List[float] = [0, 0, 0]
) -> Dict[str, object]:
    """Set camera position and rotation.

    Args:
        location: Camera location in world coordinates [x, y, z].
        rotation_euler: Camera rotation in euler angles [x, y, z].

    Returns:
        Dictionary with status and rendered image.
    """
    global _investigator
    if _investigator is None:
        return {"status": "error", "output": {"text": ["Not initialized. Call initialize first."]}}
    return _investigator.set_camera(location, rotation_euler)


@mcp.tool()
def reload_scene() -> Dict[str, object]:
    """Reload the original Blender scene file.

    Returns:
        Dictionary with status message.
    """
    global _investigator
    if _investigator is None:
        return {"status": "error", "output": {"text": ["Not initialized. Call initialize first."]}}
    _investigator.executor.blender_file = _investigator.blender_file
    return {"status": "success", "output": {"text": ["Scene reloaded successfully"]}}


def main() -> None:
    """Run the MCP server or execute test mode."""
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        print("Running investigator tools test...")
        test_tools()
    else:
        mcp.run()


def test_tools() -> None:
    """Test investigator tool functions using environment variable configuration."""
    print("=" * 50)
    print("Testing Scene Tools")
    print("=" * 50)

    # Read test paths from environment variables
    blender_file = os.getenv("BLENDER_FILE", "data/blendergym/placement2/blender_file.blend")
    test_save_dir = os.getenv("THOUGHT_SAVE", "output/test/investigator/")
    blender_command = os.getenv("BLENDER_COMMAND", "utils/blender/infinigen/blender/blender")
    blender_script = os.getenv("BLENDER_SCRIPT", "data/blendergym/pipeline_render_script.py")
    gpu_devices = os.getenv("GPU_DEVICES", "0,1,2,3,4,5,6,7")

    if not os.path.exists(blender_file):
        print(f"Blender file not found: {blender_file}")
        print("Skipping all tests.")
        return

    print(f"Using blender file: {blender_file}")

    # Test initialize
    print("\n1. Testing initialize...")
    args = {
        "output_dir": test_save_dir,
        "blender_file": blender_file,
        "blender_command": blender_command,
        "blender_script": blender_script,
        "gpu_devices": gpu_devices
    }
    result = initialize(args)
    print(f"Result: {result}")

    # Test get scene info
    print("\n2. Testing get_scene_info...")
    scene_info = get_scene_info()
    print(f"Result: {scene_info}")

    print("\n" + "=" * 50)
    print("Test completed!")
    print("=" * 50)
    print(f"\nTest files saved to: {test_save_dir}")


if __name__ == "__main__":
    main()
