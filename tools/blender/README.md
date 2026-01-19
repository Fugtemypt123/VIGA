# Blender Tools

MCP tool servers for Blender Python execution and scene investigation.

## Files

| File | Description |
|------|-------------|
| `exec.py` | Code execution server (`execute_code`, `undo`, `render`) |
| `investigator.py` | Camera and scene tools (`set_camera`, `investigate`, `get_scene_info`) |
| `investigator_core.py` | Core investigation logic |
| `script_generators.py` | Blender script generation utilities |
| `glb_import.py` | GLB/GLTF model import utilities |

## Tools

### exec.py

- `execute_code` - Execute Blender Python code in the scene
- `undo` - Undo the last operation
- `render` - Render the current scene

### investigator.py

- `set_camera` - Move camera to specific position/rotation
- `investigate` - Adjust camera via natural language commands
- `get_scene_info` - Get object attributes and scene summary

## Environment

Requires `blender` conda environment (Python 3.11).
