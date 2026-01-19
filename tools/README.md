# Tools

MCP (Model Context Protocol) tool servers for VIGA agents.

## Directory Structure

```
tools/
├── generator_base.py       # Generator end_process tool
├── verifier_base.py        # Verifier end_process tool
├── initialize_plan.py      # make_plan tool
│
├── blender/                # Blender execution tools
│   ├── exec.py             # execute_code, undo, render
│   ├── investigator.py     # set_camera, investigate, get_scene_info
│   ├── investigator_core.py
│   ├── script_generators.py
│   └── glb_import.py
│
├── slides/                 # PPTX generation tools
│   └── exec.py             # execute_code for slides
│
├── assets/                 # 3D asset generation
│   ├── meshy.py            # get_better_assets via Meshy API
│   └── meshy_api.py        # Meshy API client
│
└── sam3d/                  # SAM-based scene reconstruction
    ├── init.py             # Scene initialization tools
    ├── bridge.py           # SAM model bridge
    ├── sam_worker.py       # SAM segmentation worker
    ├── sam3_worker.py      # SAM3 worker
    └── sam3d_worker.py     # SAM3D worker
```

## Tool Servers

| Server | Tools | Environment |
|--------|-------|-------------|
| `generator_base.py` | `end_process` | agent |
| `verifier_base.py` | `end_process` | agent |
| `initialize_plan.py` | `make_plan` | agent |
| `blender/exec.py` | `execute_code`, `undo`, `render` | blender |
| `blender/investigator.py` | `set_camera`, `investigate`, `get_scene_info` | blender |
| `slides/exec.py` | `execute_code` | pptx |
| `assets/meshy.py` | `get_better_assets` | agent |
| `sam3d/init.py` | `initialize_scene` | sam3d |

## Configuration

Tool-to-environment mapping is configured in [utils/path.py](../utils/path.py).

## Subdirectories

- [blender/](blender/) - Blender Python execution and scene investigation
- [slides/](slides/) - PowerPoint slide generation
- [assets/](assets/) - 3D asset generation via Meshy API
- [sam3d/](sam3d/) - SAM-based 3D scene reconstruction
