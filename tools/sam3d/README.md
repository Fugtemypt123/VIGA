# SAM3D Tools

SAM-based 3D scene reconstruction tools.

## Files

| File | Description |
|------|-------------|
| `init.py` | MCP server for scene initialization |
| `bridge.py` | Bridge between MCP server and SAM workers |
| `sam_worker.py` | SAM segmentation worker process |
| `sam3_worker.py` | SAM3 worker process |
| `sam3d_worker.py` | SAM3D worker process |

## Architecture

```
init.py (MCP Server)
    │
    ▼
bridge.py (coordinates workers)
    │
    ├── sam_worker.py   (2D segmentation)
    ├── sam3_worker.py  (3D extension)
    └── sam3d_worker.py (3D reconstruction)
```

## Dependencies

Requires models in `utils/third_party/`:
- `sam/` - Segment Anything Model
- `sam3/` - SAM3 extension
- `sam3d/` - SAM3D reconstruction

## Environment

Workers run in the `sam3d` conda environment.
