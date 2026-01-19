# Utils

Shared utilities and third-party dependencies.

## Files

| File | Description |
|------|-------------|
| `common.py` | Common utilities (logging, image processing, etc.) |
| `path.py` | Tool-to-environment path mapping |
| `_api_keys.py` | API keys configuration (user-created, gitignored) |

## Directories

| Directory | Description |
|-----------|-------------|
| `third_party/` | Git submodules and external dependencies |
| `SlidesLib/` | PowerPoint generation library |
| `library/` | Additional library utilities |

## Configuration Files

Create these files before running VIGA:

### `_api_keys.py`

```python
OPENAI_API_KEY = "your-openai-key"
CLAUDE_API_KEY = "your-claude-key"
GEMINI_API_KEY = "your-gemini-key"
MESHY_API_KEY = "your-meshy-key"
```

### `_path.py` (optional)

Override default tool Python paths if needed. See [requirements/README.md](../requirements/README.md) for details.

## Third-Party Dependencies

Located in `third_party/`:

| Submodule | Purpose |
|-----------|---------|
| `infinigen/` | Blender installation and procedural generation |
| `sam/` | Segment Anything Model |
| `sam3/` | SAM3 extension |
| `sam3d/` | SAM3D reconstruction |
| `slides/` | Slides generation library |

Initialize submodules:

```bash
git submodule update --init --recursive
```
