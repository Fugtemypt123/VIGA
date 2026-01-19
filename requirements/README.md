# Requirements

Conda environment setup for VIGA. Alternative to Docker-based deployment.

## Quick Start

```bash
# 1. Create main agent environment
conda create -n agent python=3.10 -y
conda activate agent
pip install -r requirements/requirement_agent.txt

# 2. Create tool environments (based on modes you need)
# For 3D modes (BlenderGym, BlenderBench, Static/Dynamic Scene)
conda create -n blender python=3.11 -y
conda activate blender
pip install -r requirements/requirement_blender.txt

# For SlideBench mode
conda create -n pptx python=3.10 -y
conda activate pptx
pip install -r requirements/requirement_pptx.txt
```

## Requirement Files

| File | Environment | Python | Modes |
|------|-------------|--------|-------|
| `requirement_agent.txt` | agent | 3.10 | All (main runtime) |
| `requirement_blender.txt` | blender | 3.11 | 3D modes |
| `requirement_pptx.txt` | pptx | 3.10 | SlideBench |
| `requirement_eval-blender.txt` | eval-blender | 3.11 | 3D evaluation |
| `requirement_eval-pptx.txt` | eval-pptx | 3.10 | Slides evaluation |

## External Dependencies

### Blender (for 3D modes)

```bash
cd utils/third_party
git clone https://github.com/princeton-vl/infinigen.git
cd infinigen
conda activate blender
bash scripts/install/interactive_blender.sh
```

### LibreOffice (for SlideBench)

```bash
sudo apt-get install -y libreoffice unoconv
```

## Configuration

Create two config files in `utils/`:

### `_api_keys.py`

```python
OPENAI_API_KEY = "your-key"
CLAUDE_API_KEY = "your-key"
GEMINI_API_KEY = "your-key"
MESHY_API_KEY = "your-key"
```

### `_path.py`

```python
path_to_cmd = {
    "tools/blender/exec.py": "/path/to/conda/envs/blender/bin/python",
    "tools/blender/investigator.py": "/path/to/conda/envs/blender/bin/python",
    "tools/slides/exec.py": "/path/to/conda/envs/pptx/bin/python",
}
```

Find your conda paths with `conda env list`.

## Verification

```bash
# Test agent environment
conda activate agent
python -c "import openai; import mcp; print('OK')"

# Test blender environment
conda activate blender
python -c "import bpy; print('OK')"

# Test pptx environment
conda activate pptx
python -c "import pptx; print('OK')"
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| ModuleNotFoundError | Ensure correct conda env is activated |
| Infinigen install fails | Check Python 3.11, install `build-essential cmake` |
| CUDA not found | Run `nvidia-smi`, reinstall PyTorch with correct CUDA |
| PPTX conversion fails | Reinstall LibreOffice |
| Wrong Python path | Update `_path.py` with `which python` output |
