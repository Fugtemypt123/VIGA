# Data

Datasets and task scripts for VIGA evaluation benchmarks.

## Datasets

| Directory | Description | Task Type |
|-----------|-------------|-----------|
| [blendergym/](blendergym/) | Single-step 3D graphics editing | Blender Python |
| [blenderbench/](blenderbench/) | Multi-step 3D scene generation (Levels 1-3) | Blender Python |
| [static_scene/](static_scene/) | 3D scene reconstruction from scratch | Blender Python |
| [dynamic_scene/](dynamic_scene/) | 4D animated scenes with physics | Blender Python |
| [slidebench/](slidebench/) | 2D programmatic slide generation | python-pptx |

## Structure

Each dataset directory contains:

```
<dataset>/
├── README.md              # Dataset-specific documentation
├── generator_script.py    # Blender script for Generator agent
├── verifier_script.py     # Blender script for Verifier agent
└── <task_data>/           # Task-specific files and resources
```

## Usage

Datasets are accessed through the runner scripts:

```bash
# BlenderGym
python runners/blendergym/ours.py --dataset-path data/blendergym --task all

# BlenderBench
python runners/blenderbench/ours.py --dataset-path data/blenderbench --task all

# Static Scene
python runners/static_scene.py --dataset-path data/static_scene --task all

# SlideBench
python runners/slidebench/ours.py --dataset-path data/slidebench --task all
```

## Data Download

See the main [README.md](../README.md) for dataset download instructions.
