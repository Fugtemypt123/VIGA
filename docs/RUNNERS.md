# Runner Documentation

Runners provide batch execution for datasets with parallel processing support.

## BlenderGym Runner

Run 3D scene modification tasks:

```bash
python runners/blendergym.py \
    --dataset-path data/blendergym \
    --task all \
    --model gpt-4o \
    --max-rounds 10 \
    --max-workers 8
```

### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--dataset-path` | Path to BlenderGym dataset | data/blendergym |
| `--task` | Task type: all, blendshape, geometry, lighting, material, placement | all |
| `--task-id` | Specific task ID (e.g., "1") | None |
| `--test-id` | Retest failed tasks from previous run | None |
| `--max-rounds` | Maximum interaction rounds | 10 |
| `--model` | Vision model | gpt-4o |
| `--memory-length` | Memory length | 24 |
| `--max-workers` | Parallel workers | 8 |
| `--sequential` | Run sequentially | False |
| `--no-tools` | Disable tools | False |
| `--generator-tools` | Generator tool servers | tools/blender/exec.py,tools/generator_base.py,tools/initialize_plan.py |
| `--verifier-tools` | Verifier tool servers | tools/verifier_base.py,tools/blender/investigator.py |

## BlenderStudio Runner

Run progressive 3D scene generation tasks:

```bash
python runners/blenderstudio.py \
    --dataset-path data/blenderstudio \
    --task all \
    --model gpt-4o \
    --max-workers 8
```

### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--dataset-path` | Path to BlenderStudio dataset | data/blenderstudio |
| `--task` | Task level: all, level1, level2, level3 | all |
| `--task-id` | Specific task ID | None |
| `--test-id` | Retest failed tasks | None |
| `--max-rounds` | Maximum interaction rounds | 10 |
| `--model` | Vision model | gpt-4o |
| `--max-workers` | Parallel workers | 8 |

## Static Scene Runner

Create 3D scenes from scratch using target images:

```bash
python runners/static_scene.py \
    --dataset-path data/static_scene \
    --task all \
    --model gpt-4o \
    --max-rounds 100
```

### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--dataset-path` | Path to static scene dataset | data/static_scene |
| `--task` | Task name or "all" | all |
| `--test-id` | Test ID for output naming | None |
| `--max-rounds` | Maximum interaction rounds | 100 |
| `--model` | Vision model | gpt-5 |
| `--memory-length` | Memory length | 12 |
| `--prompt-setting` | Prompt setting: none, procedural, scene_graph, get_asset | none |
| `--init-setting` | Init setting: none, minimal, reasonable | none |
| `--max-workers` | Parallel workers | 1 |
| `--text-only` | Use only text as reference | False |
| `--generator-tools` | Generator tools | tools/blender/exec.py,tools/generator_base.py,tools/assets/meshy.py,tools/initialize_plan.py |
| `--verifier-tools` | Verifier tools | tools/blender/investigator.py,tools/verifier_base.py |

## Dynamic Scene Runner

Generate animated 3D scenes:

```bash
python runners/dynamic_scene.py \
    --dataset-path data/dynamic_scene \
    --task all \
    --model gpt-4o \
    --max-rounds 100
```

### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--dataset-path` | Path to dynamic scene dataset | data/dynamic_scene |
| `--task` | Task name or "all" | all |
| `--test-id` | Test ID for output naming | None |
| `--max-rounds` | Maximum interaction rounds | 100 |
| `--model` | Vision model | gpt-5 |
| `--prompt-setting` | Prompt setting: none, init | none |
| `--text-only` | Use only text as reference | False |
| `--generator-tools` | Generator tools | tools/blender/exec.py,tools/generator_base.py,tools/initialize_plan.py |
| `--verifier-tools` | Verifier tools | tools/blender/investigator.py,tools/verifier_base.py |

## AutoPresent Runner

Generate presentation slides:

```bash
python runners/autopresent.py \
    --dataset-path data/autopresent/examples \
    --task all \
    --model gpt-4o \
    --max-workers 8
```

### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--dataset-path` | Path to AutoPresent dataset | data/autopresent/examples |
| `--task` | Task category: all, art_photos, business, design, entrepreneur, environment, food, marketing, social_media, technology | all |
| `--task-id` | Specific task ID | None |
| `--test-id` | Retest failed tasks | None |
| `--max-rounds` | Maximum interaction rounds | 10 |
| `--model` | Vision model | gpt-4o |
| `--max-workers` | Parallel workers | 8 |
| `--sequential` | Run sequentially | False |
| `--generator-tools` | Generator tools | tools/slides/exec.py,tools/generator_base.py |
| `--verifier-tools` | Verifier tools | tools/verifier_base.py |

## Main Entry Point

For running individual tasks, use the main entry point:

```bash
python main.py --mode <mode> --model <model> [options]
```

### Common Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--mode` | Mode: blendergym, blenderstudio, static_scene, dynamic_scene, autopresent | Required |
| `--model` | Vision model (gpt-4o, claude-sonnet-4, gemini-2.5-pro, etc.) | gpt-4o |
| `--api-key` | API key for the model | From env |
| `--api-base-url` | API base URL | From env |
| `--max-rounds` | Maximum interaction rounds | 10 |
| `--memory-length` | Chat history length | 12 |
| `--output-dir` | Output directory | None |
| `--init-code-path` | Path to initial code file | None |
| `--init-image-path` | Path to initial images | None |
| `--target-image-path` | Path to target images | None |
| `--target-description` | Natural language task description | None |
| `--generator-tools` | Comma-separated generator tool scripts | tools/generator_base.py |
| `--verifier-tools` | Comma-separated verifier tool scripts | tools/verifier_base.py |
| `--no-tools` | Disable tool calling mode | False |
| `--clear-memory` | Clear memory between rounds | False |

### Blender-specific Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--blender-command` | Path to Blender executable | utils/infinigen/blender/blender |
| `--blender-file` | Blender template file | None |
| `--blender-script` | Blender execution script | data/blendergym/pipeline_render_script.py |
| `--gpu-devices` | GPU devices for rendering | From CUDA_VISIBLE_DEVICES |
