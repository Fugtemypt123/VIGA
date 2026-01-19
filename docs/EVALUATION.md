# Evaluation

## Metrics

VIGA is evaluated using three complementary metrics:

| Metric | Description |
|--------|-------------|
| **PL Loss** | Program-level loss measuring textual and structural differences between predicted and target code (lower is better) |
| **N-CLIP Score** | Negative-CLIP score measuring semantic alignment between rendered results and target images (lower is better) |
| **VLM Score** | VLM-based metric rating task completion, visual quality, spatial accuracy, and detail fidelity on a 0–5 scale (higher is better) |

## BlenderBench

We introduce **BlenderBench**, a more challenging benchmark that extends BlenderGym with 30 curated tasks covering:

| Task Type | Description |
|-----------|-------------|
| **Task 1: Camera Adjustment** | Adjust camera parameters to align with the target's viewpoint |
| **Task 2: Multi-step Graphic Editing** | Perform multi-step reasoning to infer latent variables such as lighting or occluded object states |
| **Task 3: Compositional Graphic Editing** | Simulate complex real-world environments that combine both spatial alignment and reasoning challenges |

BlenderBench addresses limitations of existing benchmarks by requiring both observation tools (for viewpoint control) and contextual memory (for multi-step reasoning).

## Running Evaluations

Evaluation scripts are located in the `evaluators/` directory. Each mode has its own evaluation script that computes the relevant metrics based on the output of the agent runs.

The evaluation process typically involves:
1. Running the agent on a dataset (see [Runners documentation](RUNNERS.md))
2. Executing the appropriate evaluation script on the output directory
3. Analyzing the computed metrics to assess performance

See the mode-specific evaluation scripts in `evaluators/` for detailed usage instructions.
