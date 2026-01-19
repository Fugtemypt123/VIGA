# BlenderGym Evaluation

Evaluation scripts for BlenderGym benchmark results.

## Files

| File | Description |
|------|-------------|
| `evaluate.py` | Compute metrics for agent outputs |
| `gather.py` | Aggregate results across tasks |

## Usage

### Step 1: Evaluate

```bash
python evaluators/blendergym/evaluate.py \
    --output-dir output/blendergym/<run_id>
```

### Step 2: Gather Results

```bash
python evaluators/blendergym/gather.py \
    --output-dir output/blendergym/<run_id>
```

## Metrics

See [evaluators/README.md](../README.md) for metric descriptions.
