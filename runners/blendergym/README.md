# BlenderGym Runner

Scripts for running VIGA on the BlenderGym benchmark.

## Files

| File | Description |
|------|-------------|
| `ours.py` | Main runner for VIGA agent |
| `baseline.py` | Baseline model runner |
| `alchemy.py` | Iterative alchemy pipeline |
| `run_all_code.py` | Batch code execution |

## Usage

### Run VIGA Agent

```bash
python runners/blendergym/ours.py \
    --dataset-path data/blendergym \
    --task all \
    --model gpt-4o \
    --max-rounds 10
```

### Run Baseline

```bash
python runners/blendergym/baseline.py \
    --dataset-path data/blendergym \
    --task all
```

See [runners/README.md](../README.md) for detailed arguments.
