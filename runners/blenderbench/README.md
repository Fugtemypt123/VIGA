# BlenderBench Runner

Scripts for running VIGA on the BlenderBench benchmark.

## Files

| File | Description |
|------|-------------|
| `ours.py` | Main runner for VIGA agent |
| `main.py` | Alternative entry point |
| `alchemy.py` | Iterative alchemy pipeline |

## Usage

### Run VIGA Agent

```bash
python runners/blenderbench/ours.py \
    --dataset-path data/blenderbench \
    --task all \
    --model gpt-4o \
    --max-rounds 10
```

See [runners/README.md](../README.md) for detailed arguments.
