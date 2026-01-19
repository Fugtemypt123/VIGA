# BlenderBench Evaluation

Evaluation scripts for BlenderBench benchmark results.

## Files

| File | Description |
|------|-------------|
| `ref_based_eval.py` | Reference-based evaluation |
| `ref_free_eval.py` | Reference-free VLM evaluation |
| `gather.py` | Aggregate results across tasks |

## Usage

### Reference-Based Evaluation

```bash
python evaluators/blenderbench/ref_based_eval.py \
    --output-dir output/blenderbench/<run_id>
```

### Reference-Free Evaluation

Requires OpenAI API key for VLM scoring:

```bash
export OPENAI_API_KEY="your_api_key"
python evaluators/blenderbench/ref_free_eval.py \
    --output-dir output/blenderbench/<run_id>
```

### Gather Results

```bash
python evaluators/blenderbench/gather.py \
    --output-dir output/blenderbench/<run_id>
```

## Metrics

See [evaluators/README.md](../README.md) for metric descriptions.
