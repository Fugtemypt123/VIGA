# Prompts

System prompts and examples for VIGA agents across different modes.

## Directory Structure

```
prompts/
├── prompt_manager.py      # Prompt loading utilities
│
├── blendergym/            # BlenderGym mode
│   ├── generator_system.md
│   ├── verifier_system.md
│   └── examples/
│
├── blenderbench/          # BlenderBench mode
│   └── ...
│
├── static_scene/          # Static scene mode
│   └── ...
│
├── dynamic_scene/         # Dynamic scene mode
│   └── ...
│
└── slidebench/            # SlideBench mode
    └── ...
```

## Prompt Types

| Type | Description |
|------|-------------|
| `generator_system.md` | System prompt for Generator agent |
| `verifier_system.md` | System prompt for Verifier agent |
| `examples/` | Few-shot examples for in-context learning |

## Usage

```python
from prompts.prompt_manager import PromptManager

pm = PromptManager(mode="blendergym")
generator_prompt = pm.get_generator_system()
verifier_prompt = pm.get_verifier_system()
```
