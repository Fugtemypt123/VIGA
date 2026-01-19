# Architecture

How VIGA reconstructs visual content through iterative code generation.

## Core Loop

VIGA uses a **dual-agent** design where two specialized agents collaborate:

```
┌─────────────────┐
│   Target Image  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│    Generator    │────▶│     Render      │
│  (writes code)  │     │  (runs code)    │
└─────────────────┘     └────────┬────────┘
         ▲                       │
         │                       ▼
         │               ┌─────────────────┐
         └───────────────│    Verifier     │
            feedback     │ (analyzes diff) │
                         └─────────────────┘
```

**Generator** writes code to construct or modify the scene.
**Verifier** compares the rendered output to the target and provides feedback.
The loop continues until the output matches the target.

## Agent Tools

### Generator Tools

| Tool | Purpose |
|------|---------|
| `make_plan` | Write a high-level action plan before editing |
| `execute_code` | Run program to construct/update the scene |
| `get_scene_info` | Query object attributes (position, rotation, scale) |
| `get_better_assets` | Retrieve or generate improved 3D assets |
| `end_process` | Signal completion |

### Verifier Tools

| Tool | Purpose |
|------|---------|
| `initialize_viewpoint` | Compute canonical diagnostic views |
| `set_camera` | Move camera to specific pose |
| `investigate` | Adjust camera via natural language ("rotate left") |
| `set_keyframe` | Navigate timeline in 4D scenes |
| `set_visibility` | Toggle object visibility |
| `get_scene_info` | Summarize scene state as text |
| `end_process` | Signal completion |

## Memory Management

VIGA uses a **sliding window** to manage conversation context:

- Keeps the most recent L iterations (typically L=12-24)
- Reduces cost from O(N^2) to O(N*k)
- Balances context length with reasoning quality

## Project Layout

```
VIGA/
├── main.py              # Entry point
├── agents/              # Generator and Verifier implementations
├── tools/               # MCP tool servers
├── prompts/             # System prompts per mode
├── runners/             # Batch execution scripts
├── evaluators/          # Metric computation
├── data/                # Benchmark datasets
├── utils/               # Shared utilities
└── output/              # Run outputs
```

## Output Structure

Each run produces a trajectory capturing the iterative process:

```
output/<mode>/<timestamp>/<task>/
├── generator_thoughts/   # Generator reasoning per iteration
├── verifier_thoughts/    # Verifier analysis per iteration
├── renders/              # Rendered images (1/, 2/, ...)
├── codes/                # Generated code per iteration
├── scores.json           # Final metrics
└── blender_file.blend    # Final scene (3D modes)
```

## Supported Models

VIGA works with multiple vision-language models:

- **OpenAI**: gpt-4o, gpt-4-turbo, gpt-4o-mini
- **Anthropic**: claude-sonnet-4, claude-opus-4.5
- **Google**: gemini-2.5-pro, gemini-2.0-flash
- **Qwen**: qwen-vl-max, qwen-vl-plus
