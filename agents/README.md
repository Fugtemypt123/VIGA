# Agents

Core agent implementations for the VIGA dual-agent system.

## Architecture

VIGA uses a dual-agent loop where Generator and Verifier agents collaborate:

```
Target Image → Generator (code) → Render → Verifier (analysis) → Memory → Loop
```

## Files

| File | Description |
|------|-------------|
| `generator.py` | Generator agent - synthesizes code using tools |
| `verifier.py` | Verifier agent - analyzes rendered outputs |
| `tool_client.py` | MCP tool client for external tool communication |
| `prompt_builder.py` | Builds prompts with memory and context |

## Generator Agent

The Generator agent writes code to modify 3D scenes, slides, or HTML based on:
- Target image analysis
- Verifier feedback
- Tool outputs (scene info, renders)

## Verifier Agent

The Verifier agent analyzes rendered outputs and provides:
- Viewpoint recommendations
- Discrepancy analysis
- Improvement suggestions

## Tool Client

`ExternalToolClient` manages MCP server connections for tools like:
- Blender execution
- Scene investigation
- Asset generation
