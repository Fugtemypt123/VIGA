# AgenticVerifier

MCP-based agent library for dual-agent (Generator/Verifier) interactive frameworks, supporting both 3D (Blender) and 2D (PPTX) modes. Plug and play for automated code generation, execution, and verification workflows.

## Overview

AgenticVerifier is a multi-agent system for iterative code generation and verification. It supports:
- 3D scene generation and validation using Blender
- 2D slide (PPTX) generation and validation
- Automated feedback loop between Generator and Verifier agents
- MCP stdio-based agent communication (no manual server setup required)
- **Seamless extensibility**: Can easily add any new server types
- **Multi-layer server chaining**: Support for complex server hierarchies and workflows

## System Architecture

### Overall Architecture

```
┌───────────────────────────────────────────────────────────────-──┐
│                    Dual-Agent Interactive System                 │
├───────────────────────────────────────────────────────────────-──┤
│                                                                  │
│                         ┌─────────────┐                          │
│                         │    Client   │                          │
│                         │ (Controller)│                          │
│                         └─────┬───────┘                          │
│                               │                                  │
│                               ▼                                  │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                      MCP Agent Layer                        │ │
│  │                                                             │ │
│  │         ┌─────────────┐              ┌───────────────┐      │ │
│  │         │  Generator  │◄────────────►│    Verifier   │      │ │
│  │         │    Agent    │              │     Agent     │      │ │
│  │         │   (MCP)     │              │     (MCP)     │      │ │
│  │         └─────────────┘              └───────────────┘      │ │
│  │         │             │               │             │       │ │
│  │         ▼             ▼               ▼             ▼       │ │
│  │  ┌─────────────┐ ┌──────────┐  ┌───────────┐  ┌───────────┐ │ │
│  │  │   Blender   │ │   pptx   │  │   Image   │  │   Scene   │ │ │
│  │  │   Server    │ │  Server  │  │   Server  │  │   Server  │ │ │
│  │  │   (MCP)     │ │  (MCP)   │  │   (MCP)   │  │   (MCP)   │ │ │
│  │  └─────────────┘ └──────────┘  └───────────┘  └───────────┘ │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────-────────────────────────────────────────┘
```

## Core Components

### 1. Generator Agent

The Generator Agent creates and iteratively improves code based on feedback from the Verifier Agent.

#### Key Features

- **Code Generation**: Generates code based on initial requirements and feedback
- **Memory Management**: Maintains conversation history and context
- **Automatic Execution**: Can automatically execute generated code (3D mode)
- **Feedback Integration**: Incorporates verifier feedback into next generation

#### Usage Example

```python
# Initialize generator
generator = GeneratorAgentClient("agents/generator_mcp.py")
await generator.connect()

# Create session
await generator.create_session(
    mode="blendergym",
    vision_model="gpt-4o",
    api_key=api_key,
    task_name="blendshape",
    max_rounds=10,
    init_code_path="data/blendergym/blendshape1/start.py",
    init_image_path="data/blendergym/blendshape1/renders/start",
    target_image_path="data/blendergym/blendshape1/renders/goal",
    thought_save="output/generator_thoughts.json",
    blender_server_path="servers/generator/blender.py",
    blender_command="utils/blender/infinigen/blender/blender",
    blender_file="data/blendergym/blendshape1/blender_file.blend",
    blender_script="data/blendergym/pipeline_render_script.py",
    render_save="output/renders",
    script_save="output/scripts"
)

# Generate code
result = await generator.generate_code()
code = result.get("code")

# Add feedback and generate again
await generator.add_feedback("Make the lighting more dramatic")
result = await generator.generate_code()
```

### 2. Verifier Agent

The Verifier Agent analyzes generated scenes and provides feedback for improvement.

#### Key Features

- **Visual Analysis**: Compares current scenes with target images
- **3D Scene Investigation**: Focuses camera, zooms, and moves around objects
- **Image Comparison**: Uses AI to identify visual differences
- **Feedback Generation**: Provides specific improvement suggestions

#### Available Tools

1. **Image Comparison Tool**
   - Compares two images and highlights differences
   - Provides detailed descriptions of visual changes
   - Uses AI vision model for analysis

2. **3D Scene Investigation Tool**
   - **Focus**: Set camera to track specific objects
   - **Zoom**: Adjust camera distance (in/out)
   - **Move**: Move camera around objects (up/down/left/right)

#### Usage Example

```python
# Initialize verifier
verifier = VerifierAgentClient("agents/verifier_mcp.py")
await verifier.connect()

# Create session with tool servers
await verifier.create_session(
    mode="blendergym",
    vision_model="gpt-4o",
    api_key=api_key,
    task_name="blendshape",
    max_rounds=10,
    target_image_path="data/blendergym/blendshape1/renders/goal",
    thought_save="output/verifier_thoughts",
    image_server_path="servers/verifier/image.py",
    scene_server_path=None
)

# Verify scene
result = await verifier.verify_scene(
    code=generated_code,
    render_path="output/renders",
    round_num=1
)

# Check result
if result.get("status") == "end":
    print("Scene matches target!")
elif result.get("status") == "continue":
    feedback = result["output"]
    print(f"Feedback: {feedback}")
```

## Quick Start

### Method 1: Using main.py (Single Instance)

The `main.py` runs a single instance with MCP stdio connections and automatically handles agent communication.

```bash
export OPENAI_API_KEY=your-openai-key

# For blendergym mode (3D Blender)
python main.py \
  --mode blendergym \
  --init-code-path data/blendergym/blendshape1/start.py \
  --target-image-path data/blendergym/blendshape1/renders/goal \
  --max-rounds 10 \
  --output-dir output

# For autopresent mode (2D PPTX)
python main.py \
  --mode autopresent \
  --init-code-path data/autopresent/start.py \
  --target-image-path data/autopresent/target \
  --max-rounds 10 \
  --output-dir output
```

**Available arguments:**
- `--mode`: Choose `blendergym`, `autopresent`, `blendergym-hard`, or `demo`
- `--init-code-path`: Path to the initial code file (**required**)
- `--target-image-path`: Directory of target images
- `--max-rounds`: Maximum number of interaction rounds (default: 10)
- `--output-dir`: Output directory (default: `output`)
- `--task-name`: Task name for hints extraction (default: `blendshape`)
- `--generator-script`: Path to generator MCP script (default: `agents/generator_mcp.py`)
- `--verifier-script`: Path to verifier MCP script (default: `agents/verifier_mcp.py`)

### Method 2: Using runners/ (Benchmark Testing)

The `runners/` directory contains scripts for testing all instances in a benchmark.

```bash
export OPENAI_API_KEY=your-openai-key

# Run autopresent benchmark (tests all autopresent instances)
python runners/autopresent.py

# Run blendergym benchmark (tests all blendergym instances)
python runners/blendergym.py
```

These runner scripts automatically iterate through all available instances in the benchmark and generate comprehensive evaluation results.

## Advanced Usage

### Custom Agent Development

You can extend the system by creating custom agents:

```python
from agents.generator_mcp import MCPGeneratorAgent
from agents.verifier_mcp import MCPVerifierAgent

# Custom generator with specialized logic
class CustomGenerator(MCPGeneratorAgent):
    async def generate_code(self, context):
        # Custom generation logic
        pass

# Custom verifier with specialized analysis
class CustomVerifier(MCPVerifierAgent):
    async def analyze_scene(self, scene_data):
        # Custom analysis logic
        pass
```

### Tool Server Integration

Add custom tool servers to extend functionality:

```python
from mcp.server.fastmcp import FastMCP

@mcp.tool()
async def custom_tool(param: str) -> dict:
    """Custom tool for specialized operations."""
    # Tool implementation
    return {"result": "success"}
```

## Notes

- **Recommended approach**: Use the new MCP-based `main.py` for easier setup and better resource management
- 3D mode requires Blender installed and available in your system PATH
- 2D PPTX mode requires `unoconv` and LibreOffice  
- The OpenAI API key must be set as the `OPENAI_API_KEY` environment variable
- Python 3.8+ is recommended
- Start with individual component testing before running the full system
- Agent communication is now handled via MCP stdio (no HTTP servers needed for agents)
- Executor servers (Blender, Slides) still use HTTP and need to be started separately
- Tool servers (Image, Scene) are automatically connected via MCP

## Contributing

Contributions are welcome! Please open issues or submit pull requests.