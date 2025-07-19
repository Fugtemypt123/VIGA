# AgenticVerifier

MCP-based agent library for dual-agent (Generator/Verifier) interactive frameworks, supporting both 3D (Blender) and 2D (PPTX) modes. Plug and play for automated code generation, execution, and verification workflows.

## Overview

AgenticVerifier is a multi-agent system for iterative code generation and verification. It supports:
- 3D scene generation and validation using Blender
- 2D slide (PPTX) generation and validation
- Automated feedback loop between Generator and Verifier agents
- MCP stdio-based agent communication (no manual server setup required)
- Extensible agent and executor server architecture

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
│  │     ┌─────────────┐                  ┌───────────────┐      │ │
│  │     │  Generator  │◄────────────────►│    Verifier   │      │ │
│  │     │    Agent    │                  │     Agent     │      │ │
│  │     │   (MCP)     │                  │     (MCP)     │      │ │
│  │     └─────┬───────┘                  └───────────────┘      │ │
│  │           │                            │           │        │ │
│  │           ▼                            ▼           ▼        │ │
│  │   ┌─────────────┐          ┌─────────────┐  ┌─────────────┐ │ │
│  │   │   Blender   │          │    Image    │  │   Scene     │ │ │
│  │   │   Server    │          │   Server    │  │   Server    │ │ │
│  │   │   (MCP)     │          │   (MCP)     │  │   (MCP)     │ │ │
│  │   └─────────────┘          └─────────────┘  └─────────────┘ │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────-────────────────────────────────────────┘
```

## Core Components

### 1. Generator Agent

The Generator Agent is responsible for creating and iteratively improving code based on feedback from the Verifier Agent.

#### Key Features

- **Code Generation**: Generates code based on initial requirements and feedback
- **Memory Management**: Maintains conversation history and context
- **Automatic Execution**: Can automatically execute generated code (3D mode)
- **Feedback Integration**: Incorporates verifier feedback into next generation
- **Session Management**: Handles multiple generation sessions

#### Workflow

1. **Initialization**: Set up with target images, hints, and initial code
2. **Code Generation**: Generate code using vision model and context
3. **Execution**: Execute code (automatic or manual)
4. **Feedback Processing**: Receive feedback from verifier
5. **Iteration**: Generate improved code based on feedback

#### Usage Example

```python
# Initialize generator
generator = GeneratorAgentClient("agents/generator_mcp.py")
await generator.connect()

# Create session
await generator.create_session(
    vision_model="gpt-4o",
    api_key=api_key,
    thoughtprocess_save="generator_thought.json",
    max_rounds=10,
    generator_hints="Focus on realistic lighting",
    init_code=initial_code,
    target_image_path="target_images",
    # Blender-specific parameters for 3D mode
    blender_command="blender",
    blender_file="scene.blend",
    render_save="renders"
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
- **Tool Integration**: Uses specialized tools for detailed analysis

#### Available Tools

1. **Image Comparison Tool**
   - Compares two images and highlights differences
   - Provides detailed descriptions of visual changes
   - Uses AI vision model for analysis

2. **3D Scene Investigation Tool**
   - **Focus**: Set camera to track specific objects
   - **Zoom**: Adjust camera distance (in/out)
   - **Move**: Move camera around objects (up/down/left/right)

#### Workflow

1. **Session Creation**: Initialize with target images and hints
2. **Scene Analysis**: Analyze current scene against target
3. **Tool Usage**: Use specialized tools for detailed investigation
4. **Feedback Generation**: Generate improvement suggestions
5. **Status Determination**: Decide if scene matches target

#### Usage Example

```python
# Initialize verifier
verifier = VerifierAgentClient("agents/verifier_mcp.py")
await verifier.connect()

# Create session with tool servers
await verifier.create_session(
    vision_model="gpt-4o",
    api_key=api_key,
    thoughtprocess_save="verifier_thought.json",
    max_rounds=10,
    verifier_hints="Check for proper object placement",
    target_image_path="target_images",
    image_server_path="servers/verifier/image.py",
    scene_server_path="servers/verifier/scene.py"
)

# Verify scene
result = await verifier.verify_scene(
    code=generated_code,
    render_path="renders",
    round_num=1
)

# Check result
if result.get("status") == "end":
    print("Scene matches target!")
elif result.get("status") == "continue":
    feedback = result["output"]
    print(f"Feedback: {feedback}")
```

### 3. Dual-Agent Interaction

The dual-agent system creates an iterative feedback loop between generation and verification.

#### Detailed Process

1. **Initialization Phase**
   - Load initial code and target images
   - Set up generator and verifier sessions
   - Configure tool servers and executors

2. **Generation Phase**
   - Generator creates code based on requirements and feedback
   - Code is automatically executed (3D mode) or manually executed
   - Rendered images are saved for analysis

3. **Verification Phase**
   - Verifier compares current scene with target images
   - Uses specialized tools for detailed analysis
   - Generates specific feedback for improvements

4. **Feedback Integration**
   - Feedback is passed back to generator
   - Generator incorporates feedback into next iteration
   - Process continues until success or max rounds reached

## Requirements

- Python >= 3.8
- [Blender](https://www.blender.org/) (required for 3D mode, must be available in PATH)
- [unoconv](https://github.com/unoconv/unoconv) (required for PPTX-to-image conversion in 2D mode, needs LibreOffice)
- OpenAI API Key (set as environment variable `OPENAI_API_KEY`)
- Linux is recommended

### Python Dependencies

Install the core dependencies:

```bash
pip install requests pillow numpy openai "mcp[cli]"
```

For 3D (Blender/Infinigen) workflows, also install:

```bash
pip install opencv-python matplotlib scipy networkx tqdm
```

For PPTX (slides) workflows, also install:

```bash
pip install python-pptx
```

## Directory Structure

- `main.py`: Main entry point for the dual-agent interactive loop (MCP-based, no server setup required)
- `main_http.py`: Alternative HTTP-based main entry (requires manual server setup)
- `agents/`: Core logic for Generator and Verifier agents
- `servers/`: Executor and agent server implementations (Blender, Slides, Verifier, etc.)
- `examples/`: Example scripts for usage and testing
- `test_generator.py`: Standalone test script for Generator Agent
- `test_verifier.py`: Standalone test script for Verifier Agent
- `utils/`, `data/`: Utilities and data resources

## Quick Start

### Recommended: Using New MCP-Based Main Script

The new `main.py` uses MCP stdio connections and automatically handles agent communication without requiring manual server setup.

```bash
export OPENAI_API_KEY=your-openai-key

# For 3D (Blender) mode
python main.py \
  --mode 3d \
  --init-code path/to/init.py \
  --target-image-path path/to/target/images \
  --max-rounds 10

# For 2D (PPTX) mode  
python main.py \
  --mode 2d \
  --init-code path/to/init.py \
  --target-image-path path/to/target/images \
  --max-rounds 10
```

**Key advantages of the new approach:**
- ✅ **No manual server setup required** - agents connect automatically via MCP stdio
- ✅ **Simplified deployment** - single command execution
- ✅ **Better resource management** - automatic cleanup on exit
- ✅ **Cleaner process management** - no orphaned server processes

**Available arguments:**
- `--mode`: Choose `3d` (Blender) or `2d` (PPTX) mode
- `--init-code`: Path to the initial code file (**required**)
- `--target-image-path`: Directory of target images
- `--max-rounds`: Maximum number of interaction rounds (default: 10)
- `--generator-script`: Path to generator MCP script (default: `agents/generator_mcp.py`)
- `--verifier-script`: Path to verifier MCP script (default: `agents/verifier_mcp.py`)
- `--generator-hints` / `--verifier-hints`: Optional hints for the agents
- `--render-save`: Render save directory (default: `renders`)
- `--code-save`: Slides code save directory for 2D mode (default: `slides_code`)
- `--blender-url` / `--slides-url`: Executor server URLs (still HTTP-based)

**Example usage:**

```bash
# 3D Blender workflow with custom parameters
python main.py \
  --mode 3d \
  --init-code examples/blender_init.py \
  --target-image-path data/target_renders \
  --generator-hints "Focus on realistic lighting and materials" \
  --verifier-hints "Check for proper object placement and lighting" \
  --max-rounds 15 \
  --render-save output/renders

# 2D PPTX workflow
python main.py \
  --mode 2d \
  --init-code examples/slides_init.py \
  --target-image-path data/target_slides \
  --generator-hints "Create professional slide layouts" \
  --verifier-hints "Ensure text readability and visual consistency" \
  --max-rounds 8 \
  --code-save output/slides
```

### Individual Component Testing

Before running the full dual-agent system, you can test each component individually:

#### Test Generator Agent

```bash
export OPENAI_API_KEY=your-openai-key

python test_generator.py \
  --init-code "print('Hello, World!')" \
  --generator-hints "Generate clean, well-commented code"
```

**Available options:**
- `--generator-script`: Path to generator MCP script (default: `agents/generator_mcp.py`)
- `--vision-model`: OpenAI vision model (default: `gpt-4o`)
- `--thoughtprocess-save`: Path to save thought process (default: `test_generator_thought.json`)
- `--max-rounds`: Maximum rounds (default: 5)
- `--generator-hints`: Hints for generator
- `--init-code`: Initial code to work with
- `--init-image-path`: Path to initial images
- `--target-image-path`: Path to target images
- `--target-description`: Target description

#### Test Verifier Agent

```bash
export OPENAI_API_KEY=your-openai-key

python test_verifier.py \
  --verifier-hints "Focus on identifying key visual differences"
```

**Available options:**
- `--verifier-script`: Path to verifier MCP script (default: `agents/verifier_mcp.py`)
- `--vision-model`: OpenAI vision model (default: `gpt-4o`)
- `--thoughtprocess-save`: Path to save thought process (default: `test_verifier_thought.json`)
- `--max-rounds`: Maximum rounds (default: 3)
- `--verifier-hints`: Hints for verifier
- `--blender-save`: Blender save path

The verifier test will automatically create test images to demonstrate the verification functionality.

### Alternative: Manual Server Setup (Legacy)

If you need more control or want to use the legacy HTTP-based approach, you can use `main_http.py` with manual server setup:

```bash
# In separate terminals:

# Generator Agent server
python agents/generator_mcp.py

# Verifier Agent server  
python agents/verifier_mcp.py

# Blender Executor server (for 3D mode)
python servers/generator/blender.py

# Slides Executor server (for 2D mode)
python servers/generator/slides.py

# Verifier image/scene servers
python servers/verifier/image.py
python servers/verifier/scene.py

# Then run the main script
python main_http.py --mode 3d --init-code path/to/init.py
```

### Example Scripts

See the `examples/` directory for ready-to-run scripts:

**Generator example:**
```bash
python examples/generator_mcp_usage.py --mode blender --init-code path/to/init.py
```

**Verifier example:**
```bash
python examples/verifier_mcp_usage.py --target-image-path path/to/target/images
```

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

### Session Management

The system supports multiple concurrent sessions:

```python
# Create multiple sessions
session1 = await verifier.create_session(...)
session2 = await verifier.create_session(...)

# Work with different sessions
await verifier.verify_scene(session_id=session1, ...)
await verifier.verify_scene(session_id=session2, ...)
```

## Troubleshooting

### Common Issues

1. **Import Error for MCP**: Make sure you have installed `"mcp[cli]"`:
   ```bash
   pip install "mcp[cli]"
   ```

2. **OpenAI API Key**: Ensure your API key is set:
   ```bash
   export OPENAI_API_KEY=your-openai-key
   ```

3. **Missing Dependencies**: Install all required packages:
   ```bash
   pip install requests pillow numpy openai "mcp[cli]"
   ```

4. **Blender Not Found**: For 3D mode, ensure Blender is installed and available in PATH.

5. **Agent Connection Issues**: The new MCP-based approach should handle connections automatically. If you encounter issues, check that the agent script paths are correct:
   ```bash
   ls agents/generator_mcp.py agents/verifier_mcp.py
   ```

6. **Executor Server Issues**: Note that Blender and Slides executors still require separate HTTP servers. Make sure they're running before starting the main script.

7. **Tool Server Connection Issues**: If verifier tools aren't working, check that the tool server paths are correct:
   ```bash
   ls servers/verifier/image.py servers/verifier/scene.py
   ```

### Debug Mode

Enable debug logging for troubleshooting:

```bash
export PYTHONPATH=.
python -u main.py --mode 3d --init-code path/to/init.py 2>&1 | tee debug.log
```

### Performance Optimization

- Use `--max-rounds` to limit iterations
- Set appropriate `--vision-model` for your use case
- Monitor API usage and costs
- Use local caching for repeated operations

## Migration from HTTP to MCP

If you're upgrading from the old HTTP-based approach:

1. **Use the new `main.py`** instead of manually starting agent servers
2. **Agent servers are now auto-started** via MCP stdio connections
3. **Executor servers still need manual setup** (Blender, Slides, etc.)
4. **Arguments have changed**: `--generator-url` and `--verifier-url` are replaced with `--generator-script` and `--verifier-script`

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

---

For more details, see the code and comments in `main.py` and the `examples/` directory.