# AgenticVerifier

MCP-based agent library for dual-agent (Generator/Verifier) interactive frameworks, supporting 3D (Blender), 2D (PPTX), and Design2Code (HTML/CSS) modes. Plug and play for automated code generation, execution, and verification workflows.

## Overview

AgenticVerifier is a multi-agent system for iterative code generation and verification. It supports:
- 3D scene generation and validation using Blender
- 2D slide (PPTX) generation and validation
- **Design2Code**: Convert visual designs to HTML/CSS code
- **Static Scene**: Build 3D static scenes from scratch
- **Dynamic Scene**: Build 3D animated/rigged scenes from scratch
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
│  │  ┌─────────────┐ ┌──────────┐                               │ │
│  │  │    HTML     │ │   Web    │                               │ │
│  │  │   Server    │ │  Server  │                               │ │
│  │  │   (MCP)     │ │  (MCP)   │                               │ │
│  │  └─────────────┘ └──────────┘                               │ │
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

### 3. Dual-Agent Interaction

The dual-agent system creates an iterative feedback loop between generation and verification.

#### Workflow

1. **Initialization**: Set up generator and verifier sessions with target images and initial code
2. **Generation**: Generator creates code based on requirements and feedback
3. **Execution**: Code is automatically executed (3D mode) or manually executed
4. **Verification**: Verifier compares current scene with target images and generates feedback
5. **Iteration**: Feedback is passed back to generator for next iteration
6. **Completion**: Process continues until success or max rounds reached

#### Key Features

- **Automated Feedback Loop**: Seamless integration between generation and verification
- **Multi-Round Iteration**: Supports up to 10 rounds of improvement
- **Automatic Execution**: 3D mode includes automatic Blender execution
- **Visual Analysis**: AI-powered image comparison and scene investigation
- **Context Preservation**: Maintains conversation history across rounds

## Quick Start

### API Keys Setup

1. Prepare your API keys and configure them in `runners/api_keys.py`:
   - [Meshy API](https://www.meshy.ai/discover) - for 3D asset generation
   - [VA API](https://va.landing.ai/demo/agentic-od) - for object detection
   - [OpenAI API](https://platform.openai.com/settings/organization/api-keys) - for AI models

```bash
# Option 1: Set environment variables (fallback)
export MESHY_API_KEY="your_meshy_api_key"
export VA_API_KEY="your_va_api_key"
export OPENAI_API_KEY="your_openai_api_key"

# Option 2: Edit runners/api_keys.py directly (recommended)
# Edit the values in the api_keys.py file
```

### Build from Scratch (Image-to-Scene)

2. Run the demo script:
```bash
python runners/demo.py --target-image-path=your_initial_image_path
```

To run the test on benchmarks (BlenderGym, AutoPresent, Design2Code, Static Scene, Dynamic Scene), follow the instruction below:

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

# For design2code mode (HTML/CSS generation)
python main.py \
  --mode design2code \
  --init-image-path data/design2code/example/design.png \
  --target-description data/design2code/example/description.txt \
  --max-rounds 5 \
  --output-dir output
```

**Available arguments:**
- `--mode`: Choose `blendergym`, `autopresent`, `blendergym-hard`, `demo`, `design2code`, `static_scene`, or `dynamic_scene`
- `--init-code-path`: Path to the initial code file (**required for blendergym/autopresent**)
- `--init-image-path`: Path to design screenshot (**required for design2code**)
- `--target-image-path`: Directory of target images or single image file
- `--target-description`: Path to description file or direct description text
- `--max-rounds`: Maximum number of interaction rounds (default: 10)
- `--output-dir`: Output directory (default: `output`)
- `--task-name`: Task name for hints extraction (default: `blendshape`)
- `--generator-script`: Path to generator MCP script (default: `agents/generator_mcp.py`)
- `--verifier-script`: Path to verifier MCP script (default: `agents/verifier_mcp.py`)

### Method 2: Using runners/ (Benchmark Testing)

The `runners/` directory contains scripts for testing all instances in a benchmark.

```bash
export OPENAI_API_KEY=your-openai-key
export OPENAI_BASE_URL=your-openai-url # https://api.openai.com/v1

# Run autopresent benchmark (tests all autopresent instances)
python runners/autopresent.py

# Run blendergym benchmark (tests all blendergym instances)
python runners/blendergym.py

# Run blendergym-hard benchmark (tests complex 3D scene reconstruction)
python runners/blendergym_hard.py

# Run design2code benchmark (tests HTML/CSS generation)
python runners/design2code.py

# Run static scene benchmark (tests 3D static scene generation from scratch)
python runners/static_scene.py

# Run dynamic scene benchmark (tests 3D animated scene generation from scratch)
python runners/dynamic_scene.py
```

These runner scripts automatically iterate through all available instances in the benchmark and generate comprehensive evaluation results.

## Evaluation

After running benchmarks, you can evaluate the results using the evaluation scripts.

### Step 1: Run Evaluation

```bash
# For blendergym results
python evaluators/blendergym/evaluate.py 20250818_101357

# For autopresent results  
python evaluators/autopresent/evaluate.py 20250818_101357
```

The test ID (e.g., `20250818_101357`) is automatically generated when running the benchmark and appears in the `output/blendergym/` or `output/autopresent/` directory.

([NOTE](): You must run the evaluation before rerun the tests)

### Step 2: Gather Results

```bash
# For blendergym results
python evaluators/blendergym/gather.py 20250817_035024

# For autopresent results
python evaluators/autopresent/gather.py 20250817_035024
```

The gather script processes the intermediate evaluation results and generates a comprehensive summary.

## Supported Modes

### 1. BlenderGym Mode
3D scene modification and enhancement based on target images using Blender.

### 2. AutoPresent Mode  
2D slide (PPTX) generation and validation with automated feedback.

### 3. BlenderGym-Hard Mode
Complex 3D scene reconstruction with advanced tools like Meshy integration and scene investigation.

### 4. Design2Code Mode

The Design2Code mode enables the dual-agent system to convert visual designs (screenshots) into clean, semantic HTML and CSS code. This implementation is inspired by the [Design2Code benchmark](https://github.com/NoviScl/Design2Code) and provides a complete pipeline for:

1. **Generator Agent**: Analyzes design screenshots and generates HTML/CSS code
2. **Verifier Agent**: Compares generated webpages with target designs and provides feedback
3. **HTML Execution**: Renders HTML code and generates screenshots for comparison
4. **Visual Comparison**: Uses AI vision models to compare designs and provide detailed feedback

### Features

#### Generator Agent Features
- **Design Analysis**: Analyzes design screenshots to understand layout, colors, typography
- **HTML Generation**: Creates semantic, accessible HTML5 code
- **CSS Styling**: Generates modern CSS with responsive design
- **Best Practices**: Follows web standards and accessibility guidelines

#### Verifier Agent Features
- **Visual Comparison**: Compares generated webpages with target designs
- **Layout Analysis**: Checks positioning, spacing, and structure
- **Code Quality**: Analyzes HTML structure and CSS organization
- **Accessibility**: Verifies semantic structure and accessibility features

#### HTML Execution Server
- **Screenshot Generation**: Uses headless browser to capture webpage screenshots
- **Image Optimization**: Optimizes screenshots for comparison
- **Multi-browser Support**: Supports Chrome, Chromium, Firefox
- **Error Handling**: Robust error handling and fallback options

#### Web Comparison Server
- **AI-powered Comparison**: Uses vision models for detailed design comparison
- **Difference Highlighting**: Highlights visual differences between designs
- **Similarity Scoring**: Provides quantitative similarity scores
- **Detailed Feedback**: Generates actionable feedback for improvements

### Usage

```bash
# Basic usage
python main.py \
    --mode design2code \
    --init-image-path data/design2code/example/design.png \
    --target-description data/design2code/example/description.txt \
    --output-dir output/design2code_test \
    --max-rounds 5

# With custom browser
python main.py \
    --mode design2code \
    --init-image-path path/to/design.png \
    --target-description "Create a modern landing page" \
    --browser-command chromium-browser \
    --max-rounds 3
```

### Test Script

Use the provided test script to quickly test the functionality:

```bash
python test_design2code.py
```

This script will:
1. Create a sample design image if needed
2. Set up test parameters
3. Run the Design2Code pipeline
4. Generate HTML files and screenshots

### Requirements

- **Browser**: Chrome/Chromium for HTML screenshots
- **API Key**: OpenAI API key for vision models
- **Dependencies**: PIL, numpy, requests

### Browser Setup

Install Chrome on Ubuntu:
```bash
wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | sudo apt-key add -
echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" | sudo tee /etc/apt/sources.list.d/google-chrome.list
sudo apt update
sudo apt install google-chrome-stable
```

### Example Workflow

1. **Input**: Design screenshot + description
2. **Generation**: Generator creates HTML/CSS code
3. **Execution**: HTML server renders code and takes screenshot
4. **Verification**: Verifier compares with target design
5. **Feedback**: Detailed feedback for improvements
6. **Iteration**: Process repeats until satisfactory result

### 5. Static Scene Mode

Build 3D static scenes from scratch based on target images or descriptions.

**Features:**
- Generate 3D scenes from single target images
- Support for scene descriptions
- Automatic object placement and lighting
- Scene investigation and feedback

**Usage:**
```bash
python main.py \
    --mode static_scene \
    --target-image-path data/static_scene/example/target.png \
    --target-description "A modern living room with sofa and coffee table" \
    --output-dir output/static_scene_test \
    --max-rounds 10
```

### 6. Dynamic Scene Mode

Build 3D animated/rigged scenes from scratch with temporal dynamics.

**Features:**
- Generate animated 3D scenes from target videos/animations
- Support for character rigging and animation
- Physics simulations and temporal coherence
- Advanced animation analysis and feedback

**Usage:**
```bash
python main.py \
    --mode dynamic_scene \
    --target-image-path data/dynamic_scene/example/target.mp4 \
    --target-description "A walking character animation with physics" \
    --output-dir output/dynamic_scene_test \
    --max-rounds 15
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

## Project Structure

```
AgenticVerifier/
├── agents/                 # Agent implementations
│   ├── config_manager.py   # Configuration management
│   ├── generator.py        # Generator agent
│   ├── verifier.py         # Verifier agent
│   ├── prompt_builder.py   # Prompt building system
│   └── ...
├── runners/                # Benchmark runners
│   ├── api_keys.py         # API key configuration
│   ├── blendergym.py       # BlenderGym benchmark
│   ├── autopresent.py      # AutoPresent benchmark
│   ├── blendergym_hard.py  # BlenderGym-Hard benchmark
│   ├── design2code.py      # Design2Code benchmark
│   ├── static_scene.py     # Static Scene benchmark
│   ├── dynamic_scene.py    # Dynamic Scene benchmark
│   └── ...
├── prompts/                # Prompt templates
│   ├── blendergym/         # BlenderGym prompts
│   ├── autopresent/        # AutoPresent prompts
│   ├── blendergym_hard/    # BlenderGym-Hard prompts
│   ├── design2code/        # Design2Code prompts
│   ├── static_scene/       # Static Scene prompts
│   ├── dynamic_scene/      # Dynamic Scene prompts
│   └── prompt_manager.py   # Centralized prompt management
├── tools/                  # Tool implementations
├── evaluators/             # Evaluation scripts
├── data/                   # Benchmark datasets
├── output/                 # Generated results
└── main.py                 # Main entry point
```

## Notes

- **API Keys**: Configure in `runners/api_keys.py` or set environment variables
- **Recommended approach**: Use the new MCP-based `main.py` for easier setup and better resource management
- 3D modes require Blender installed and available in your system PATH
- 2D PPTX mode requires `unoconv` and LibreOffice  
- **Design2Code mode requires Chrome/Chromium browser for HTML screenshots**
- Python 3.8+ is recommended
- Start with individual component testing before running the full system
- Agent communication is now handled via MCP stdio (no HTTP servers needed for agents)
- Executor servers (Blender, Slides, HTML) still use HTTP and need to be started separately
- Tool servers (Image, Scene, Web) are automatically connected via MCP

## Contributing

Contributions are welcome! Please open issues or submit pull requests.