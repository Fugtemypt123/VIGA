# AgenticVerifier

MCP-based agent library for dual-agent (Generator/Verifier) interactive frameworks, supporting both 3D (Blender) and 2D (PPTX) modes. Plug and play for automated code generation, execution, and verification workflows.

## Overview

AgenticVerifier is a multi-agent system for iterative code generation and verification. It supports:
- 3D scene generation and validation using Blender
- 2D slide (PPTX) generation and validation
- Automated feedback loop between Generator and Verifier agents
- Extensible agent and executor server architecture

## Requirements

- Python >= 3.8
- [Blender](https://www.blender.org/) (required for 3D mode, must be available in PATH)
- [unoconv](https://github.com/unoconv/unoconv) (required for PPTX-to-image conversion in 2D mode, needs LibreOffice)
- OpenAI API Key (set as environment variable `OPENAI_API_KEY`)
- Linux is recommended

### Python Dependencies

Install the core dependencies:

```bash
pip install requests pillow numpy openai
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

- `main.py`: Main entry point for the dual-agent interactive loop
- `agents/`: Core logic for Generator and Verifier agents
- `servers/`: Executor and agent server implementations (Blender, Slides, Verifier, etc.)
- `examples/`: Example scripts for usage and testing
- `utils/`, `data/`: Utilities and data resources

## Quick Start

### 1. Start All Required Servers

In separate terminals (or in the background), start the following servers:

```bash
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
```

### 2. Run the Main Dual-Agent Loop

**For 3D (Blender) mode:**

```bash
export OPENAI_API_KEY=your-openai-key

python main.py \
  --mode 3d \
  --init-code path/to/init.py \
  --target-image-path path/to/target/images \
  --max-rounds 10
```

**For 2D (PPTX) mode:**

```bash
python main.py \
  --mode 2d \
  --init-code path/to/init.py \
  --target-image-path path/to/target/images \
  --max-rounds 10
```

**Common arguments:**
- `--init-code`: Path to the initial code file
- `--target-image-path`: Directory of target images
- `--max-rounds`: Maximum number of interaction rounds
- `--generator-hints` / `--verifier-hints`: (Optional) Hints for the agents
- `--render-save` / `--code-save`: Output directories for renders or code

### 3. Example Scripts

See the `examples/` directory for ready-to-run scripts:

**Generator example:**
```bash
python examples/generator_mcp_usage.py --mode blender --init-code path/to/init.py --api-key your-openai-key
```

**Verifier example:**
```bash
python examples/verifier_mcp_usage.py --api-key your-openai-key --target-image-path path/to/target/images
```

## Notes

- 3D mode requires Blender installed and available in your system PATH
- 2D PPTX mode requires `unoconv` and LibreOffice
- The OpenAI API key must be set as the `OPENAI_API_KEY` environment variable
- Python 3.8+ is recommended

## Contributing

Contributions are welcome! Please open issues or submit pull requests.

---

For more details, see the code and comments in `main.py` and the `examples/` directory.
