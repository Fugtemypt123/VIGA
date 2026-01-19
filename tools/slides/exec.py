"""Slides Executor MCP Server for PowerPoint generation.

Handles Python code execution for slide generation and screenshot capture
using unoconv for PPTX to image conversion.
"""

import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

from mcp.server.fastmcp import FastMCP

# Tool configuration for the agent
tool_configs: List[Dict[str, object]] = [
    {
        "type": "function",
        "function": {
            "name": "execute_and_evaluate",
            "description": "Execute Python code for slide generation and trigger verifier evaluation.\nReturns either:\n  (1) On error: detailed error information; or \n  (2) On success: a rendered slide image and further modification suggestions from a separate verifier agent. You should follow the verifier's suggestions to refine the slide.",
            "parameters": {
                "type": "object",
                "properties": {
                    "thought": {
                        "type": "string",
                        "description": "Think step by step about the current state and reason about what code to write next. Describe your reasoning process clearly."
                    },
                    "code_diff": {
                        "type": "string",
                        "description": "Before outputting the final code, precisely list the line-level edits you will make. Use this minimal diff-like format ONLY:\n\n-: [lines to remove]\n+: [lines to add]\n\nRules:\n1) Show only the smallest necessary edits (avoid unrelated changes).\n2) Keep ordering: list removals first, then additions.\n3) Do not include commentary here—only the edit blocks.\n4) If starting from scratch, use `-: []` and put all new lines under `+: [...]`.\n5) Every line is a literal code line (no markdown, no fences)."
                    },
                    "code": {
                        "type": "string",
                        "description": "Provide the COMPLETE, UPDATED Python code AFTER applying the edits listed in `code_diff`. The full code must include both the modified lines and the unchanged lines to ensure a coherent, runnable script."
                    }
                },
                "required": ["thought", "code_diff", "code"]
            }
        }
    }
]

# Create MCP instance
mcp = FastMCP("slides-executor")

# Global executor instance
_executor: Optional["SlidesExecutor"] = None


class SlidesExecutor:
    """Executes Python code to generate PowerPoint slides.

    This class manages Python code execution for slide generation,
    converting PPTX files to images using unoconv.

    Attributes:
        task_dir: Directory containing task resources.
        output_dir: Directory for output files.
        count: Counter for naming output directories.
    """

    def __init__(self, resource_dir: str, output_dir: str) -> None:
        """Initialize the slides executor.

        Args:
            resource_dir: Path to the resource directory.
            output_dir: Path for output files.
        """
        self.task_dir = Path(resource_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.count = 0

    def _execute_slide_code(self, code_path: str) -> str:
        """Execute Python code to generate a PPTX and convert to image.

        Args:
            code_path: Path to the Python script to execute.

        Returns:
            "Success" on success, or error message on failure.
        """
        generate_dir = "utils/slides"
        env = os.environ.copy()
        env['PYTHONPATH'] = f"{generate_dir}:{env.get('PYTHONPATH', '')}"
        try:
            subprocess.run(
                ["python", code_path],
                capture_output=True,
                text=True,
                check=True,
                env=env
            )
            pptx_file = code_path.replace("runned_code.py", "refine.pptx")
            subprocess.run(
                ["/usr/bin/python3", "/usr/bin/unoconv", "-f", "jpg", pptx_file],
                check=True
            )
            return "Success"
        except subprocess.CalledProcessError as e:
            logging.error(f"PPTX compilation failed: {e.stderr}")
            return f"Error: {e.stderr}"

    def execute(self, code: str) -> Dict[str, object]:
        """Execute Python code and generate slide image.

        Saves the code, modifies output paths, executes it, and converts
        the resulting PPTX to an image.

        Args:
            code: The complete Python code to execute.

        Returns:
            Dictionary with 'status' and 'output' containing either
            the image path or error message.
        """
        try:
            self.count += 1
            round_dir = self.output_dir / f"{self.count}"
            round_dir.mkdir(exist_ok=True)
            code_path = round_dir / "refine.py"
            runned_code_path = round_dir / "runned_code.py"
            slide_path = code_path.with_suffix(".pptx")
            image_path = code_path.with_suffix(".jpg")

            with open(code_path, "w") as f:
                f.write(code)

            # Replace hardcoded .save("xx.pptx") with dynamic save path
            code = re.sub(r'output\.pptx', f'{slide_path}', code)

            with open(runned_code_path, "w") as f:
                f.write(code)

            result = self._execute_slide_code(str(runned_code_path))

            if result == "Success" and image_path.exists():
                return {
                    "status": "success",
                    "output": {
                        "image": [str(image_path)],
                        "text": ["Code executed successfully, image generated."],
                        "require_verifier": True
                    }
                }
            else:
                return {"status": "error", "output": {"text": [result]}}
        except Exception as e:
            return {"status": "error", "output": {"text": [str(e)]}}

@mcp.tool()
def initialize(args: Dict[str, object]) -> Dict[str, object]:
    """Initialize the Slides executor.

    Args:
        args: Configuration dictionary with 'resource_dir' and 'output_dir' keys.

    Returns:
        Dictionary with status and tool configurations on success,
        or error message on failure.
    """
    global _executor
    try:
        _executor = SlidesExecutor(
            args.get("resource_dir"),
            args.get("output_dir")
        )
        return {
            "status": "success",
            "output": {
                "text": ["Slides executor initialized successfully"],
                "tool_configs": tool_configs
            }
        }
    except Exception as e:
        return {"status": "error", "output": {"text": [str(e)]}}

@mcp.tool()
def execute_and_evaluate(
    thought: str = '',
    code_diff: str = '',
    code: str = ''
) -> Dict[str, object]:
    """Execute Python code to generate and render a PPTX slide.

    Note: thought and code_diff are part of the tool API contract for the
    model but are not used in execution.

    Args:
        thought: Analysis of current state and plan for changes.
        code_diff: Code modifications in diff format.
        code: Complete Python code that generates a .pptx file.

    Returns:
        Dictionary with execution status and image path or error.
    """
    # thought and code_diff are part of the API contract but unused here
    _ = thought, code_diff

    global _executor
    if _executor is None:
        return {
            "status": "error",
            "output": {"text": ["Executor not initialized. Call initialize first."]}
        }
    try:
        result = _executor.execute(code)
        return result
    except Exception as e:
        return {"status": "error", "output": {"text": [str(e)]}}

def test_specific_file() -> bool:
    """Test the executor with a specific test file.

    Returns:
        True if the test passes, False otherwise.
    """
    test_file_path = "output/test/1/runned_code.py"

    print(f"Starting test file: {test_file_path}")
    print("=" * 60)

    if not os.path.exists(test_file_path):
        print(f"File does not exist: {test_file_path}")
        return False

    print(f"File exists: {test_file_path}")

    try:
        with open(test_file_path, 'r', encoding='utf-8') as f:
            code_content = f.read()
        print(f"File read successfully, code length: {len(code_content)} characters")
    except Exception as e:
        print(f"File read failed: {e}")
        return False

    try:
        task_dir = "data/autopresent/examples/art_photos/slide_1"
        temp_output_dir = "output/test"
        executor = SlidesExecutor(task_dir, temp_output_dir)
        print(f"Executor initialized, output directory: {temp_output_dir}")
    except Exception as e:
        print(f"Executor initialization failed: {e}")
        return False

    try:
        print("Executing code...")
        result = executor.execute(code_content)

        if result["status"] == "success":
            print("Code executed successfully!")
            print(f"  Generated image path: {result.get('output', 'N/A')}")
            return True
        else:
            print(f"Code execution failed: {result.get('output', 'Unknown error')}")
            return False

    except Exception as e:
        print(f"Exception during execution: {e}")
        return False


def main() -> None:
    """Run the MCP server or execute test mode."""
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        success = test_specific_file()
        sys.exit(0 if success else 1)
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
