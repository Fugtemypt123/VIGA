#!/usr/bin/env python3
"""HTML Executor MCP Server for Design2Code mode.

Handles HTML/CSS code execution and screenshot generation using
headless browser rendering.
"""

import base64
import io
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from mcp.server.fastmcp import FastMCP
from PIL import Image

# Tool configuration for the agent
tool_configs: List[Dict[str, object]] = [
    {
        "type": "function",
        "function": {
            "name": "execute_and_evaluate",
            "description": "Execute HTML/CSS code and trigger verifier evaluation.\nReturns either:\n  (1) On error: detailed error information; or \n  (2) On success: a screenshot of the rendered HTML page and further modification suggestions from a separate verifier agent.",
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
                        "description": "Provide the COMPLETE, UPDATED HTML/CSS code AFTER applying the edits listed in `code_diff`. The full code must include both the modified lines and the unchanged lines to ensure a coherent, runnable script."
                    }
                },
                "required": ["thought", "code_diff", "code"]
            }
        }
    }
]

# Create MCP instance
mcp = FastMCP("html-executor")

# Global executor instance
_executor: Optional["HTMLExecutor"] = None

class HTMLExecutor:
    """Executes HTML/CSS code and generates screenshots.

    This class manages HTML file creation, headless browser rendering,
    and screenshot optimization for the Design2Code pipeline.

    Attributes:
        output_dir: Directory path for saving HTML files and screenshots.
        browser_command: Command to invoke the headless browser.
        count: Counter for naming output files.
    """

    def __init__(self, output_dir: str, browser_command: str = "google-chrome") -> None:
        """Initialize the HTML executor.

        Args:
            output_dir: Directory path for output files.
            browser_command: Browser command to use for rendering.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.browser_command = browser_command
        self.count = 0

    def _save_html_file(self, html_code: str, filename: str = "index.html") -> str:
        """Save HTML code to a file.

        Args:
            html_code: The HTML content to save.
            filename: Name of the output file.

        Returns:
            The absolute path to the saved HTML file.
        """
        html_path = os.path.join(self.output_dir, filename)
        
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_code)
        
        return html_path
    
    def _take_screenshot(
        self,
        html_path: str,
        output_path: str,
        width: int = 1920,
        height: int = 1080
    ) -> Tuple[bool, str]:
        """Take a screenshot of the HTML page using headless browser.

        Args:
            html_path: Path to the HTML file to render.
            output_path: Path where the screenshot will be saved.
            width: Browser window width in pixels.
            height: Browser window height in pixels.

        Returns:
            Tuple of (success, message) where success indicates if the
            screenshot was taken successfully.
        """
        try:
            # Use Chrome/Chromium in headless mode to take screenshot
            cmd = [
                self.browser_command,
                "--headless",
                "--disable-gpu",
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--window-size={},{}".format(width, height),
                "--screenshot={}".format(output_path),
                "file://{}".format(os.path.abspath(html_path))
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0 and os.path.exists(output_path):
                return True, "Screenshot taken successfully"
            else:
                return False, f"Browser error: {result.stderr}"
                
        except subprocess.TimeoutExpired:
            return False, "Screenshot timeout"
        except FileNotFoundError:
            # Try alternative browser commands
            alternatives = ["chromium-browser", "chromium", "firefox", "google-chrome"]
            for alt_browser in alternatives:
                try:
                    cmd[0] = alt_browser
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                    if result.returncode == 0 and os.path.exists(output_path):
                        return True, f"Screenshot taken with {alt_browser}"
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    continue
            return False, f"Browser not found. Tried: {self.browser_command}, {', '.join(alternatives)}"
        except Exception as e:
            return False, f"Screenshot error: {str(e)}"
    
    def _optimize_image(self, image_path: str) -> str:
        """Optimize the screenshot image for smaller file size.

        Converts RGBA/LA/P mode images to RGB and resizes if too large.

        Args:
            image_path: Path to the image to optimize.

        Returns:
            Path to the optimized image file.
        """
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode in ('RGBA', 'LA', 'P'):
                    img = img.convert('RGB')
                
                # Resize if too large (max 1920x1080)
                max_width, max_height = 1920, 1080
                if img.width > max_width or img.height > max_height:
                    img.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
                
                # Save optimized version
                optimized_path = image_path.replace('.png', '_optimized.png')
                img.save(optimized_path, 'PNG', optimize=True, quality=85)
                
                return optimized_path
        except Exception as e:
            logging.warning(f"Image optimization failed: {e}")
            return image_path
    
    def execute(self, html_code: str) -> Dict[str, object]:
        """Execute HTML code and generate screenshot.

        Saves the HTML code, renders it in a headless browser, and
        captures a screenshot.

        Args:
            html_code: The complete HTML code to render.

        Returns:
            Dictionary with 'status' and 'output' containing either
            the screenshot path or error message.
        """
        try:
            self.count += 1
            html_path = self._save_html_file(html_code, f"{self.count}.html")
            output_path = self.output_dir / f"{self.count}.png"
            success, message = self._take_screenshot(html_path, str(output_path))
            if not success:
                return {"status": "error", "output": {"text": [message]}}
            optimized_path = self._optimize_image(str(output_path))
            return {"status": "success", "output": {"image": [optimized_path], "text": ["Screenshot taken successfully."], "require_verifier": True}}
        except Exception as e:
            logging.error(f"HTML execution failed: {e}")
            return {"status": "error", "output": {"text": [str(e)]}}

@mcp.tool()
def initialize(args: Dict[str, object]) -> Dict[str, object]:
    """Initialize the HTML executor.

    Args:
        args: Configuration dictionary with 'output_dir' and optional
            'browser_command' keys.

    Returns:
        Dictionary with status and tool configurations on success,
        or error message on failure.
    """
    global _executor
    try:
        _executor = HTMLExecutor(
            args.get("output_dir"),
            args.get("browser_command", "google-chrome")
        )
        return {
            "status": "success",
            "output": {
                "text": ["HTML executor initialized successfully."],
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
    """Execute HTML/CSS code and generate screenshot.

    Args:
        thought: Analysis of current state and plan for changes.
        code_diff: Code modifications in diff format.
        code: Complete HTML/CSS code to render.

    Returns:
        Dictionary with execution status and screenshot path or error.
    """
    global _executor
    if _executor is None:
        return {
            "status": "error",
            "output": {"text": ["HTML executor not initialized. Call initialize first."]}
        }
    try:
        result = _executor.execute(code)
        return result
    except Exception as e:
        return {"status": "error", "output": {"text": [str(e)]}}

def main() -> None:
    """Run the MCP server or execute test mode."""
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        args = {
            "output_dir": "output/test/design2code/",
            "browser_command": "google-chrome"
        }
        initialize(args)
        test_code = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Test Page</title>
</head>
<body>
    <h1>Hello World</h1>
</body>
</html>"""
        result = execute_and_evaluate(code=test_code)
        print("Result:", result)
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()