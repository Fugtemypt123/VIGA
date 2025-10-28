#!/usr/bin/env python3
"""
HTML execution server for Design2Code mode.
Handles HTML/CSS code execution and screenshot generation.
"""
import os
import tempfile
import subprocess
import base64
import io
from pathlib import Path
from typing import Dict, Tuple, Optional
from PIL import Image
import logging
from mcp.server.fastmcp import FastMCP

# tool config for agent
tool_configs = [
    {
        "type": "function",
        "function": {
            "name": "execute_and_evaluate",
            "description": "Execute HTML/CSS code and trigger verifier evaluation. This tool combines code execution with automatic verification. Always use this tool when you want to execute your code changes.\nReturns either:\n  (1) On error: detailed error information; or \n  (2) On success: a screenshot of the rendered HTML page and further modification suggestions from a separate verifier agent.\nImportant: The execution environment cannot read files generated in previous runs. Therefore, each response must output the complete, standalone HTML/CSS code that generates the entire page from scratch. If you want to make small changes relative to the previous version, use code_edit parameter to structure your reasoning before producing the final script: First, pinpoint the exact lines to modify in the previous script. Then provide a unified-style mini diff using the following format (no extra commentary inside the block):\n-: [lines to remove]\n+: [lines to add]\n. After the diff, apply those changes and output the full, updated HTML/CSS code (not just the patch).",
            "parameters": {
                "type": "object",
                "properties": {
                    "thought": {
                        "type": "string",
                        "description": "Analyze the current state and provide a clear plan for the required changes. Consider HTML structure, CSS styling, and responsive design optimization opportunities."
                    },
                    "code_edit": {
                        "type": "string", 
                        "description": "Provide your code modifications in the following format:\n-: [lines to remove]\n+: [lines to add]\nFocus on HTML structure and CSS styling."
                    },
                    "full_code": {
                        "type": "string",
                        "description": "Merge your code changes into the full code with proper formatting. Ensure proper HTML string."
                    }
                },
                "required": ["thought", "full_code"]
            }
        }
    }
]

# Create MCP instance
mcp = FastMCP("html-executor")

# Global executor instance
_executor = None

class HTMLExecutor:
    """Executes HTML/CSS code and generates screenshots."""
    
    def __init__(self, output_dir: str, browser_command: str = "google-chrome"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.browser_command = browser_command
        self.count = 0
        
    def _save_html_file(self, html_code: str, filename: str = "index.html") -> str:
        """Save HTML code to a temporary file."""
        html_path = os.path.join(self.output_dir, filename)
        
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_code)
        
        return html_path
    
    def _take_screenshot(self, html_path: str, output_path: str, 
                        width: int = 1920, height: int = 1080) -> Tuple[bool, str]:
        """Take a screenshot of the HTML page using headless browser."""
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
        """Optimize the screenshot image."""
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
    
    def execute(self, html_code: str) -> Dict:
        """Execute HTML code and generate screenshot."""
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
def initialize(args: dict) -> dict:
    """
    Initialize the HTML executor.
    """
    global _executor
    try:
        _executor = HTMLExecutor(args.get("output_dir"), args.get("browser_command", "google-chrome"))
        return {"status": "success", "output": {"text": ["HTML executor initialized successfully."], "tool_configs": tool_configs}}
    except Exception as e:
        return {"status": "error", "output": {"text": [str(e)]}}

@mcp.tool()
def execute_and_evaluate(thought: str = '', code_edit: str = '', full_code: str = '') -> dict:
    """
    Execute HTML/CSS code and generate screenshot.
    
    Args:
        thought: Analysis of current state and plan for changes
        code: The HTML/CSS code to execute
    """
    global _executor
    if _executor is None:
        return {"status": "error", "output": {"text": ["HTML executor not initialized. Call initialize_executor first."]}}
    try:
        result = _executor.execute(full_code)
        return result
    except Exception as e:
        return {"status": "error", "output": {"text": [str(e)]}}

def main():
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        args = {"output_dir": "output/test/design2code/", "browser_command": "google-chrome"}
        initialize(args)
        code = """<!DOCTYPE html>\\n<html lang=\\\"en\\\">\\n<head>\\n    <meta charset=\\\"UTF-8\\\">\\n    <meta name=\\\"viewport\\\" content=\\\"width=device-width, initial-scale=1.0\\\">\\n    <title>Crosschain Risk Framework</title>\\n    <style>\\n        body {\\n            display: flex;\\n            margin: 0;\\n            font-family: Arial, sans-serif;\\n        }\\n        /* Sidebar */\\n        .sidebar {\\n            width: 20%;\\n            background-color: #f7f8fa;\\n            padding: 20px;\\n            box-shadow: 2px 0 5px rgba(0,0,0,0.1);\\n            min-height: 100vh;\\n        }\\n        .sidebar h2, .sidebar ul {\\n            margin: 0;\\n            padding: 0;\\n        }\\n        .sidebar ul {\\n            list-style-type: none;\\n            padding-top: 20px;\\n        }\\n        .sidebar li {\\n            padding: 10px 0;\\n        }\\n        \\n        /* Main Content */\\n        .main-content {\\n            width: 80%;\\n            padding: 20px;\\n        }\\n        h1, h2 {\\n            margin: 20px 0;\\n        }\\n\\n        /* Header */\\n        .header {\\n            display: flex;\\n            justify-content: flex-end;\\n            align-items: center;\\n            padding: 10px;\\n            background-color: #2c3e50;\\n            color: white;\\n        }\\n        .search-bar {\\n            margin-right: 15px;\\n        }\\n        .github-icon {\\n            width: 24px;\\n            height: 24px;\\n        }\\n    </style>\\n</head>\\n<body>\\n    <div class=\\\"sidebar\\\">\\n        <h2>Crosschain Risk Framework</h2>\\n        <ul>\\n            <li>Introduction</li>\\n            <li>Categories of Risk</li>\\n            <li>Network Consensus Risk</li>\\n            <li>Protocol Architecture Risk</li>\\n        </ul>\\n    </div>\\n    <div class=\\\"main-content\\\">\\n        <div class=\\\"header\\\">\\n            <input class=\\\"search-bar\\\" type=\\\"text\\\" placeholder=\\\"Search\\\" style=\\\"width:200px; height:30px;\\\">\\n            <img class=\\\"github-icon\\\" src=\\\"github-icon.png\\\" alt=\\\"GitHub\\\">\\n        </div>\\n        <h1>Introduction</h1>\\n        <p><!-- Introduction content goes here --></p>\\n    </div>\\n</body>\\n</html>"""
        result = execute_and_evaluate(full_code=code)
        print("Result: ", result)
    else:
        mcp.run(transport="stdio")

if __name__ == "__main__":
    main()