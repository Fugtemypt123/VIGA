import os
import subprocess
import re
from pathlib import Path
import logging
from mcp.server.fastmcp import FastMCP

# tool config for agent
tool_configs = [
    {
        "type": "function",
        "function": {
            "name": "execute_and_evaluate",
            "description": "Execute Python code for slide generation and trigger verifier evaluation. This tool combines code execution with automatic verification. Always use this tool when you want to execute your code changes.\nReturns either:\n  (1) On error: detailed error information; or \n  (2) On success: a rendered slide image and further modification suggestions from a separate verifier agent.\nImportant: The execution environment cannot read files generated in previous runs. Therefore, each response must output the complete, standalone Python code that generates the entire slide from scratch. If you want to make small changes relative to the previous version, use code_edit parameter to structure your reasoning before producing the final script: First, pinpoint the exact lines to modify in the previous script. Then provide a unified-style mini diff using the following format (no extra commentary inside the block):\n-: [lines to remove]\n+: [lines to add]\n. After the diff, apply those changes and output the full, updated Python code (not just the patch).",
            "parameters": {
                "type": "object",
                "properties": {
                    "thought": {
                        "type": "string",
                        "description": "Analyze the current state and provide a clear plan for the required changes. Consider slide design, layout, typography, and visual hierarchy optimization opportunities."
                    },
                    "code_edit": {
                        "type": "string", 
                        "description": "Provide your code modifications in the following format:\n-: [lines to remove]\n+: [lines to add]\nFocus on slide design and python-pptx library usage."
                    },
                    "full_code": {
                        "type": "string",
                        "description": "Merge your code changes into the full code with proper formatting. Ensure proper Python code using python-pptx library."
                    }
                },
                "required": ["thought", "full_code"]
            }
        }
    }
]

mcp = FastMCP("slides-executor")

_executor = None

class SlidesExecutor:
    def __init__(self, task_dir: str, output_dir: str):
        self.task_dir = Path(task_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.count = 0

    def _execute_slide_code(self, code_path: str) -> str:
        generate_dir = "utils/slides"
        env = os.environ.copy()
        env['PYTHONPATH'] = f"{generate_dir}:{env.get('PYTHONPATH', '')}"
        try:
            result = subprocess.run(["python", code_path], capture_output=True, text=True, check=True, env=env)
            pptx_file = code_path.replace("runned_code.py", "refine.pptx")
            subprocess.run(["/usr/bin/python3", "/usr/bin/unoconv", "-f", "jpg", pptx_file], check=True)
            return "Success"
        except subprocess.CalledProcessError as e:
            logging.error(f"PPTX compilation failed: {e.stderr}")
            return f"Error: {e.stderr}"

    def execute(self, code: str) -> dict:
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
            code = re.sub(r'presentation\.save\("([^"]+\.pptx)"\)',
                          f'presentation.save("{slide_path}")',
                          code)
            # replace all '{xxx}/media/image_{}.jpg' to 'self.task_dir/media/image_{}.jpg'
            code = re.sub(r'media/image_(\d+)\.jpg',
                          f'{self.task_dir}/media/image_\\1.jpg',
                          code)
            
            with open(runned_code_path, "w") as f:
                f.write(code)

            result = self._execute_slide_code(str(runned_code_path))

            if result == "Success" and image_path.exists():
                return {"status": "success", "output": [str(image_path)]}
            else:
                return {"status": "failure", "output": result}
        except Exception as e:
            return {"status": "failure", "output": str(e)}

@mcp.tool()
def initialize(args: dict) -> dict:
    """
    Initialize Slides executor and set all necessary parameters.
    """
    global _executor
    try:
        _executor = SlidesExecutor(args.get("task_dir"), args.get("output_dir"))
        return {"status": "success", "output": "Slides executor initialized successfully"}
    except Exception as e:
        return {"status": "error", "output": str(e)}

@mcp.tool()
def execute_and_evaluate(thought: str = '', code_edit: str = '', full_code: str = '') -> dict:
    """
    Compile and render PPTX from Python code.
    Args:
        thought: Analysis of current state and plan for changes
        code_edit: Code modifications in diff format
        full_code: Complete Python code that generates a .pptx
    """
    global _executor
    if _executor is None:
        return {"status": "error", "output": "Executor not initialized. Call initialize_executor first."}
    try:
        result = _executor.execute(full_code)
        return result
    except Exception as e:
        return {"status": "error", "output": str(e)}

def test_specific_file():
    """
    Test if specific file output/autopresent/20250817_172322/slide_11/1/refine.py can run normally
    """
    test_file_path = "data/autopresent/examples/art_photos/slide_1/start.py"
    
    print(f"Starting test file: {test_file_path}")
    print("=" * 60)
    
    # Check if file exists
    if not os.path.exists(test_file_path):
        print(f"❌ File does not exist: {test_file_path}")
        return False
    
    print(f"✅ File exists: {test_file_path}")
    
    # Read file content
    try:
        with open(test_file_path, 'r', encoding='utf-8') as f:
            code_content = f.read()
        print(f"✅ File read successfully, code length: {len(code_content)} characters")
    except Exception as e:
        print(f"❌ File read failed: {e}")
        return False
    
    # Create temporary executor for testing
    try:
        task_dir = "data/autopresent/examples/art_photos/slide_1"
        temp_output_dir = "output/test"
        executor = SlidesExecutor(task_dir, temp_output_dir)
        print(f"✅ Executor initialized successfully, output directory: {temp_output_dir}")
    except Exception as e:
        print(f"❌ Executor initialization failed: {e}")
        return False
    
    # Execute code
    try:
        print("Executing code...")
        result = executor.execute(code_content)
        
        if result["status"] == "success":
            print("✅ Code executed successfully!")
            print(f"   Generated image path: {result.get('output', 'N/A')}")
            print(f"   Image base64 length: {len(result.get('image_base64', ''))} characters")
            
            return True
        else:
            print(f"❌ Code execution failed: {result.get('output', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Exception occurred during execution: {e}")
        return False

def main():
    # If running this script directly, execute test
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        success = test_specific_file()
        sys.exit(0 if success else 1)
    else:
        # Run MCP service normally
        mcp.run(transport="stdio")

if __name__ == "__main__":
    main()
