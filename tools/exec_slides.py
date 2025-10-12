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
            "description": "Execute code modifications and trigger verifier evaluation. This tool combines code execution with automatic verification. Always use this tool when you want to execute your code changes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "thought": {
                        "type": "string",
                        "description": "Analyze the current state and provide a clear plan for the required changes. Consider scene representation consistency and infinigen optimization opportunities."
                    },
                    "code_edition": {
                        "type": "string", 
                        "description": "Provide your code modifications in the following format:\n-: [lines to remove]\n+: [lines to add]\nFocus on scene consistency and use infinigen functions when appropriate."
                    },
                    "full_code": {
                        "type": "string",
                        "description": "Merge your code changes into the full code with proper formatting. Ensure consistent scene representation."
                    }
                },
                "required": ["thought", "code_edition", "full_code"]
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

    def execute(self, code: str, round: int) -> dict:
        try:
            round_dir = self.output_dir / f"{round}"
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
def exec_script(code: str, round: int) -> dict:
    """
    Compile and render PPTX from Python code.
    Args:
        code: str - Python code that generates a .pptx
        round: int - round index
    """
    global _executor
    if _executor is None:
        return {"status": "error", "output": "Executor not initialized. Call initialize_executor first."}
    try:
        result = _executor.execute(code, round)
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
        result = executor.execute(code_content, 1)
        
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
