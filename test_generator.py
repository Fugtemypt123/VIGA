 #!/usr/bin/env python3
"""
Standalone test script for the Generator Agent.
This script connects to the generator MCP server and demonstrates its functionality.
"""
import asyncio
import argparse
import os
import sys
import json
from pathlib import Path
from contextlib import AsyncExitStack
from typing import Optional
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import TextContent
from prompts.blender import blender_generator_hints

api_key = os.getenv("OPENAI_API_KEY")

class GeneratorTester:
    def __init__(self):
        self.generator_session: Optional[ClientSession] = None
        self.blender_session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()

    async def connect_to_generator(self, generator_script_path: str):
        """Connect to the Generator MCP server."""
        server_params = StdioServerParameters(
            command="python",
            args=[generator_script_path],
            env=None
        )
        
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.generator_session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        
        await self.generator_session.initialize()
        
        # List available tools
        response = await self.generator_session.list_tools()
        tools = response.tools
        print(f"Connected to Generator with tools: {[tool.name for tool in tools]}")

    async def connect_to_blender_executor(self, blender_script_path: str):
        """Connect to the Blender Executor MCP server."""
        server_params = StdioServerParameters(
            command="python",
            args=[blender_script_path],
            env=None
        )
        
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        stdio, write = stdio_transport
        self.blender_session = await self.exit_stack.enter_async_context(ClientSession(stdio, write))
        
        await self.blender_session.initialize()
        
        # List available tools
        response = await self.blender_session.list_tools()
        tools = response.tools
        print(f"Connected to Blender Executor with tools: {[tool.name for tool in tools]}")

    def _extract_code_from_result(self, result):
        """Extract generated code from the result."""
        if result.content and len(result.content) > 0:
            content_item = result.content[0]
            if isinstance(content_item, TextContent):
                content = content_item.text
                try:
                    # Parse JSON response to extract code
                    json_data = json.loads(content)
                    if 'current_code' in json_data:
                        return json_data['current_code']
                    elif 'generated_code' in json_data:
                        return json_data['generated_code']
                    elif 'code' in json_data:
                        return json_data['code']
                except json.JSONDecodeError:
                    pass
        return None

    async def test_generator_workflow(self, args):
        """Test the complete generator workflow."""
        print("\n=== Testing Generator Agent ===")
        
        # Ensure generator session is initialized
        if self.generator_session is None:
            print("Error: Generator session not initialized")
            return
        
        # 1. Initialize generator
        print("Step 1: Initializing generator...")
        init_result = await self.generator_session.call_tool("initialize_generator", {
            "vision_model": args.vision_model,
            "api_key": api_key,
            "thoughtprocess_save": args.thoughtprocess_save,
            "max_rounds": args.max_rounds,
            "generator_hints": args.generator_hints,
            "init_code": args.init_code,
            "init_image_path": args.init_image_path,
            "target_image_path": args.target_image_path,
            "target_description": args.target_description
        })
        print(f"Initialization result: {init_result.content}")
        
        if not self._is_success(init_result):
            print("Failed to initialize generator")
            return
        
        # 2. Initialize Blender executor
        if self.blender_session:
            print("Step 2: Initializing Blender executor...")
            blender_init_result = await self.blender_session.call_tool("initialize_executor", {
                "blender_command": args.blender_command,
                "blender_file": args.blender_file,
                "blender_script": args.blender_script,
                "script_save": args.script_save,
                "render_save": args.render_save,
                "blender_save": args.blender_save
            })
            print(f"Blender executor initialization result: {blender_init_result.content}")
            
            if not self._is_success(blender_init_result):
                print("Failed to initialize Blender executor")
                return
        
        # 3. Generate initial code
        print("\nStep 3: Generating initial code...")
        gen_result = await self.generator_session.call_tool("generate_code", {})
        print(f"Generation result: {gen_result.content}")
        
        if not self._is_success(gen_result):
            print("Failed to generate initial code")
            return
        
        # 4. Execute generated code with Blender
        print("\nStep 4: Executing generated code with Blender...")
        generated_code = self._extract_code_from_result(gen_result)
        if generated_code and self.blender_session:
            try:
                blender_result = await self.blender_session.call_tool("exec_script", {
                    "code": generated_code,
                    "round": 1
                })
                print(f"Blender execution result: {blender_result.content}")
                
                # Extract execution status for feedback
                blender_content = "{}"
                if blender_result.content and len(blender_result.content) > 0:
                    content_item = blender_result.content[0]
                    if isinstance(content_item, TextContent):
                        blender_content = content_item.text
                
                try:
                    blender_json = json.loads(blender_content)
                    if blender_json.get("status") == "success":
                        print("✅ Blender execution successful!")
                        execution_feedback = "Code executed successfully in Blender and rendered an image."
                    else:
                        print("❌ Blender execution failed!")
                        execution_feedback = f"Code execution failed: {blender_json.get('error', 'Unknown error')}"
                except json.JSONDecodeError:
                    execution_feedback = "Could not parse Blender execution result."
            except Exception as e:
                print(f"Error executing code with Blender: {e}")
                execution_feedback = f"Blender execution error: {str(e)}"
        else:
            execution_feedback = "No code extracted for Blender execution or Blender not connected."
            print("⚠️  Skipping Blender execution")
        
        # 5. Add feedback and generate again
        print("\nStep 5: Adding feedback and generating again...")
        combined_feedback = f"The code looks good, but please add more comments to explain the logic. Blender execution feedback: {execution_feedback}"
        feedback_result = await self.generator_session.call_tool("add_feedback", {
            "feedback": combined_feedback
        })
        print(f"Feedback result: {feedback_result.content}")
        
        gen_result2 = await self.generator_session.call_tool("generate_code", {})
        print(f"Second generation result: {gen_result2.content}")
        
        # 6. Save thought process
        print("\nStep 6: Saving thought process...")
        save_result = await self.generator_session.call_tool("save_thought_process", {})
        print(f"Save result: {save_result.content}")
        
        # 7. Get memory
        print("\nStep 7: Getting memory...")
        memory_result = await self.generator_session.call_tool("get_memory", {})
        memory_length = 0
        if memory_result.content and len(memory_result.content) > 0:
            content_item = memory_result.content[0]
            if isinstance(content_item, TextContent):
                memory_length = len(content_item.text)
        print(f"Memory length: {memory_length}")
        
        print("\n=== Generator Test Complete ===")

    def _is_success(self, result):
        """Check if the result indicates success."""
        if result.content and len(result.content) > 0:
            content_item = result.content[0]
            if isinstance(content_item, TextContent):
                content = content_item.text
                return '"status": "success"' in content or '"status":"success"' in content
        return False

    async def cleanup(self):
        """Clean up resources."""
        await self.exit_stack.aclose()

async def main():
    parser = argparse.ArgumentParser(description="Test Generator Agent")
    parser.add_argument("--generator-script", default="agents/generator_mcp.py",
                       help="Path to generator MCP script")
    parser.add_argument("--executor-script", default="servers/generator/blender.py",
                       help="Path to Blender executor MCP script")
    parser.add_argument("--vision-model", default="gpt-4o",
                       help="OpenAI vision model to use")
    parser.add_argument("--thoughtprocess-save", default="_test/test_generator_thought.json",
                       help="Path to save thought process")
    parser.add_argument("--max-rounds", type=int, default=5,
                       help="Maximum number of rounds")
    parser.add_argument("--generator-hints", default=blender_generator_hints["shape"],
                       help="Hints for generator")
    initial_code = Path("data/blendergym/blendshape1/start.py").read_text()
    parser.add_argument("--init-code", default=initial_code,
                       help="Initial code")
    parser.add_argument("--init-image-path", default="data/blendergym/blendshape1/renders/start",
                       help="Path to initial images")
    parser.add_argument("--target-image-path", default="data/blendergym/blendshape1/renders/goal",
                       help="Path to target images")
    parser.add_argument("--target-description", default="A simple test scene",
                       help="Target description")
    
    # Blender execution parameters
    parser.add_argument("--blender-command", default="utils/blender/infinigen/blender/blender",
                       help="Blender command path")
    parser.add_argument("--blender-file", default="data/blendergym/blendshape1/blender_file.blend",
                       help="Blender template file")
    parser.add_argument("--blender-script", default="data/blendergym/pipeline_render_script.py",
                       help="Blender execution script")
    parser.add_argument("--script-save", default="_test",
                       help="Directory to save generated scripts")
    parser.add_argument("--render-save", default="_test",
                       help="Directory to save rendered images")
    parser.add_argument("--blender-save", default=None,
                       help="Optional Blender save file")
    
    args = parser.parse_args()
    
    # Validate required files
    if not os.path.exists(args.generator_script):
        print(f"Error: Generator script not found: {args.generator_script}")
        sys.exit(1)
    
    if not os.path.exists(args.executor_script):
        print(f"Error: Blender script not found: {args.executor_script}")
        sys.exit(1)
    
    tester = GeneratorTester()
    try:
        await tester.connect_to_generator(args.generator_script)
        await tester.connect_to_blender_executor(args.executor_script)
        await tester.test_generator_workflow(args)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await tester.cleanup()

if __name__ == "__main__":
    asyncio.run(main())