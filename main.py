 #!/usr/bin/env python3
"""
Main entry for dual-agent interactive framework (generator/verifier).
Supports 3D (Blender) and 2D (PPTX) modes.
Uses MCP stdio connections instead of HTTP servers.
"""
import argparse
import os
import sys
import time
import json
import asyncio
from pathlib import Path
from typing import Optional
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import TextContent

# ========== Agent Client Wrappers ==========

class GeneratorAgentClient:
    def __init__(self, script_path: str):
        self.script_path = script_path
        self.session = None
        self.exit_stack = AsyncExitStack()
        self.initialized = False

    async def connect(self):
        """Connect to the Generator MCP server."""
        server_params = StdioServerParameters(
            command="python",
            args=[self.script_path],
            env=None
        )
        
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        
        await self.session.initialize()
        print(f"Connected to Generator: {self.script_path}")

    async def create_session(self, **kwargs):
        """Initialize the generator with session parameters."""
        if not self.session:
            raise RuntimeError("Not connected. Call connect() first.")
        
        result = await self.session.call_tool("initialize_generator", kwargs)
        if result.content and len(result.content) > 0:
            content = result.content[0].text
            if '"status": "success"' in content or '"status":"success"' in content:
                self.initialized = True
                return "success"
        raise RuntimeError(f"Failed to create session: {result.content}")

    async def generate_code(self, feedback: Optional[str] = None):
        """Generate code, optionally with feedback."""
        if not self.initialized:
            raise RuntimeError("Generator not initialized. Call create_session() first.")
        
        params = {}
        if feedback:
            params["feedback"] = feedback
        
        result = await self.session.call_tool("generate_code", params)
        if result.content and len(result.content) > 0:
            content = result.content[0].text
            return json.loads(content)
        raise RuntimeError(f"Failed to generate code: {result.content}")

    async def add_feedback(self, feedback: str):
        """Add feedback to the generator."""
        if not self.initialized:
            raise RuntimeError("Generator not initialized. Call create_session() first.")
        
        result = await self.session.call_tool("add_feedback", {"feedback": feedback})
        if not (result.content and len(result.content) > 0 and 
                ('"status": "success"' in result.content[0].text or '"status":"success"' in result.content[0].text)):
            raise RuntimeError(f"Failed to add feedback: {result.content}")

    async def save_thought_process(self):
        """Save the thought process."""
        if not self.initialized:
            raise RuntimeError("Generator not initialized. Call create_session() first.")
        
        result = await self.session.call_tool("save_thought_process", {})
        if not (result.content and len(result.content) > 0 and 
                ('"status": "success"' in result.content[0].text or '"status":"success"' in result.content[0].text)):
            raise RuntimeError(f"Failed to save thought process: {result.content}")

    async def cleanup(self):
        """Clean up resources."""
        await self.exit_stack.aclose()


class VerifierAgentClient:
    def __init__(self, script_path: str):
        self.script_path = script_path
        self.session = None
        self.exit_stack = AsyncExitStack()
        self.initialized = False
        self.session_id = None

    async def connect(self):
        """Connect to the Verifier MCP server."""
        server_params = StdioServerParameters(
            command="python",
            args=[self.script_path],
            env=None
        )
        
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        
        await self.session.initialize()
        print(f"Connected to Verifier: {self.script_path}")

    async def create_session(self, **kwargs):
        """Create a verification session with automatic tool server connection."""
        if not self.session:
            raise RuntimeError("Not connected. Call connect() first.")
        
        result = await self.session.call_tool("create_verification_session", kwargs)
        if result.content and len(result.content) > 0:
            content = result.content[0].text
            try:
                json_data = json.loads(content)
                if json_data.get("status") == "success":
                    self.initialized = True
                    self.session_id = json_data.get("session_id")
                    return "success"
            except json.JSONDecodeError:
                pass
        raise RuntimeError(f"Failed to create session: {result.content}")

    async def verify_scene(self, code: str, render_path: str, round_num: int):
        """Verify the scene with given parameters."""
        if not self.initialized or not self.session_id:
            raise RuntimeError("Verifier not initialized. Call create_session() first.")
        
        result = await self.session.call_tool("verify_scene", {
            "session_id": self.session_id,
            "code": code,
            "render_path": render_path,
            "round_num": round_num
        })
        if result.content and len(result.content) > 0:
            content = result.content[0].text
            return json.loads(content)
        raise RuntimeError(f"Failed to verify scene: {result.content}")

    async def save_thought_process(self):
        """Save the thought process."""
        if not self.initialized or not self.session_id:
            raise RuntimeError("Verifier not initialized. Call create_session() first.")
        
        result = await self.session.call_tool("save_thought_process", {
            "session_id": self.session_id
        })
        if not (result.content and len(result.content) > 0 and 
                ('"status": "success"' in result.content[0].text or '"status":"success"' in result.content[0].text)):
            raise RuntimeError(f"Failed to save thought process: {result.content}")

    async def cleanup(self):
        """Clean up resources."""
        await self.exit_stack.aclose()

# ========== Main Dual-Agent Loop ==========

async def main():
    parser = argparse.ArgumentParser(description="Dual-agent interactive framework")
    parser.add_argument("--mode", choices=["blendergym", "autopresent", "blendergym-hard", "demo"], required=True, help="Choose 3D (Blender) or 2D (PPTX) mode")
    parser.add_argument("--vision-model", default="gpt-4o", help="OpenAI vision model")
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"), help="OpenAI API key")
    parser.add_argument("--max-rounds", type=int, default=10, help="Max interaction rounds")
    parser.add_argument("--init-code-path", default="data/blendergym/blendshape1/start.py", help="Path to initial code file")
    parser.add_argument("--init-image-path", default="data/blendergym/blendshape1/renders/start", help="Path to initial images")
    parser.add_argument("--target-image-path", default="data/blendergym/blendshape1/renders/goal", help="Path to target images")
    parser.add_argument("--target-description", default=None, help="Target description for 2D mode")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    parser.add_argument("--task-name", default="blendershape", help="Task name for hints extraction")
    
    # Agent server paths 
    parser.add_argument("--generator-script", default="agents/generator_mcp.py", help="Generator MCP script path")
    parser.add_argument("--verifier-script", default="agents/verifier_mcp.py", help="Verifier MCP script path")
    
    # Blender execution parameters (for generator)
    parser.add_argument("--blender-server-path", default="servers/generator/blender.py", help="Path to Blender MCP server script")
    parser.add_argument("--blender-command", default="utils/blender/infinigen/blender/blender", help="Blender command path")
    parser.add_argument("--blender-file", default="data/blendergym/blendshape1/blender_file.blend", help="Blender template file")
    parser.add_argument("--blender-script", default="data/blendergym/pipeline_render_script.py", help="Blender execution script")
    parser.add_argument("--save-blender-file", action="store_true", help="Save blender file")
    
    # Slides execution parameters (for generator)
    parser.add_argument("--slides-server-path", default="servers/generator/slides.py", help="Path to Slides MCP server script")
    
    # Tool server paths (for verifier)
    parser.add_argument("--image-server-path", default="servers/verifier/image.py", help="Path to image processing MCP server script")
    parser.add_argument("--scene-server-path", default="servers/verifier/scene.py", help="Path to scene investigation MCP server script")
    
    args = parser.parse_args()

    # Prepare output dirs
    os.makedirs(args.output_dir, exist_ok=True)

    # Init agents
    generator = GeneratorAgentClient(args.generator_script)
    verifier = VerifierAgentClient(args.verifier_script)

    try:
        # Connect to agents
        await generator.connect()
        await verifier.connect()

        # Create generator session
        generator_params = {
            "vision_model": args.vision_model,
            "api_key": args.api_key,
            "task_name": args.task_name,
            "max_rounds": args.max_rounds,
            "init_code_path": args.init_code_path,
            "init_image_path": args.init_image_path,
            "target_image_path": args.target_image_path,
            "target_description": args.target_description,
            "thought_save" : args.output_dir / "thoughts"
        }
        
        # Add mode-specific parameters
        if args.mode == "blendergym":
            generator_params.update({
                "blender_server_path": args.blender_server_path,
                "blender_command": args.blender_command,
                "blender_file": args.blender_file,
                "blender_script": args.blender_script,
                "render_save": args.output_dir / "renders",
                "script_save": args.output_dir / "scripts",
                "blender_save": args.output_dir / "blender_file.blend" if args.save_blender_file else None
            })
        elif args.mode == "autopresent":
            generator_params.update({
                "slides_server_path": args.slides_server_path,
                "code_save": args.output_dir / "scripts",
                "image_save": args.output_dir / "images",
                "slide_save": args.output_dir / "slides",
            })
        else:
            raise NotImplementedError("Mode not implemented")
        
        await generator.create_session(**generator_params)
        
        # Create verifier session
        verifier_params = {
            "vision_model": args.vision_model,
            "api_key": args.api_key,
            "max_rounds": args.max_rounds,
            "task_name": args.task_name,
            "target_image_path": args.target_image_path,
            "target_description": args.target_description,
            "thought_save": args.output_dir / "thoughts"
        }
        
        # Add mode-specific parameters
        if args.mode == "blendergym" or args.mode == "autopresent":
            verifier_params.update({
                "image_server_path": args.image_server_path,
                "scene_server_path": None
            })
        else:
            raise NotImplementedError("Mode not implemented")
        
        await verifier.create_session(**verifier_params)

        # Main loop
        for round_num in range(args.max_rounds):
            print(f"\n=== Round {round_num+1} ===")
            
            print("Step 1: Generator generating code...")
            gen_result = await generator.generate_code()
            if gen_result.get("status") == "max_rounds_reached":
                print("Max rounds reached. Stopping.")
                break
            if gen_result.get("status") == "error":
                print(f"Generator error: {gen_result['error']}")
                break
            
            # Extract code from result
            code = gen_result.get("code") or gen_result.get("current_code") or gen_result.get("generated_code")
            if not code:
                print("No code generated")
                break
                
            print(f"Generated code (truncated):\n{code[:200]}...")
            
            # Check if automatic execution happened
            if gen_result.get("execution_result"):
                exec_result = gen_result["execution_result"]
                if exec_result.get("status") == "success":
                    print("✅ Automatic execution completed!")
                    result = exec_result.get("result", {})
                    if result.get("status") == "success":
                        print("   - Generation successful")
                    else:
                        print(f"   - Generation failed: {result.get('output', 'Unknown error')}")
                        await generator.add_feedback(f"Execution error: {result.get('output')}")
                        continue
                else:
                    print(f"   - Execution failed: {exec_result.get('error', 'Unknown error')}")
                    await generator.add_feedback(f"Execution error: {exec_result.get('error')}")
                    continue
            else:
                # Manual execution (when automatic execution is not available)
                # print("Step 2: Executing code...")
                # print("   - Execution handled by generator agent internally")
                raise ValueError("Unknown error in automatic execution")
            
            print("Step 3: Verifier analyzing scene...")
            verify_result = await verifier.verify_scene(
                code=code,
                render_path=result,
                round_num=round_num
            )
            
            print(f"Verifier result: {verify_result.get('status')}")
            if verify_result.get("status") == "end":
                print("Verifier: OK! Task complete.")
                break
            elif verify_result.get("status") == "continue":
                feedback = verify_result["output"]
                print(f"Verifier feedback: {feedback}")
                await generator.add_feedback(feedback)
            else:
                print(f"Verifier error: {verify_result.get('error')}")
                break
            
            print("Step 4: Saving thought processes...")
            await generator.save_thought_process()
            await verifier.save_thought_process()
            await asyncio.sleep(1)
            
    except Exception as e:
        print(f"Error in main loop: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        print("Cleaning up...")
        await generator.cleanup()
        await verifier.cleanup()
    
    print("\n=== Dual-agent interaction finished ===")

if __name__ == "__main__":
    asyncio.run(main())