#!/usr/bin/env python3
"""
Standalone test script for the Verifier Agent.
This script connects to the verifier MCP server and demonstrates its functionality.
"""
import asyncio
import argparse
import os
import sys
import tempfile
import json
from pathlib import Path
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import TextContent
from PIL import Image, ImageDraw
from prompts.blender import blender_verifier_hints

api_key = os.getenv("OPENAI_API_KEY")

class VerifierTester:
    def __init__(self):
        self.session = None
        self.exit_stack = AsyncExitStack()

    async def connect_to_verifier(self, verifier_script_path: str):
        """Connect to the Verifier MCP server."""
        server_params = StdioServerParameters(
            command="python",
            args=[verifier_script_path],
            env=None
        )
        
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        
        await self.session.initialize()
        
        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        print(f"Connected to Verifier with tools: {[tool.name for tool in tools]}")

    def create_test_images(self, target_dir: str, render_dir: str):
        """Create test images for verification."""
        os.makedirs(target_dir, exist_ok=True)
        os.makedirs(render_dir, exist_ok=True)
        
        # Create target images
        target1 = Image.new('RGB', (400, 300), 'lightblue')
        draw1 = ImageDraw.Draw(target1)
        draw1.rectangle([100, 100, 300, 200], fill='red')
        draw1.text((150, 250), "Target Scene", fill='black')
        target1.save(os.path.join(target_dir, 'render1.png'))
        
        target2 = Image.new('RGB', (400, 300), 'lightgreen')
        draw2 = ImageDraw.Draw(target2)
        draw2.ellipse([175, 125, 225, 175], fill='blue')  # circle
        draw2.text((150, 250), "Target View 2", fill='black')
        target2.save(os.path.join(target_dir, 'render2.png'))
        
        # Create current render images (slightly different)
        render1 = Image.new('RGB', (400, 300), 'lightblue')
        draw_r1 = ImageDraw.Draw(render1)
        draw_r1.rectangle([110, 110, 290, 190], fill='darkred')  # Slightly different position/size
        draw_r1.text((150, 250), "Current Scene", fill='black')
        render1.save(os.path.join(render_dir, 'render1.png'))
        
        render2 = Image.new('RGB', (400, 300), 'lightgreen')
        draw_r2 = ImageDraw.Draw(render2)
        draw_r2.ellipse([180, 130, 230, 180], fill='darkblue')  # Slightly different circle
        draw_r2.text((150, 250), "Current View 2", fill='black')
        render2.save(os.path.join(render_dir, 'render2.png'))
        
        print(f"Created test images in {target_dir} and {render_dir}")

    async def test_verifier_workflow(self, args):
        """Test the complete verifier workflow."""
        print("\n=== Testing Verifier Agent ===")
        
        # Create test images
        with tempfile.TemporaryDirectory() as temp_dir:
            target_dir = os.path.join(temp_dir, "target")
            render_dir = os.path.join(temp_dir, "renders")
            self.create_test_images(target_dir, render_dir)
            
            # 1. Create verification session with automatic tool server connection
            print("Step 1: Creating verification session with tool server connection...")
            session_result = await self.session.call_tool("create_verification_session", {
                "vision_model": args.vision_model,
                "api_key": api_key,
                "thoughtprocess_save": args.thoughtprocess_save,
                "max_rounds": args.max_rounds,
                "verifier_hints": args.verifier_hints,
                "target_image_path": target_dir,
                "blender_save": args.blender_save,
                "image_server_path": args.image_server_path,
                "scene_server_path": args.scene_server_path
            })
            print(f"Session result: {session_result.content}")
            
            if not self._is_success(session_result):
                print("Failed to create verification session")
                return
            
            # Check if tool servers were connected successfully
            self._check_tool_connection(session_result)
            
            # Extract session ID
            session_id = self._extract_session_id(session_result)
            if not session_id:
                print("Failed to extract session ID")
                return
            
            print(f"Created session: {session_id}")
            
            # 2. Test PIL code execution (using connected tools)
            print("\nStep 2: Testing PIL code execution...")
            pil_code = """
from PIL import Image, ImageDraw
import io
import base64

# Create a test image
img = Image.new('RGB', (200, 200), 'white')
draw = ImageDraw.Draw(img)
draw.rectangle([50, 50, 150, 150], fill='green')
draw.text((75, 175), "Test PIL", fill='black')
result = img
"""
            pil_result = await self.session.call_tool("exec_pil_code", {
                "code": pil_code,
                "image_server_path": args.image_server_path
            })
            print(f"PIL execution result: Success={self._check_pil_success(pil_result)}")
            
            # 3. Test image comparison (using connected tools)
            print("\nStep 3: Testing image comparison...")
            compare_result = await self.session.call_tool("compare_images", {
                "path1": os.path.join(target_dir, 'render1.png'),
                "path2": os.path.join(render_dir, 'render1.png'),
                "image_server_path": args.image_server_path
            })
            print(f"Image comparison completed: {len(str(compare_result.content)) > 50}")
            
            # 4. Verify scene with tools (this will trigger tool calls from the model)
            print("\nStep 4: Verifying scene with tool access...")
            test_code = """
import bpy
import bmesh

# Clear existing mesh objects
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# Create a cube
bpy.ops.mesh.primitive_cube_add(location=(0, 0, 0))
cube = bpy.context.active_object
cube.name = "TestCube"

# Modify the cube slightly
bpy.context.view_layer.objects.active = cube
bpy.ops.object.mode_set(mode='EDIT')
bm = bmesh.from_mesh(cube.data)
bmesh.ops.bevel(bm, geom=bm.edges, offset=0.1)
bm.to_mesh(cube.data)
bpy.ops.object.mode_set(mode='OBJECT')

# Add material
material = bpy.data.materials.new(name="TestMaterial")
material.use_nodes = True
bsdf = material.node_tree.nodes["Principled BSDF"]
bsdf.inputs[0].default_value = (0.8, 0.2, 0.2, 1.0)  # Red color
cube.data.materials.append(material)
"""
            verify_result = await self.session.call_tool("verify_scene", {
                "session_id": session_id,
                "code": test_code,
                "render_path": render_dir,
                "round_num": 1
            })
            print(f"Scene verification result: {self._extract_verification_status(verify_result)}")
            
            # Print any tool usage that occurred during verification
            self._analyze_verification_response(verify_result)
            
            # 5. Test a second round with different feedback
            print("\nStep 5: Testing second verification round...")
            test_code_2 = """
import bpy

# Get the existing cube
cube = bpy.data.objects.get("TestCube")
if cube:
    # Scale it up
    cube.scale = (1.5, 1.5, 1.5)
    
    # Change material color to blue
    if cube.data.materials:
        material = cube.data.materials[0]
        if material.use_nodes:
            bsdf = material.node_tree.nodes["Principled BSDF"]
            bsdf.inputs[0].default_value = (0.2, 0.2, 0.8, 1.0)  # Blue color
else:
    print("Cube not found!")
"""
            verify_result_2 = await self.session.call_tool("verify_scene", {
                "session_id": session_id,
                "code": test_code_2,
                "render_path": render_dir,
                "round_num": 2
            })
            print(f"Second verification result: {self._extract_verification_status(verify_result_2)}")
            
            # 6. Save thought process
            print("\nStep 6: Saving thought process...")
            save_result = await self.session.call_tool("save_thought_process", {
                "session_id": session_id
            })
            print(f"Save result: {save_result.content}")
            
            # 7. Get session info
            print("\nStep 7: Getting session info...")
            info_result = await self.session.call_tool("get_session_info", {
                "session_id": session_id
            })
            print(f"Session info: {self._extract_session_info(info_result)}")
            
            # 8. List all sessions
            print("\nStep 8: Listing all sessions...")
            list_result = await self.session.call_tool("list_sessions", {})
            print(f"Active sessions: {self._extract_session_count(list_result)}")
            
            # 9. Test cleanup
            print("\nStep 9: Testing cleanup...")
            cleanup_result = await self.session.call_tool("cleanup_verifier", {})
            print(f"Cleanup result: {cleanup_result.content}")
            
            # 10. Delete the session
            print("\nStep 10: Deleting session...")
            delete_result = await self.session.call_tool("delete_session", {
                "session_id": session_id
            })
            print(f"Delete result: {delete_result.content}")
            
            print("\n=== Verifier Test Complete ===")

    def _is_success(self, result):
        """Check if the result indicates success."""
        if result.content and len(result.content) > 0:
            content_item = result.content[0]
            if isinstance(content_item, TextContent):
                content = content_item.text
                return '"status": "success"' in content or '"status":"success"' in content
        return False

    def _check_tool_connection(self, result):
        """Check if tool servers were connected successfully."""
        if result.content and len(result.content) > 0:
            content_item = result.content[0]
            if isinstance(content_item, TextContent):
                content = content_item.text
                if "Connected to external tool servers" in content:
                    print("✅ External tool servers connected successfully")
                elif "tool connection failed" in content:
                    print("❌ Tool server connection failed")
                    # Try to extract error details
                    try:
                        import json
                        json_data = json.loads(content)
                        if 'tool_error' in json_data:
                            print(f"   Error: {json_data['tool_error']}")
                    except:
                        pass
                elif "partial_success" in content:
                    print("⚠️  Partial success - session created but tool connection issues")

    def _check_pil_success(self, result):
        """Check if PIL execution was successful."""
        if result.content and len(result.content) > 0:
            content_item = result.content[0]
            if isinstance(content_item, TextContent):
                content = content_item.text
                return '"success": true' in content or '"success":true' in content
        return False

    def _extract_session_id(self, result):
        """Extract session ID from result."""
        if result.content and len(result.content) > 0:
            content_item = result.content[0]
            if isinstance(content_item, TextContent):
                content = content_item.text
                try:
                    json_data = json.loads(content)
                    return json_data.get("session_id")
                except json.JSONDecodeError:
                    # Fallback to simple string extraction
                    start = content.find('"session_id": "') + len('"session_id": "')
                    end = content.find('"', start)
                    if start > len('"session_id": "') - 1 and end > start:
                        return content[start:end]
        return None

    def _extract_verification_status(self, result):
        """Extract verification status from result."""
        if result.content and len(result.content) > 0:
            content_item = result.content[0]
            if isinstance(content_item, TextContent):
                content = content_item.text
                try:
                    json_data = json.loads(content)
                    return json_data.get("status", "unknown")
                except json.JSONDecodeError:
                    return "parse_error"
        return "no_content"

    def _analyze_verification_response(self, result):
        """Analyze verification response for tool usage."""
        if result.content and len(result.content) > 0:
            content_item = result.content[0]
            if isinstance(content_item, TextContent):
                content = content_item.text
                try:
                    json_data = json.loads(content)
                    output = json_data.get("output", "")
                    
                    # Check for tool usage indicators
                    if "Image comparison result" in output:
                        print("   ✅ Image comparison tool was used")
                    if "Focused camera on object" in output:
                        print("   ✅ 3D investigation tool was used (focus)")
                    if "Zoomed" in output:
                        print("   ✅ 3D investigation tool was used (zoom)")
                    if "Moved camera" in output:
                        print("   ✅ 3D investigation tool was used (move)")
                    
                    # Print the output for analysis
                    if output and output.strip():
                        print(f"   Verification output: {output[:200]}...")
                except json.JSONDecodeError:
                    pass

    def _extract_session_info(self, result):
        """Extract session info summary."""
        if result.content and len(result.content) > 0:
            content_item = result.content[0]
            if isinstance(content_item, TextContent):
                content = content_item.text
                try:
                    json_data = json.loads(content)
                    return f"Round {json_data.get('current_round', 0)}/{json_data.get('max_rounds', 0)}, Memory: {json_data.get('memory_length', 0)} items"
                except json.JSONDecodeError:
                    pass
        return "unknown"

    def _extract_session_count(self, result):
        """Extract number of active sessions."""
        if result.content and len(result.content) > 0:
            content_item = result.content[0]
            if isinstance(content_item, TextContent):
                content = content_item.text
                try:
                    json_data = json.loads(content)
                    sessions = json_data.get("sessions", [])
                    return len(sessions)
                except json.JSONDecodeError:
                    pass
        return "unknown"

    async def cleanup(self):
        """Clean up resources."""
        await self.exit_stack.aclose()

async def main():
    parser = argparse.ArgumentParser(description="Test Verifier Agent")
    parser.add_argument("--verifier-script", default="agents/verifier_mcp.py",
                       help="Path to verifier MCP script")
    parser.add_argument("--vision-model", default="gpt-4o",
                       help="OpenAI vision model to use")
    parser.add_argument("--thoughtprocess-save", default="_test/test_verifier_thought.json",
                       help="Path to save thought process")
    parser.add_argument("--max-rounds", type=int, default=3,
                       help="Maximum number of rounds")
    parser.add_argument("--verifier-hints", default=blender_verifier_hints["shape"],
                       help="Hints for verifier")
    parser.add_argument("--blender-save", default=None,
                       help="Blender save path")
    
    # Tool server paths
    parser.add_argument("--image-server-path", default="servers/verifier/image.py",
                       help="Path to image processing MCP server script")
    parser.add_argument("--scene-server-path", default="servers/verifier/scene.py",
                       help="Path to scene investigation MCP server script")
    
    args = parser.parse_args()
    
    # Validate required files
    if not os.path.exists(args.verifier_script):
        print(f"Error: Verifier script not found: {args.verifier_script}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs("_test", exist_ok=True)
    
    tester = VerifierTester()
    try:
        await tester.connect_to_verifier(args.verifier_script)
        await tester.test_verifier_workflow(args)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await tester.cleanup()

if __name__ == "__main__":
    asyncio.run(main())