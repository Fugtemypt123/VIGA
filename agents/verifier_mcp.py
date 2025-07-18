import os
import json
import uuid
import asyncio
from PIL import Image
import io
import base64
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from openai import OpenAI
from mcp.server.fastmcp import FastMCP
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from contextlib import AsyncExitStack

@dataclass
class VerificationSession:
    """Represents a verification session with all its state."""
    session_id: str
    vision_model: str
    api_key: str
    thoughtprocess_save: str
    max_rounds: int
    verifier_hints: Optional[str]
    target_image_path: Optional[str]
    blender_save: Optional[str]
    memory: List[Dict]
    current_round: int
    created_at: str
    last_updated: str
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'VerificationSession':
        return cls(**data)

class ExternalToolClient:
    """Client for connecting to external MCP tool servers."""
    
    def __init__(self):
        self.image_session = None
        self.scene_session = None
        self.exit_stack = AsyncExitStack()
        self.connection_timeout = 30  # 30 seconds timeout
    
    async def connect_image_server(self, image_server_path: str):
        """Connect to the image processing MCP server with timeout."""
        try:
            server_params = StdioServerParameters(
                command="python",
                args=[image_server_path],
                env=None
            )
            
            # Use asyncio.wait_for to add timeout
            stdio_transport = await asyncio.wait_for(
                self.exit_stack.enter_async_context(stdio_client(server_params)),
                timeout=self.connection_timeout
            )
            stdio, write = stdio_transport
            
            self.image_session = await asyncio.wait_for(
                self.exit_stack.enter_async_context(ClientSession(stdio, write)),
                timeout=self.connection_timeout
            )
            
            await asyncio.wait_for(
                self.image_session.initialize(),
                timeout=self.connection_timeout
            )
            
            # List available tools
            response = await asyncio.wait_for(
                self.image_session.list_tools(),
                timeout=10  # Shorter timeout for tool listing
            )
            tools = response.tools
            print(f"Connected to Image server with tools: {[tool.name for tool in tools]}")
            
        except asyncio.TimeoutError:
            raise RuntimeError(f"Failed to connect to Image server: Connection timeout after {self.connection_timeout}s")
        except Exception as e:
            raise RuntimeError(f"Failed to connect to Image server: {str(e)}")
    
    async def connect_scene_server(self, scene_server_path: str):
        """Connect to the scene investigation MCP server with timeout."""
        try:
            server_params = StdioServerParameters(
                command="python", 
                args=[scene_server_path],
                env=None
            )
            
            # Use asyncio.wait_for to add timeout
            stdio_transport = await asyncio.wait_for(
                self.exit_stack.enter_async_context(stdio_client(server_params)),
                timeout=self.connection_timeout
            )
            stdio, write = stdio_transport
            
            self.scene_session = await asyncio.wait_for(
                self.exit_stack.enter_async_context(ClientSession(stdio, write)),
                timeout=self.connection_timeout
            )
            
            await asyncio.wait_for(
                self.scene_session.initialize(),
                timeout=self.connection_timeout
            )
            
            # List available tools
            response = await asyncio.wait_for(
                self.scene_session.list_tools(),
                timeout=10  # Shorter timeout for tool listing
            )
            tools = response.tools
            print(f"Connected to Scene server with tools: {[tool.name for tool in tools]}")
            
        except asyncio.TimeoutError:
            raise RuntimeError(f"Failed to connect to Scene server: Connection timeout after {self.connection_timeout}s")
        except Exception as e:
            raise RuntimeError(f"Failed to connect to Scene server: {str(e)}")
    
    async def exec_pil_code(self, code: str) -> Dict:
        """Execute PIL code using external server."""
        if not self.image_session:
            raise RuntimeError("Image server not connected")
        
        result = await self.image_session.call_tool("exec_pil_code", {"code": code})
        return json.loads(result.content[0].text) if result.content else {}
    
    async def compare_images(self, path1: str, path2: str) -> str:
        """Compare images using external server."""
        if not self.image_session:
            raise RuntimeError("Image server not connected")
        
        result = await self.image_session.call_tool("compare_images", {
            "path1": path1, 
            "path2": path2
        })
        content = json.loads(result.content[0].text) if result.content else {}
        return content.get("description", "")
    
    async def get_scene_info(self, blender_path: str) -> Dict:
        """Get scene information using external server."""
        if not self.scene_session:
            raise RuntimeError("Scene server not connected")
        
        result = await self.scene_session.call_tool("get_scene_info", {
            "blender_path": blender_path
        })
        return json.loads(result.content[0].text) if result.content else {}
    
    async def focus_on_object(self, blender_path: str, save_dir: str, 
                            round_num: int, object_name: str) -> str:
        """Focus on object using external server."""
        if not self.scene_session:
            raise RuntimeError("Scene server not connected")
        
        result = await self.scene_session.call_tool("focus", {
            "blender_path": blender_path,
            "save_dir": save_dir,
            "round_num": round_num,
            "object_name": object_name
        })
        content = json.loads(result.content[0].text) if result.content else {}
        return content.get("image", "")
    
    async def zoom_camera(self, save_dir: str, direction: str) -> str:
        """Zoom camera using external server."""
        if not self.scene_session:
            raise RuntimeError("Scene server not connected")
        
        result = await self.scene_session.call_tool("zoom", {
            "save_dir": save_dir,
            "direction": direction
        })
        content = json.loads(result.content[0].text) if result.content else {}
        return content.get("image", "")
    
    async def move_camera(self, save_dir: str, direction: str) -> str:
        """Move camera using external server."""
        if not self.scene_session:
            raise RuntimeError("Scene server not connected")
        
        result = await self.scene_session.call_tool("move", {
            "save_dir": save_dir,
            "direction": direction
        })
        content = json.loads(result.content[0].text) if result.content else {}
        return content.get("image", "")
    
    async def cleanup(self):
        """Clean up connections with timeout."""
        try:
            await asyncio.wait_for(self.exit_stack.aclose(), timeout=10)
        except asyncio.TimeoutError:
            logging.warning("Cleanup timeout, forcing close")
        except Exception as e:
            logging.warning(f"Cleanup error: {e}")

class MCPVerifierAgent:
    """
    An MCP agent that verifies visual scenes and provides feedback.
    This agent follows the MCP server pattern with session management and tool integration.
    """
    
    def __init__(self, image_server_path: str = None, scene_server_path: str = None):
        """Initialize the MCP Verifier Agent."""
        self.sessions: Dict[str, VerificationSession] = {}
        self.logger = logging.getLogger(__name__)
        self.tool_client = ExternalToolClient()
        self.image_server_path = image_server_path
        self.scene_server_path = scene_server_path
        self._tools_connected = False
    
    async def connect_tools(self):
        """Connect to external tool servers."""
        if self._tools_connected:
            return {"status": "success", "message": "Tools already connected"}
        
        try:
            if self.image_server_path:
                await self.tool_client.connect_image_server(self.image_server_path)
            if self.scene_server_path:
                await self.tool_client.connect_scene_server(self.scene_server_path)
            self._tools_connected = True
            
            connected_tools = []
            if self.image_server_path:
                connected_tools.append("image processing")
            if self.scene_server_path:
                connected_tools.append("scene investigation")
            
            return {
                "status": "success", 
                "message": f"Connected to external tool servers: {', '.join(connected_tools)}"
            }
        except Exception as e:
            return {
                "status": "error", 
                "error": str(e)
            }
    
    async def _ensure_tools_connected(self):
        """Ensure external tool servers are connected."""
        if not self._tools_connected:
            result = await self.connect_tools()
            if result["status"] != "success":
                raise RuntimeError(f"Failed to connect tools: {result.get('error', 'Unknown error')}")
    
    def create_session(self, 
                      vision_model: str,
                      api_key: str,
                      thoughtprocess_save: str,
                      max_rounds: int = 10,
                      verifier_hints: Optional[str] = None,
                      target_image_path: Optional[str] = None,
                      blender_save: Optional[str] = None) -> str:
        """
        Create a new verification session.
        
        Returns:
            str: Session ID
        """
        session_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        
        # Initialize memory if target images are provided
        memory = []
        if target_image_path:
            memory = self._build_system_prompt(
                vision_model, api_key, verifier_hints, target_image_path
            )
        
        session = VerificationSession(
            session_id=session_id,
            vision_model=vision_model,
            api_key=api_key,
            thoughtprocess_save=thoughtprocess_save,
            max_rounds=max_rounds,
            verifier_hints=verifier_hints,
            target_image_path=target_image_path,
            blender_save=blender_save,
            memory=memory,
            current_round=0,
            created_at=now,
            last_updated=now
        )
        
        self.sessions[session_id] = session
        self.logger.info(f"Created new verification session: {session_id}")
        return session_id
    
    def _build_system_prompt(self, vision_model: str, api_key: str, hints: str, 
                           target_image_path: str) -> List[Dict]:
        """Build the system prompt for the verifier."""
        full_prompt = []
        
        # Add system prompt
        full_prompt.append({
            "role": "system",
            "content": """You're a 3D visual feedback assistant tasked with providing revision suggestions to a 3D scene designer. At the beginning, you will be given several images that describe the target 3D scene. In each subsequent interaction, you will receive a few images of the current 3D scene along with the code that generated it.
Your responsibilities include:
1. **Visual Difference Identification**: Identify differences between the target scene and the current scene. You may use visual tools to assist in this process and should pay close attention to detail. Only answer the most obvious 1-2 differences at a time, don't answer too many.
2. **Code Localization**: Pinpoint locations in the code that could be modified to reduce or eliminate these differences. This may require counterfactual reasoning and inference from the visual discrepancies."""
        })
        
        user_content = []
        
        # Add target images
        target_image_path_1 = os.path.join(target_image_path, 'render1.png')
        if os.path.exists(target_image_path_1):
            user_content.extend([
                {
                    "type": "text",
                    "text": f"Target Image (View 1): {target_image_path_1}"
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{self._get_image_base64(target_image_path_1)}"}
                }
            ])
        
        target_image_path_2 = os.path.join(target_image_path, 'render2.png')
        if os.path.exists(target_image_path_2):
            user_content.extend([
                {
                    "type": "text",
                    "text": f"Target Image (View 2): {target_image_path_2}"
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{self._get_image_base64(target_image_path_2)}"}
                }
            ])
        
        # Add hints
        if hints is not None:
            user_content.append({
                "type": "text",
                "text": f"Hints:\n{hints}"
            })
        
        # Add output format
        output_format = """Output Structure:
1. Thought: Analyze the current state and provide a clear plan for the required changes.
2. Visual Difference: Describe the main differences found (between the current and target scene). Only answer the most obvious 1-2 differences at a time, don't answer too many.
3. Code Localization: Pinpoint locations in the code that could be modified to reduce or eliminate these most obvious differences.
If the current scene is very close to the target scene, only output an "OK!" and do not output other characters."""
        
        user_content.append({
            "type": "text",
            "text": output_format
        })
        
        full_prompt.append({
            "role": "user",
            "content": user_content
        })
        return full_prompt
    
    def _get_image_base64(self, image_path: str) -> str:
        """Convert image to base64 string."""
        try:
            image = Image.open(image_path)
            img_byte_array = io.BytesIO()
            image.save(img_byte_array, format='PNG')
            img_byte_array.seek(0) 
            base64enc_image = base64.b64encode(img_byte_array.read()).decode('utf-8') 
            return base64enc_image
        except Exception as e:
            self.logger.error(f"Failed to convert image to base64: {e}")
            return ""
    
    async def verify_scene(self, session_id: str, code: str, render_path: str, round_num: int) -> Dict[str, Any]:
        """
        Verify a scene against the target.
        
        Args:
            session_id: The session ID
            code: The code that generated the scene
            render_path: Path to the rendered images
            round_num: Current round number
            
        Returns:
            Dict containing verification result
        """
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.sessions[session_id]
        
        if session.current_round >= session.max_rounds:
            return {
                "status": "max_rounds_reached",
                "message": f"Maximum rounds ({session.max_rounds}) reached",
                "round": session.current_round
            }
        
        # Get paths for both views
        view1_path = os.path.join(render_path, 'render1.png')
        view2_path = os.path.join(render_path, 'render2.png')
        
        # Prepare verification message
        verify_message = {
            "role": "user",
            "content": [{"type": "text", "text": f"Please analyze the current state:\nCode: {code}"}]
        }
        
        scene_content = [
            {"type": "text", "text": f"Current scene (View 1): {view1_path}"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{self._get_image_base64(view1_path)}"}}
        ]
        
        # Add view2 if available
        if os.path.exists(view2_path):
            scene_content.extend([
                {"type": "text", "text": f"Current scene (View 2): {view2_path}"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{self._get_image_base64(view2_path)}"}}
            ])
        
        verify_message["content"].extend(scene_content)
        
        # Add output format
        output_format = """Output Structure:
1. Thought: Analyze the current state and provide a clear plan for the required changes.
2. Visual Difference: Describe the main differences found (between the current and target scene). Only answer the most obvious 1-2 differences at a time, don't answer too many.
3. Code Localization: Pinpoint locations in the code that could be modified to reduce or eliminate these most obvious differences.
If the current scene is very close to the target scene, only output an "OK!" and do not output other characters."""
        
        verify_message["content"].append({"type": "text", "text": output_format})
        
        session.memory.append(verify_message)
        
        # Start verification loop
        try:
            client = OpenAI(api_key=session.api_key)
            
            for i in range(session.max_rounds):
                # Get analysis from model
                response = client.chat.completions.create(
                    model=session.vision_model,
                    messages=session.memory,
                    tools=self._get_tools(),
                    tool_choice="auto"
                )
                
                message = response.choices[0].message
                session.memory.append(message.model_dump())
                
                # Handle tool calls and get analysis
                if message.tool_calls:
                    for tool_call in message.tool_calls:
                        tool_response = await self._handle_tool_call(tool_call, session)
                        session.memory.append({
                            "role": "tool", 
                            "tool_call_id": tool_call.id, 
                            "name": tool_call.function.name, 
                            "content": tool_response['text']
                        })
                        
                        if tool_response['image']:
                            session.memory.append({
                                "role": "user", 
                                "content": [
                                    {"type": "text", "text": "Generated image:"}, 
                                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{self._get_image_base64(tool_response['image'])}"}}
                                ]
                            })
                else:
                    # Check if the scene matches the target
                    if "OK" in message.content and "Code Localization" not in message.content:
                        result = {"status": "end", "output": message.content}
                    else:
                        result = {"status": "continue", "output": message.content}
                    break
                
                # Save thought process after each round
                self.save_thought_process(session_id)
            
            session.current_round += 1
            session.last_updated = datetime.now().isoformat()
            
            return result
            
        except Exception as e:
            self.logger.error(f"Verification failed for session {session_id}: {e}")
            return {
                "status": "error",
                "error": str(e),
                "round": session.current_round
            }
    
    def _get_tools(self) -> List[Dict]:
        """Get the tools available for the verifier."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "investigate_3d",
                    "description": """A tool for detailed 3D scene investigation with the following operations:
   - Focus on an object: Set the camera to track a specific object
   - Zoom in/out: Adjust the camera distance from the object (each adjustment changes radius by 1 unit)
   - Move camera: Move the camera on a sphere around the object (each movement covers 1/4 pi r^2 area)
   The camera moves on a sphere around the target object, where zooming changes the sphere's radius.
Focus the tool on the object whose shape or position needs to be adjusted, and adjust one (or more) appropriate angles to observe their specific changes.""",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "operation": {
                                "type": "string",
                                "enum": ["focus", "zoom", "move"],
                                "description": "The operation to perform: focus on object (set the camera to track a specific object), zoom in/out (adjust the camera distance from the object (each adjustment changes radius by 1 unit)), or move camera (move the camera on a sphere around the object (each movement covers 1/4 pi r^2 area))"
                            },
                            "object_name": {
                                "type": "string",
                                "description": "Name of the object to focus on (required for focus operation)"
                            },
                            "direction": {
                                "type": "string",
                                "enum": ["in", "out", "up", "down", "left", "right"],
                                "description": "Direction for zoom (in/out) or camera movement (up/down/left/right)"
                            }
                        },
                        "required": ["operation"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "compare_images",
                    "description": """A tool for comparing two images and identifying visual differences. This tool is more capable at identifying subtle differences than visual inspection alone. It highlights differences in red and provides detailed descriptions of what has changed between the images. Use this tool when you need to precisely identify differences between current and target scenes.""",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "current_image_path": {
                                "type": "string",
                                "description": "Path to the current scene image"
                            },
                            "target_image_path": {
                                "type": "string", 
                                "description": "Path to the target scene image to compare against"
                            },
                            "view_name": {
                                "type": "string",
                                "description": "Optional name for the view being compared (e.g., 'view1', 'view2')"
                            }
                        },
                        "required": ["current_image_path", "target_image_path"]
                    }
                }
            }
        ]
    
    async def _handle_tool_call(self, tool_call, session) -> Dict[str, Any]:
        """Handle tool calls from the model."""
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)
        
        try:
            if function_name == "investigate_3d":
                await self._ensure_tools_connected()
                
                # Handle different operations
                if function_args['operation'] == 'focus':
                    if 'object_name' not in function_args:
                        raise ValueError("object_name is required for focus operation")
                    output = await self.tool_client.focus_on_object(
                        session.blender_save or "", 
                        session.thoughtprocess_save, 
                        session.current_round, 
                        function_args['object_name']
                    )
                    if output:
                        return {'text': f"Focused camera on object: {function_args['object_name']}", 'image': output}
                    else:
                        return {'text': f"Failed to focus camera on object: {function_args['object_name']}", 'image': None}
                
                elif function_args['operation'] == 'zoom':
                    if 'direction' not in function_args:
                        raise ValueError("direction is required for zoom operation")
                    output = await self.tool_client.zoom_camera(
                        session.thoughtprocess_save, 
                        function_args['direction']
                    )
                    if output:
                        return {'text': f"Zoomed {function_args['direction']}", 'image': output}
                    else:
                        return {'text': f"Failed to zoom {function_args['direction']}", 'image': None}
                
                elif function_args['operation'] == 'move':
                    if 'direction' not in function_args:
                        raise ValueError("direction is required for move operation")
                    output = await self.tool_client.move_camera(
                        session.thoughtprocess_save, 
                        function_args['direction']
                    )
                    if output:
                        return {'text': f"Moved camera {function_args['direction']}", 'image': output}
                    else:
                        return {'text': f"Failed to move camera {function_args['direction']}", 'image': None}
                
                else:
                    raise ValueError(f"Unknown operation: {function_args['operation']}")
            
            elif function_name == "compare_images":
                await self._ensure_tools_connected()
                
                current_image_path = function_args.get('current_image_path')
                target_image_path = function_args.get('target_image_path')
                view_name = function_args.get('view_name', 'comparison')
                
                if not current_image_path or not target_image_path:
                    raise ValueError("Both current_image_path and target_image_path are required")
                
                # Check if files exist
                if not os.path.exists(current_image_path):
                    raise ValueError(f"Current image not found: {current_image_path}")
                if not os.path.exists(target_image_path):
                    raise ValueError(f"Target image not found: {target_image_path}")
                
                diff_description = await self.tool_client.compare_images(current_image_path, target_image_path)
                
                return {
                    'text': f"Image comparison result for {view_name}: {diff_description}",
                    'image': None
                }
            
            else:
                raise ValueError(f"Unknown tool: {function_name}")
                
        except Exception as e:
            return {'text': f"Error executing tool: {str(e)}", 'image': None}
    
    def save_thought_process(self, session_id: str) -> None:
        """Save the thought process for a session."""
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.sessions[session_id]
        try:
            with open(session.thoughtprocess_save, "w") as f:
                json.dump(session.memory, f, indent=4, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Failed to save thought process for session {session_id}: {e}")
    
    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """Get information about a session."""
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.sessions[session_id]
        return {
            "session_id": session.session_id,
            "current_round": session.current_round,
            "max_rounds": session.max_rounds,
            "created_at": session.created_at,
            "last_updated": session.last_updated,
            "memory_length": len(session.memory)
        }
    
    def get_memory(self, session_id: str) -> List[Dict]:
        """Get the memory for a session."""
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        
        return self.sessions[session_id].memory
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all active sessions."""
        return [self.get_session_info(session_id) for session_id in self.sessions.keys()]
    
    def delete_session(self, session_id: str) -> None:
        """Delete a session."""
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        
        del self.sessions[session_id]
        self.logger.info(f"Deleted session: {session_id}")
    
    def reset_session_memory(self, session_id: str) -> None:
        """Reset the memory for a session."""
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.sessions[session_id]
        session.memory = []
        session.current_round = 0
        session.last_updated = datetime.now().isoformat()
    
    async def cleanup(self) -> None:
        """Clean up external tool client connections."""
        await self.tool_client.cleanup()


def main():
    """Main function to run the MCP Verifier Agent as an MCP server."""
    mcp = FastMCP("verifier")
    
    # Store agent instances per session to handle different tool server paths
    agent_instances = {}
    
    @mcp.tool()
    async def create_verification_session(
        vision_model: str,
        api_key: str,
        thoughtprocess_save: str,
        max_rounds: int = 10,
        verifier_hints: str = None,
        target_image_path: str = None,
        blender_save: str = None,
        image_server_path: str = "servers/verifier/image.py",
        scene_server_path: str = "servers/verifier/scene.py"
    ) -> dict:
        """
        Create a new verification session with automatic tool server connection.
        """
        try:
            # Create or get agent instance for these tool server paths
            agent_key = f"{image_server_path}:{scene_server_path}"
            if agent_key not in agent_instances:
                agent_instances[agent_key] = MCPVerifierAgent(
                    image_server_path=image_server_path,
                    scene_server_path=scene_server_path
                )
            
            agent = agent_instances[agent_key]
            
            # Connect to tool servers if not already connected
            connect_result = await agent.connect_tools()
            
            session_id = agent.create_session(
                vision_model=vision_model,
                api_key=api_key,
                thoughtprocess_save=thoughtprocess_save,
                max_rounds=max_rounds,
                verifier_hints=verifier_hints,
                target_image_path=target_image_path,
                blender_save=blender_save
            )
            
            if connect_result["status"] == "success":
                return {
                    "status": "success",
                    "session_id": session_id,
                    "message": f"Verification session created successfully. {connect_result['message']}"
                }
            else:
                return {
                    "status": "partial_success",
                    "session_id": session_id,
                    "message": "Verification session created successfully, but tool connection failed",
                    "tool_error": connect_result.get("error", "Unknown error")
                }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    @mcp.tool()
    async def verify_scene(session_id: str, code: str, render_path: str, round_num: int) -> dict:
        """
        Verify a scene against the target.
        """
        try:
            # Find the agent that has this session
            for agent in agent_instances.values():
                if session_id in agent.sessions:
                    result = await agent.verify_scene(session_id, code, render_path, round_num)
                    return result
            return {"status": "error", "error": f"Session {session_id} not found"}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    @mcp.tool()
    async def exec_pil_code(code: str, image_server_path: str = "servers/verifier/image.py") -> dict:
        """
        Execute PIL code for image processing.
        """
        try:
            # Create or get agent instance for this image server path
            agent_key = f"{image_server_path}:"
            if agent_key not in agent_instances:
                agent_instances[agent_key] = MCPVerifierAgent(
                    image_server_path=image_server_path,
                    scene_server_path=None
                )
            
            agent = agent_instances[agent_key]
            await agent._ensure_tools_connected()
            result = await agent.tool_client.exec_pil_code(code)
            return result
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    @mcp.tool()
    async def compare_images(path1: str, path2: str, image_server_path: str = "servers/verifier/image.py") -> dict:
        """
        Compare two images and describe differences.
        """
        try:
            # Create or get agent instance for this image server path
            agent_key = f"{image_server_path}:"
            if agent_key not in agent_instances:
                agent_instances[agent_key] = MCPVerifierAgent(
                    image_server_path=image_server_path,
                    scene_server_path=None
                )
            
            agent = agent_instances[agent_key]
            await agent._ensure_tools_connected()
            result = await agent.tool_client.compare_images(path1, path2)
            return {"description": result}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    @mcp.tool()
    def save_thought_process(session_id: str) -> dict:
        """
        Save the thought process for a session.
        """
        try:
            # Find the agent that has this session
            for agent in agent_instances.values():
                if session_id in agent.sessions:
                    agent.save_thought_process(session_id)
                    return {
                        "status": "success",
                        "message": "Thought process saved successfully"
                    }
            return {"status": "error", "error": f"Session {session_id} not found"}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    @mcp.tool()
    def get_session_info(session_id: str) -> dict:
        """
        Get information about a session.
        """
        try:
            # Find the agent that has this session
            for agent in agent_instances.values():
                if session_id in agent.sessions:
                    info = agent.get_session_info(session_id)
                    return info
            return {"status": "error", "error": f"Session {session_id} not found"}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    @mcp.tool()
    def get_memory(session_id: str) -> dict:
        """
        Get the memory for a session.
        """
        try:
            # Find the agent that has this session
            for agent in agent_instances.values():
                if session_id in agent.sessions:
                    memory = agent.get_memory(session_id)
                    return {"memory": memory}
            return {"status": "error", "error": f"Session {session_id} not found"}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    @mcp.tool()
    def list_sessions() -> dict:
        """
        List all active sessions.
        """
        try:
            all_sessions = []
            for agent in agent_instances.values():
                sessions = agent.list_sessions()
                all_sessions.extend(sessions)
            return {"sessions": all_sessions}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    @mcp.tool()
    def delete_session(session_id: str) -> dict:
        """
        Delete a session.
        """
        try:
            # Find the agent that has this session
            for agent in agent_instances.values():
                if session_id in agent.sessions:
                    agent.delete_session(session_id)
                    return {
                        "status": "success",
                        "message": "Session deleted successfully"
                    }
            return {"status": "error", "error": f"Session {session_id} not found"}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    @mcp.tool()
    def reset_session_memory(session_id: str) -> dict:
        """
        Reset the memory for a session.
        """
        try:
            # Find the agent that has this session
            for agent in agent_instances.values():
                if session_id in agent.sessions:
                    agent.reset_session_memory(session_id)
                    return {
                        "status": "success",
                        "message": "Session memory reset successfully"
                    }
            return {"status": "error", "error": f"Session {session_id} not found"}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    @mcp.tool()
    async def cleanup_verifier() -> dict:
        """
        Clean up all verifier agents and their connections.
        """
        try:
            for agent in agent_instances.values():
                await agent.cleanup()
            return {"status": "success", "message": "All verifier agents cleaned up successfully"}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main() 