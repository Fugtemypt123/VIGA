import asyncio
import os
import json
from PIL import Image
import io
import base64
from openai import OpenAI
from typing import Dict, List, Optional, Any
import logging
from mcp.server.fastmcp import FastMCP
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from contextlib import AsyncExitStack
from prompts import prompts_dict

class ExternalToolClient:
    """Client for connecting to external MCP tool servers (blender/slides)."""
    
    def __init__(self):
        self.mcp_sessions = {}  # server_type -> McpSession
        self.connection_timeout = 30  # 30 seconds timeout
    
    async def connect_server(self, server_type: str, server_path: str):
        """Connect to the specified MCP server with timeout in a background task."""
        if server_type in self.mcp_sessions:
            return  # Already connected
            
        ready_event = asyncio.Event()
        
        async def mcp_session_runner() -> None:
            try:
                server_params = StdioServerParameters(
                    command="python",
                    args=[server_path],
                )
                
                exit_stack = AsyncExitStack()
                stdio_transport = await asyncio.wait_for(
                    exit_stack.enter_async_context(stdio_client(server_params)),
                    timeout=self.connection_timeout
                )
                stdio, write = stdio_transport
                session = await asyncio.wait_for(
                    exit_stack.enter_async_context(ClientSession(stdio, write)),
                    timeout=self.connection_timeout
                )
                await asyncio.wait_for(
                    session.initialize(),
                    timeout=self.connection_timeout
                )
                
            except Exception as e:
                print(f"Error during MCP connection setup: {e}")
                raise ConnectionError(f"Failed to connect to {server_type} server: {e}") from e
            finally:
                print(f"Sending {server_type} MCP connection ready event")
                ready_event.set()

            try:
                stop_event = asyncio.Event()
                
                # Store the session
                current_task = asyncio.current_task()
                assert current_task is not None, "Current task should not be None"
                
                self.mcp_sessions[server_type] = McpSession(
                    name=server_type,
                    client=session,
                    task=current_task,
                    stop_event=stop_event,
                )
                
                # List available tools
                response = await asyncio.wait_for(
                    session.list_tools(),
                    timeout=10
                )
                tools = response.tools
                print(f"Connected to {server_type.capitalize()} server with tools: {[tool.name for tool in tools]}")
                
                # Wait for the stop event
                await stop_event.wait()
                
            except asyncio.CancelledError:
                print(f"{server_type} MCP session cancelled")
                raise
            except Exception as e:
                print(f"Error during {server_type} MCP session: {e}")
                raise
            finally:
                print(f"Closing {server_type} MCP session")
                try:
                    await exit_stack.aclose()
                except Exception as e:
                    print(f"Error during {server_type} exit stack close: {e}")
                print(f"{server_type} MCP session closed")

        # Run the session runner in a separate task
        asyncio.create_task(mcp_session_runner())
        print(f"Waiting for {server_type} MCP connection to be ready")
        await ready_event.wait()
        print(f"{server_type} MCP connection is ready")
    
    async def initialize_executor(self, server_type: str, **kwargs) -> Dict:
        """Initialize the executor using external server with timeout."""
        session = self.mcp_sessions.get(server_type)
        if not session:
            raise RuntimeError(f"{server_type.capitalize()} server not connected")
        try:
            result = await asyncio.wait_for(
                session.client.call_tool("initialize_executor", kwargs),
                timeout=30
            )
            content = json.loads(result.content[0].text) if result.content else {}
            return content
        except asyncio.TimeoutError:
            raise RuntimeError(f"{server_type.capitalize()} executor initialization timeout after 30s")
        except Exception as e:
            raise RuntimeError(f"{server_type.capitalize()} executor initialization failed: {str(e)}")
    
    async def exec_script(self, server_type: str, code: str, round_num: int, **kwargs) -> Dict:
        """Execute script using external server with timeout."""
        session = self.mcp_sessions.get(server_type)
        if not session:
            raise RuntimeError(f"{server_type.capitalize()} server not connected")
        if server_type == "blender":
            tool_name = "exec_script"
            tool_args = {"code": code, "round": round_num}
        elif server_type == "slides":
            tool_name = "exec_pptx"
            tool_args = {"code": code, "round": round_num, "code_save": kwargs.get("code_save")}
        else:
            raise ValueError(f"Unknown server_type: {server_type}")
        try:
            result = await asyncio.wait_for(
                session.client.call_tool(tool_name, tool_args),
                timeout=60
            )
            content = json.loads(result.content[0].text) if result.content else {}
            return content
        except asyncio.TimeoutError:
            raise RuntimeError(f"{server_type.capitalize()} script execution timeout after 60s")
        except Exception as e:
            raise RuntimeError(f"{server_type.capitalize()} script execution failed: {str(e)}")
    
    async def call_tool(self, server_type: str, tool_name: str, tool_args: dict, timeout: int = 120) -> Any:
        """Call a specific tool on the external server with timeout."""
        session = self.mcp_sessions.get(server_type)
        if not session:
            raise RuntimeError(f"{server_type.capitalize()} server not connected")
        try:
            result = await asyncio.wait_for(
                session.client.call_tool(tool_name, tool_args),
                timeout=timeout
            )
            content = json.loads(result.content[0].text) if result.content else {}
            return content
        except asyncio.TimeoutError:
            raise RuntimeError(f"{server_type.capitalize()} tool call timeout after {timeout}s")
        except Exception as e:
            raise RuntimeError(f"{server_type.capitalize()} tool call failed: {str(e)}")
    
    async def cleanup(self):
        """Clean up connections by closing all MCP sessions."""
        for server_type, mcp_session in self.mcp_sessions.items():
            try:
                await mcp_session.close()
            except Exception as e:
                logging.warning(f"Cleanup error for {server_type}: {e}")


class McpSession:
    """Manages a single MCP session with its own task and cleanup."""
    
    def __init__(self, name: str, client: ClientSession, task: asyncio.Task, stop_event: asyncio.Event):
        self.name = name
        self.client = client
        self.task = task
        self.stop_event = stop_event

    async def close(self) -> None:
        """Close the MCP session by setting stop event and waiting for task completion."""
        print(f"Sending stop event to {self.name}")
        self.stop_event.set()
        print(f"Waiting for task {self.name} to finish")
        await self.task
        print(f"Task {self.name} finished")

class GeneratorAgent:
    """
    An MCP agent that takes code modification suggestions and implements them.
    This agent follows the MCP server pattern for better encapsulation and tool integration.
    """
    
    def __init__(self, 
                 mode: str,
                 vision_model: str,
                 api_key: str,
                 thought_save: str,
                 task_name: str,
                 max_rounds: int = 10,
                 init_code_path: Optional[str] = None,
                 init_image_path: Optional[str] = None,
                 target_image_path: Optional[str] = None,
                 target_description: Optional[str] = None,
                 blender_server_path: Optional[str] = None,
                 slides_server_path: Optional[str] = None,
                 output_dir: Optional[str] = None,
                 api_base_url: Optional[str] = None):
        """
        Initialize the Generator Agent.
        """
        self.mode = mode
        self.model = vision_model
        self.api_key = api_key
        self.task_name = task_name  # Store task_name for blendergym-hard level detection
        # Support custom OpenAI-compatible base URL
        client_kwargs = {"api_key": self.api_key}
        if api_base_url or os.getenv("OPENAI_BASE_URL"):
            client_kwargs["base_url"] = api_base_url or os.getenv("OPENAI_BASE_URL")
        self.client = OpenAI(**client_kwargs)
        self.thought_save = thought_save
        self.max_rounds = max_rounds
        self.current_round = 0
        self.tool_client = ExternalToolClient()
        self._server_connected = False
        self.output_dir = output_dir
        # Decide which server to use
        if mode == "blendergym" or mode == "blendergym-hard":
            self.server_type = "blender"
            self.server_path = blender_server_path
            # Store blender file path for Meshy asset generation
            self.blender_file_path = None  # Will be set during executor setup
        elif mode == "autopresent":
            self.server_type = "slides"
            self.server_path = slides_server_path
        else:
            raise NotImplementedError("Mode not implemented")
        
        # Initialize memory if initial parameters are provided
        if mode == "blendergym":
            self.memory = self._build_blendergym_system_prompt(mode, task_name, init_code_path, init_image_path, target_image_path)
        elif mode == "autopresent":
            self.memory = self._build_autopresent_system_prompt(mode, init_code_path, init_image_path, target_description)
        elif mode == "blendergym-hard":
            self.memory = self._build_blendergym_hard_system_prompt(mode, task_name, init_code_path, init_image_path, target_image_path)
        else:
            raise NotImplementedError("Mode not implemented")
    
    async def _ensure_server_connected(self):
        if not self._server_connected and self.server_type and self.server_path:
            await self.tool_client.connect_server(self.server_type, self.server_path)
            self._server_connected = True
    
    async def setup_executor(self, **kwargs):
        await self._ensure_server_connected()
        result = await self.tool_client.initialize_executor(self.server_type, **kwargs)
        
        # Store blender file path for Meshy asset generation
        if self.server_type == "blender" and "blender_file" in kwargs:
            self.blender_file_path = kwargs["blender_file"]
            
            # Initialize investigator for blendergym-hard
            if self.mode == "blendergym-hard" and "thought_save" in kwargs:
                try:
                    investigator_result = await self.tool_client.call_tool("blender", "initialize_investigator", {
                        "thoughtprocess_save": kwargs["thought_save"],
                        "blender_path": kwargs["blender_file"]
                    })
                    if investigator_result.get("status") == "success":
                        logging.info("Investigator initialized successfully")
                    else:
                        logging.warning(f"Investigator initialization failed: {investigator_result.get('error')}")
                except Exception as e:
                    logging.warning(f"Failed to initialize investigator: {e}")
        
        return result
    
    def _build_blendergym_hard_system_prompt(self, 
                             mode: str, 
                             task_name: str, 
                             init_code_path: str = None, 
                             init_image_path: str = None, 
                             target_image_path: str = None) -> List[Dict]:
        """
        Build the system prompt for the generator for blendergym-hard mode.
        """
        level = task_name.split('-')[0]
        full_prompt = []
        # Add system prompt
        full_prompt.append({"role": "system", "content": prompts_dict[mode]['system']['generator'][level]})
        user_content = []
        # Add initial code (except level-1)
        if level != 'level1':
            init_code = open(init_code_path, 'r').read()
            user_content = [{"type": "text", "text": f"Initial Code:\n```python\n{init_code}\n```"}]
        else:
            # add investigator tool description
            user_content = [{"type": "text", "text": "You have access to a 3D scene investigation tool that allows you to:"}]
            user_content.append({"type": "text", "text": "1. Focus the camera on specific objects in the scene"})
            user_content.append({"type": "text", "text": "2. Zoom in/out to get better views"})
            user_content.append({"type": "text", "text": "3. Move the camera around to explore different angles"})
        # Add initial images
        init_image_path_1 = os.path.join(init_image_path, 'render1.png')
        if os.path.exists(init_image_path_1):
            user_content.append({"type": "text", "text": "Initial Image (View 1):"})
            user_content.append({"type": "image_url", "image_url": {"url": self._get_image_base64(init_image_path_1)}})
        else:
            # At least we need one initial image
            raise ValueError(f"Initial image {init_image_path_1} does not exist!")
        # Add target images (for mode `blendergym`)
        target_image_path_1 = os.path.join(target_image_path, 'visprompt1.png')
        if os.path.exists(target_image_path_1):
            user_content.append({"type": "text", "text": "Target Image (View 1):"})
            user_content.append({"type": "image_url", "image_url": {"url": self._get_image_base64(target_image_path_1)}})
        else:
            raise ValueError(f"Target image {target_image_path_1} does not exist!")
        # Add hints 
        user_content.append({"type": "text", "text": f"Your task: {prompts_dict[mode]['hints'][task_name.split('-')[0]][task_name.split('-')[1]]}"})
        # Add output format
        user_content.append({"type": "text", "text": prompts_dict[mode]['format']['generator'][level]})
        # Add all user content
        full_prompt.append({"role": "user", "content": user_content})
        return full_prompt
        
    def _build_blendergym_system_prompt(self, 
                             mode: str, 
                             task_name: str, 
                             init_code_path: str = None, 
                             init_image_path: str = None, 
                             target_image_path: str = None) -> List[Dict]:
        """
        Build the system prompt for the generator for blendergym mode.
        """
        full_prompt = []
        # Add system prompt
        full_prompt.append({"role": "system", "content": prompts_dict[mode]['system']['generator']})
        
        # Add initial code & code analysis
        init_code = open(init_code_path, 'r').read()
        user_content = [{"type": "text", "text": f"Initial Code:\n```python\n{init_code}\n```"}]
        
        # Add code analysis
        code_analysis = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a Blender Python code analysis expert."},
                {"role": "user", "content": f"Please analyze the following Blender Python code line by line, \
                explaining what each part does and how it contributes to the scene:\n```python\n{init_code}\n```"}
            ]
        )
        code_analysis = code_analysis.choices[0].message.content
        user_content.append({"type": "text", "text": f"Code Analysis:\n{code_analysis}"})
        
        # Add initial images
        init_image_path_1 = os.path.join(init_image_path, 'render1.png')
        if os.path.exists(init_image_path_1):
            user_content.append({"type": "text", "text": "Initial Image (View 1):"})
            user_content.append({"type": "image_url", "image_url": {"url": self._get_image_base64(init_image_path_1)}})
        else:
            # At least we need one initial image
            raise ValueError(f"Initial image {init_image_path_1} does not exist!")
        
        init_image_path_2 = os.path.join(init_image_path, 'render2.png')
        if os.path.exists(init_image_path_2):
            user_content.append({"type": "text", "text": "Initial Image (View 2):"})
            user_content.append({"type": "image_url", "image_url": {"url": self._get_image_base64(init_image_path_2)}})
        
        # Add target images (for mode `blendergym`)
        target_image_path_1 = os.path.join(target_image_path, 'render1.png')
        if os.path.exists(target_image_path_1):
            user_content.append({"type": "text", "text": "Target Image (View 1):"})
            user_content.append({"type": "image_url", "image_url": {"url": self._get_image_base64(target_image_path_1)}})
        else:
            logging.error(f"Target image {target_image_path_1} does not exist!")
        
        target_image_path_2 = os.path.join(target_image_path, 'render2.png')
        if os.path.exists(target_image_path_2):
            user_content.append({"type": "text", "text": "Target Image (View 2):"})
            user_content.append({"type": "image_url", "image_url": {"url": self._get_image_base64(target_image_path_2)}})
        
        # Add hints 
        if prompts_dict[mode]['hints']['generator'][task_name] is not None:
            user_content.append({"type": "text", "text": f"Hints:\n{prompts_dict[mode]['hints']['generator'][task_name]}"})
        # Add output format
        user_content.append({"type": "text", "text": prompts_dict[mode]['format']['generator']})
        # Add all user content
        full_prompt.append({"role": "user", "content": user_content})
        return full_prompt
    
    def _build_autopresent_system_prompt(self, 
                             mode: str, 
                             init_code_path: str = None,
                             init_image_path: str = None, 
                             target_description: str = None) -> List[Dict]:
        full_prompt = []
        # Add system prompt
        full_prompt.append({"role": "system", "content": prompts_dict[mode]['system']['generator'] + '\n\n' + prompts_dict[mode]['api_library']})
        
        # Add user input
        user_content = []
        user_content.append({"type": "text", "text": f"Now, here is the task package, which includes the initial code, a screenshot of the initial slides, the provided images with filenames used in the slides, and my instruction:"})
        
        # Add initial code
        init_code = open(init_code_path, 'r').read()
        user_content.append({"type": "text", "text": f"Initial Code:\n```python\n{init_code}\n```"})
        
        # Add initial images
        if os.path.exists(init_image_path):
            user_content.append({"type": "text", "text": "Initial Slide Screenshot:"})
            user_content.append({"type": "image_url", "image_url": {"url": self._get_image_base64(init_image_path)}})
        else:
            user_content.append({"type": "text", "text": "Initial code cannot be executed, please check the code and fix the errors."})
            
        # Add used images
        user_content.append({"type": "text", "text": "Provided Images (they might already appear in the code):"})
        used_image_dir = os.path.join(os.path.dirname(init_image_path), 'media')
        used_images = os.listdir(used_image_dir)
        for image in used_images:
            if image.endswith('.jpg') or image.endswith('.png') or image.endswith('.jpeg'):
                user_content.append({"type": "text", "text": f"Path: {os.path.join('media', image)}"})
                user_content.append({"type": "image_url", "image_url": {"url": self._get_image_base64(os.path.join(used_image_dir, image))}})
        
        # Add target description
        user_content.append({"type": "text", "text": f"Instruction:\n{target_description}"})
        # Add hints
        user_content.append({"type": "text", "text": f"Hints:\n{prompts_dict[mode]['hints']}"})
        # Add output format
        user_content.append({"type": "text", "text": prompts_dict[mode]['format']['generator']})
        # Add all user content
        full_prompt.append({"role": "user", "content": user_content})
        return full_prompt
    
    def _get_image_base64(self, image_path: str) -> str:
        """Return a full data URL for the image, preserving original jpg/png format."""
        image = Image.open(image_path)
        img_byte_array = io.BytesIO()
        ext = os.path.splitext(image_path)[1].lower()
        
        # Convert image to appropriate mode for saving
        if ext in ['.jpg', '.jpeg']:
            save_format = 'JPEG'
            mime_subtype = 'jpeg'
            # JPEG doesn't support transparency, convert RGBA to RGB
            if image.mode in ['RGBA', 'LA', 'P']:
                # Convert P mode to RGB first, then handle RGBA
                if image.mode == 'P':
                    image = image.convert('RGBA')
                # Convert RGBA to RGB with white background
                if image.mode == 'RGBA':
                    # Create a white background
                    background = Image.new('RGB', image.size, (255, 255, 255))
                    background.paste(image, mask=image.split()[-1])  # Use alpha channel as mask
                    image = background
                elif image.mode == 'LA':
                    # Convert LA to RGB
                    image = image.convert('RGB')
        elif ext == '.png':
            save_format = 'PNG'
            mime_subtype = 'png'
            # PNG supports transparency, but convert P mode to RGBA
            if image.mode == 'P':
                image = image.convert('RGBA')
        else:
            # Fallback: keep original format if recognizable, else default to PNG
            save_format = image.format or 'PNG'
            mime_subtype = save_format.lower() if save_format.lower() in ['jpeg', 'png'] else 'png'
            # Handle P mode for fallback cases
            if image.mode == 'P':
                if save_format == 'JPEG':
                    image = image.convert('RGB')
                else:
                    image = image.convert('RGBA')
        
        image.save(img_byte_array, format=save_format)
        img_byte_array.seek(0)
        base64enc_image = base64.b64encode(img_byte_array.read()).decode('utf-8')
        return f"data:image/{mime_subtype};base64,{base64enc_image}"
    
    def _parse_generate_response(self, response: str) -> tuple:
        """
        Parse the generate response.
        Returns: (thought, edit, full_code)
        """
        try:
            full = response.split("Full Code")[1].strip()
        except:
            full = response.strip()
        
        # Remove the ```python and ``` from the full code
        if "```python" in full:
            full = full.split("```python")[1].split("```")[0].strip()
        else:
            full = full.split("```")[0].strip()
        
        return None, None, full

    def _get_tools(self) -> List[Dict]:
        """Get available tools for the generator agent."""
        if self.mode == "blendergym-hard":
            # For blendergym-hard mode, determine tools based on level
            level = self.task_name.split('-')[0]
            tools = []
            
            # Define tool definitions
            meshy_tool = {
                "type": "function",
                "function": {
                    "name": "generate_3d_asset",
                    "description": "Generate and import a 3D asset into the Blender scene using Meshy Text-to-3D API. This tool can create objects based on text descriptions and automatically import them into the current scene.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "description": {
                                "type": "string", 
                                "description": "Text description of the 3D asset to generate (e.g., 'a wooden chair', 'a modern table', 'a decorative plant')"
                            },
                            "location": {
                                "type": "string", 
                                "description": "Position where to place the asset in the scene, format: 'x,y,z' (e.g., '2,0,0')",
                                "default": "0,0,0"
                            },
                            "scale": {
                                "type": "number", 
                                "description": "Scale factor for the asset (e.g., 1.0 for normal size, 2.0 for double size)",
                                "default": 1.0
                            },
                            "refine": {
                                "type": "boolean", 
                                "description": "Whether to apply texture refinement after initial generation (takes longer but produces better quality)",
                                "default": True
                            }
                        },
                        "required": ["description"]
                    }
                }
            }
            
            investigator_tool = {
                "type": "function",
                "function": {
                    "name": "investigate_3d",
                    "description": "A tool for detailed 3D scene investigation with the following operations: focus, zoom, move.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "operation": {"type": "string", "enum": ["focus", "zoom", "move"], "description": "The operation to perform on the 3D scene."},
                            "object_name": {"type": "string", "description": "The name of the object to focus on (only for focus operation)."},
                            "direction": {"type": "string", "enum": ["in", "out", "up", "down", "left", "right"], "description": "The direction to move the camera (only for zoom and move operation)."}
                        },
                        "required": ["operation"]
                    }
                }
            }
            
            # Define exec_script tool
            exec_script_tool = {
                "type": "function",
                "function": {
                    "name": "exec_script",
                    "description": "Execute Blender Python code to modify the 3D scene. This tool allows you to write and execute Python code that can modify objects, lighting, materials, and other scene properties in Blender.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "The Blender Python code to execute. This should be valid Python code that can modify the Blender scene."
                            }
                        },
                        "required": ["code"]
                    }
                }
            }
            
            # Add tools based on level
            if level == "level1":
                # Only investigator tool (tool 3)
                tools.append(investigator_tool)
            elif level == "level2":
                # Only blender code executor (tool 2)
                tools.append(exec_script_tool)
            elif level == "level3":
                # Blender code executor (tool 2) + investigator tool (tool 3)
                tools.append(exec_script_tool)
                tools.append(investigator_tool)
            elif level == "level4":
                # All tools: meshy (tool 1) + blender code executor (tool 2) + investigator tool (tool 3)
                tools.append(meshy_tool)
                tools.append(exec_script_tool)
                tools.append(investigator_tool)
            
            return tools
        else:
            return []
    
    async def _handle_tool_call(self, tool_call) -> Dict[str, Any]:
        """Handle tool calls from the generator agent."""
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)
        
        try:
            if function_name == "generate_3d_asset":
                if self.server_type != "blender":
                    return {'text': "Error: 3D asset generation is only available for Blender mode", 'success': False}
                
                # Call the Meshy asset generation tool
                result = await self.tool_client.call_tool("blender", "add_meshy_asset", {
                    "description": function_args.get("description", ""),
                    "blender_path": self.blender_file_path if hasattr(self, 'blender_file_path') else None,
                    "location": function_args.get("location", "0,0,0"),
                    "scale": function_args.get("scale", 1.0),
                    "refine": function_args.get("refine", True)
                })
                
                if result.get("status") == "success":
                    return {
                        'text': f"Successfully generated and imported 3D asset: {function_args.get('description')}. Object name: {result.get('object_name', 'Unknown')}. Location: {result.get('location', 'Unknown')}. Scale: {result.get('scale', 'Unknown')}",
                        'success': True,
                        'asset_info': result
                    }
                else:
                    return {
                        'text': f"Failed to generate 3D asset: {result.get('error', 'Unknown error')}",
                        'success': False
                    }
            elif function_name == "exec_script":
                if self.server_type != "blender":
                    return {'text': "Error: Blender code execution is only available for Blender mode", 'success': False}
                
                # Execute the Blender Python code
                self.current_round += 1
                result = await self.tool_client.exec_script(
                    server_type=self.server_type,
                    code=function_args.get("code", ""),
                    round_num=self.current_round,
                )
                
                if result.get("status") == "success":
                    return {
                        'text': f"Successfully executed Blender Python code. Round: {self.current_round}",
                        'success': True,
                        'execution_result': result
                    }
                else:
                    return {
                        'text': f"Failed to execute Blender code: {result.get('error', 'Unknown error')}",
                        'success': False
                    }
            elif function_name == "investigate_3d":
                if self.server_type != "blender":
                    return {'text': "Error: 3D investigation is only available for Blender mode", 'success': False}
                
                op = function_args.get('operation')
                if op == 'focus':
                    result = await self.tool_client.call_tool("blender", "focus", {
                        "object_name": function_args.get("object_name", "")
                    })
                    if result.get("status") == "success":
                        return {
                            'text': f"Focused camera on object: {function_args.get('object_name', '')}",
                            'success': True,
                            'image': result.get('image')
                        }
                    else:
                        return {
                            'text': f"Failed to focus: {result.get('error', 'Unknown error')}",
                            'success': False
                        }
                elif op == 'zoom':
                    result = await self.tool_client.call_tool("blender", "zoom", {
                        "direction": function_args.get("direction", "")
                    })
                    if result.get("status") == "success":
                        return {
                            'text': f"Zoomed {function_args.get('direction', '')}",
                            'success': True,
                            'image': result.get('image')
                        }
                    else:
                        return {
                            'text': f"Failed to zoom: {result.get('error', 'Unknown error')}",
                            'success': False
                        }
                elif op == 'move':
                    result = await self.tool_client.call_tool("blender", "move", {
                        "direction": function_args.get("direction", "")
                    })
                    if result.get("status") == "success":
                        return {
                            'text': f"Moved camera {function_args.get('direction', '')}",
                            'success': True,
                            'image': result.get('image')
                        }
                    else:
                        return {
                            'text': f"Failed to move: {result.get('error', 'Unknown error')}",
                            'success': False
                        }
                else:
                    return {
                        'text': f"Unknown operation: {op}",
                        'success': False
                    }
            else:
                return {'text': f"Unknown tool: {function_name}", 'success': False}
        except Exception as e:
            return {'text': f"Error executing tool: {str(e)}", 'success': False}

    async def generate_code(self, feedback: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate code based on current memory and optional feedback.
        
        Args:
            feedback: Optional feedback from verifier or executor
            
        Returns:
            Dict containing the generated code and metadata
        """
        if feedback:
            self.memory.append({"role": "user", "content": feedback})
        
        try:
            # Check if we need to use tools
            use_tools = self.mode in ["blendergym", "blendergym-hard"] and self._server_connected
            
            if use_tools:
                # Get available tools
                available_tools = self._get_tools()
                
                # Use tools-enabled generation
                response = self.client.chat.completions.create(
                    model=self.model, 
                    messages=self.memory,
                    tools=available_tools,
                    tool_choice="auto"
                )
                message = response.choices[0].message
                self.memory.append(message.model_dump())
                
                # Handle tool calls
                if hasattr(message, 'tool_calls') and message.tool_calls:
                    for tool_call in message.tool_calls:
                        tool_response = await self._handle_tool_call(tool_call)
                        self.memory.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_call.function.name,
                            "content": tool_response['text']
                        })
                        
                        # If tool was successful, add success message
                        if tool_response.get('success'):
                            # Add the tool response text
                            self.memory.append({
                                "role": "user",
                                "content": f"Tool execution successful: {tool_response['text']}. Please continue with code generation."
                            })
                            
                            # If there's an image from investigator tool, add it to memory
                            if tool_response.get('image'):
                                self.memory.append({
                                    "role": "user",
                                    "content": [
                                        {"type": "text", "text": "Generated investigation image:"},
                                        {"type": "image_url", "image_url": {"url": self._get_image_base64(tool_response['image'])}}
                                    ]
                                })
                        else:
                            self.memory.append({
                                "role": "user",
                                "content": f"Tool execution failed: {tool_response['text']}. Please continue with code generation without using this tool."
                            })
                    
                    # Continue generation after tool calls
                    continue_response = self.client.chat.completions.create(
                        model=self.model,
                        messages=self.memory
                    )
                    generate_response = continue_response.choices[0].message.content
                    self.memory.append({"role": "assistant", "content": generate_response})
                else:
                    generate_response = message.content
            else:
                # Standard generation without tools
                generate_response = self.client.chat.completions.create(
                    model=self.model, 
                    messages=self.memory
                ).choices[0].message.content
            
            # Parse the response to extract code if needed (only for modes that generate code)
            if self.mode == "blendergym" or (self.mode == "blendergym-hard" and self.task_name.split('-')[0] != "level1"):
                _, _, full_code = self._parse_generate_response(generate_response)
            else:
                full_code = None
            
            return {
                "status": "success",
                "code": full_code,
                "response": generate_response,
                "round": self.current_round
            }
        except Exception as e:
            logging.error(f"Code generation failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "round": self.current_round
            }
    
    def add_feedback(self, feedback: str) -> None:
        """
        Add feedback to the agent's memory.
        
        Args:
            feedback: Feedback from verifier or executor
        """
        self.memory.append({"role": "user", "content": feedback})
    
    def save_thought_process(self) -> None:
        """Save the current thought process to file."""
        try:
            with open(self.thought_save, "w") as f:
                json.dump(self.memory, f, indent=4, ensure_ascii=False)
        except Exception as e:
            logging.error(f"Failed to save thought process: {e}")
    
    def get_memory(self) -> List[Dict]:
        """Get the current memory/conversation history."""
        return self.memory
    
    def reset_memory(self) -> None:
        """Reset the agent's memory."""
        self.memory = []
        self.current_round = 0
    
    async def cleanup(self) -> None:
        """Clean up external connections."""
        await self.tool_client.cleanup()


def main():
    """Main function to run the Generator Agent as an MCP server."""
    mcp = FastMCP("generator")
    
    agent_holder = {}

    @mcp.tool()
    async def initialize_generator(
        mode: str,
        vision_model: str,
        api_key: str,
        thought_save: str,
        task_name: str,
        max_rounds: int = 10,
        init_code_path: str = None,
        init_image_path: str = None,
        target_image_path: str = None,
        target_description: Optional[str] = None,
        # Blender executor parameters
        blender_server_path: str = None,
        blender_command: str = None,
        blender_file: str = None,
        blender_script: str = None,
        script_save: str = None,
        render_save: str = None,
        blender_save: Optional[str] = None,
        # Slides executor parameters
        slides_server_path: str = None,
        output_dir: str = None,
        api_base_url: Optional[str] = None,
    ) -> dict:
        """
        Initialize a new Generator Agent with optional Blender or Slides executor setup.
        """
        try:
            agent = GeneratorAgent(
                mode=mode,
                vision_model=vision_model,
                api_key=api_key,
                thought_save=thought_save,
                task_name=task_name,
                max_rounds=max_rounds,
                init_code_path=init_code_path,
                init_image_path=init_image_path,
                target_image_path=target_image_path,
                target_description=target_description,
                blender_server_path=blender_server_path,
                slides_server_path=slides_server_path,
                output_dir=output_dir,
                api_base_url=api_base_url
            )
            agent_holder['agent'] = agent
            
            setup_results = []
            
            # Setup Blender executor if parameters are provided
            if mode == "blendergym" or mode == "blendergym-hard":
                try:
                    setup_result = await agent.setup_executor(
                        blender_command=blender_command,
                        blender_file=blender_file,
                        blender_script=blender_script,
                        script_save=script_save,
                        render_save=render_save,
                        blender_save=blender_save
                    )
                    setup_results.append(("Blender", setup_result))
                except Exception as e:
                    setup_results.append(("Blender", {"status": "error", "error": str(e)}))
            
            # Setup Slides executor if parameters are provided
            elif mode == "autopresent":
                try:
                    setup_result = await agent.setup_executor(
                        task_dir=os.path.dirname(init_code_path), 
                        output_dir=output_dir
                    )
                    setup_results.append(("Slides", setup_result))
                except Exception as e:
                    setup_results.append(("Slides", {"status": "error", "error": str(e)}))
            
            else:
                raise NotImplementedError("Mode not implemented")
            
            # Determine overall status
            if not setup_results:
                return {
                    "status": "success",
                    "message": "Generator Agent initialized successfully (no executor configured)"
                }
            
            successful_setups = [name for name, result in setup_results if result.get("status") == "success"]
            failed_setups = [name for name, result in setup_results if result.get("status") != "success"]
            
            if successful_setups and not failed_setups:
                return {
                    "status": "success",
                    "message": f"Generator Agent and {', '.join(successful_setups)} executor(s) initialized successfully"
                }
            elif successful_setups and failed_setups:
                return {
                    "status": "partial_success",
                    "message": f"Generator Agent initialized successfully. {', '.join(successful_setups)} executor(s) setup successful, {', '.join(failed_setups)} executor(s) setup failed",
                    "failed_setups": failed_setups
                }
            else:
                return {
                    "status": "partial_success",
                    "message": "Generator Agent initialized successfully, but all executor setups failed",
                    "failed_setups": failed_setups
                }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    @mcp.tool()
    async def generate_code(feedback: str = None) -> dict:
        """
        Generate code using the initialized Generator Agent.
        """
        try:
            if 'agent' not in agent_holder:
                return {"status": "error", "error": "Generator Agent not initialized. Call initialize_generator first."}
            result = await agent_holder['agent'].generate_code(feedback)
            return result
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    @mcp.tool()
    def add_feedback(feedback: str) -> dict:
        """
        Add feedback to the Generator Agent's memory.
        """
        try:
            if 'agent' not in agent_holder:
                return {"status": "error", "error": "Generator Agent not initialized. Call initialize_generator first."}
            agent_holder['agent'].add_feedback(feedback)
            return {"status": "success", "message": "Feedback added successfully"}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    @mcp.tool()
    def save_thought_process() -> dict:
        """
        Save the current thought process to file.
        """
        try:
            if 'agent' not in agent_holder:
                return {"status": "error", "error": "Generator Agent not initialized. Call initialize_generator first."}
            agent_holder['agent'].save_thought_process()
            return {"status": "success", "message": "Thought process saved successfully"}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    @mcp.tool()
    def get_memory() -> dict:
        """
        Get the current memory/conversation history.
        """
        try:
            if 'agent' not in agent_holder:
                return {"status": "error", "error": "Generator Agent not initialized. Call initialize_generator first."}
            memory = agent_holder['agent'].get_memory()
            return {"memory": memory}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    @mcp.tool()
    def reset_memory() -> dict:
        """
        Reset the agent's memory.
        """
        try:
            if 'agent' not in agent_holder:
                return {"status": "error", "error": "Generator Agent not initialized. Call initialize_generator first."}
            agent_holder['agent'].reset_memory()
            return {"status": "success", "message": "Memory reset successfully"}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    @mcp.tool()
    async def cleanup_generator() -> dict:
        """
        Clean up the Generator Agent and its connections.
        """
        try:
            if 'agent' not in agent_holder:
                return {"status": "error", "error": "Generator Agent not initialized. Call initialize_generator first."}
            await agent_holder['agent'].cleanup()
            return {"status": "success", "message": "Generator Agent cleaned up successfully"}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main() 