import os
import json
import asyncio
from PIL import Image
import io
import base64
from typing import Dict, List, Optional, Any
import logging
from openai import OpenAI
from mcp.server.fastmcp import FastMCP
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from contextlib import AsyncExitStack
from prompts import prompts_dict

class ExternalToolClient:
    """Client for connecting to external MCP tool servers (image/scene)."""
    def __init__(self):
        self.mcp_sessions = {}  # server_type -> McpSession
        self.connection_timeout = 30
        
    async def connect_server(self, server_type: str, server_path: str, api_key: str = None):
        """Connect to the specified MCP server with timeout in a background task."""
        if server_type in self.mcp_sessions:
            return  # Already connected
            
        ready_event = asyncio.Event()
        
        async def mcp_session_runner() -> None:
            try:
                env = {"OPENAI_API_KEY": api_key} if api_key else None
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
                await asyncio.wait_for(session.initialize(), timeout=self.connection_timeout)
                
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
                
                response = await asyncio.wait_for(session.list_tools(), timeout=10)
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
        
    async def initialize_executor(self, server_type: str, **kwargs) -> dict:
        session = self.mcp_sessions.get(server_type)
        if not session:
            raise RuntimeError(f"{server_type.capitalize()} server not connected")
        try:
            result = await asyncio.wait_for(
                session.client.call_tool("initialize_executor", kwargs),
                timeout=10
            )
            content = json.loads(result.content[0].text) if result.content else {}
            return content
        except asyncio.TimeoutError:
            raise RuntimeError(f"{server_type.capitalize()} executor initialization timeout after 10s")
        except Exception as e:
            raise RuntimeError(f"{server_type.capitalize()} executor initialization failed: {str(e)}")
        
    async def call_tool(self, server_type: str, tool_name: str, tool_args: dict, timeout: int = 60) -> Any:
        session = self.mcp_sessions.get(server_type)
        if not session:
            raise RuntimeError(f"{server_type.capitalize()} server not connected")
        try:
            result = await asyncio.wait_for(session.client.call_tool(tool_name, tool_args), timeout=timeout)
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
            

class VerifierAgent:
    def __init__(self, 
                 mode: str, 
                 vision_model: str, 
                 api_key: str, 
                 thought_save: str, 
                 task_name: str,
                 max_rounds: int = 10, 
                 target_image_path: Optional[str] = None,
                 target_descirption: Optional[str] = None,
                 image_server_path: Optional[str] = None,
                 scene_server_path: Optional[str] = None,
                 api_base_url: Optional[str] = None):
        self.mode = mode
        self.vision_model = vision_model
        self.api_key = api_key
        # Support custom OpenAI-compatible base URL
        client_kwargs = {"api_key": self.api_key}
        if api_base_url or os.getenv("OPENAI_BASE_URL"):
            client_kwargs["base_url"] = api_base_url or os.getenv("OPENAI_BASE_URL")
        self.client = OpenAI(**client_kwargs)
        self.thought_save = thought_save
        os.makedirs(self.thought_save, exist_ok=True)
        self.max_rounds = max_rounds
        self.target_image_path = None
        self.current_image_path = None
        self.current_round = 0
        self.tool_client = ExternalToolClient()
        self._tools_connected = False
        
        if mode == "blendergym" or mode == "autopresent":
            self.server_type = "image"
            self.server_path = image_server_path
        elif mode == "blendergym-hard" or mode == "demo":
            self.server_type = "scene"
            self.server_path = scene_server_path
        else:
            raise NotImplementedError("Mode not implemented")
        
        if mode == "blendergym":
            self.memory = self._build_blendergym_system_prompt(
                mode, task_name, target_image_path
            )
        elif mode == "autopresent":
            self.memory = self._build_autopresent_system_prompt(
                mode, target_descirption
            )
        elif mode == "blendergym-hard" or mode == "demo":
            self.memory = self._build_blendergym_hard_system_prompt(
                mode, task_name, target_image_path
            )
        else:
            raise NotImplementedError("Mode not implemented")
        
    def _get_image_base64(self, image_path: str) -> str:
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
        b64 = base64.b64encode(img_byte_array.read()).decode('utf-8')
        return f"data:image/{mime_subtype};base64,{b64}"
    
    def _build_blendergym_hard_system_prompt(self, 
                             mode: str,
                             task_name: str,
                             target_image_path: str) -> List[Dict]:
        full_prompt = []
        # System prompt
        full_prompt.append({
            "role": "system",
            "content": prompts_dict[mode]['system']['verifier']
        })
        user_content = []
        
        # Add target image/description
        target_image_path_1 = os.path.join(target_image_path, 'visprompt1.png')
        if os.path.exists(target_image_path_1):
            self.target_image_path = os.path.abspath(target_image_path_1)
            user_content.extend([
                {"type": "text", "text": "Target Image (View 1):"},
                {"type": "image_url", "image_url": {"url": self._get_image_base64(target_image_path_1)}}
            ])

        # Add hints
        if mode == 'blendergym-hard':
            user_content.append({"type": "text", "text": f"Hints:\n{prompts_dict[mode]['hints'][task_name.split('-')[0]][task_name.split('-')[1]]}"})
        elif mode == 'demo':
            user_content.append({"type": "text", "text": f"Hints:\n{prompts_dict[mode]['hints']}"})
        else:
            raise NotImplementedError("Mode not implemented")
            
        full_prompt.append({"role": "user", "content": user_content})
        return full_prompt
    
    def _build_blendergym_system_prompt(self, 
                             mode: str,
                             task_name: str,
                             target_image_path: str) -> List[Dict]:
        full_prompt = []
        # System prompt
        full_prompt.append({
            "role": "system",
            "content": prompts_dict[mode]['system']['verifier']
        })
        user_content = []
        
        # Add target image/description
        target_image_path_1 = os.path.join(target_image_path, 'render1.png')
        if os.path.exists(target_image_path_1):
            self.target_image_path = os.path.abspath(target_image_path_1)
            user_content.extend([
                {"type": "text", "text": "Target Image (View 1):"},
                {"type": "image_url", "image_url": {"url": self._get_image_base64(target_image_path_1)}}
            ])
        target_image_path_2 = os.path.join(target_image_path, 'render2.png')
        if os.path.exists(target_image_path_2):
            user_content.extend([
                {"type": "text", "text": "Target Image (View 2):"},
                {"type": "image_url", "image_url": {"url": self._get_image_base64(target_image_path_2)}}
            ])
            
        # Add hints
        if prompts_dict[mode]['hints']['verifier'][task_name] is not None:
            user_content.append({"type": "text", "text": f"Hints:\n{prompts_dict[mode]['hints']['verifier'][task_name]}"})
            
        full_prompt.append({"role": "user", "content": user_content})
        return full_prompt
    
    def _build_autopresent_system_prompt(self, 
                             mode: str,
                             target_descirption: str) -> List[Dict]:
        full_prompt = []
        # System prompt
        full_prompt.append({
            "role": "system",
            "content": prompts_dict[mode]['system']['verifier']
        })
        user_content = []
        
        # Add target description
        user_content.append({
            "type": "text",
            "text": f"Task Instruction:\n{target_descirption}"
        })
        
        # Add hints
        if prompts_dict[mode]['hints'] is not None:
            user_content.append({"type": "text", "text": f"Hints:\n{prompts_dict[mode]['hints']}"})
            
        full_prompt.append({"role": "user", "content": user_content})
        return full_prompt
    
    async def _ensure_tools_connected(self):
        if not self._tools_connected:
            if self.server_type == "image":
                await self.tool_client.connect_server("image", self.server_path, self.api_key)
            elif self.server_type == "scene":
                await self.tool_client.connect_server("scene", self.server_path)
            self._tools_connected = True
            
    async def setup_executor(self, **kwargs):
        await self._ensure_tools_connected()
        if self.server_type == "image":
            result = await self.tool_client.initialize_executor("image", **kwargs)
            return result
        elif self.server_type == "scene":
            # Initialize scene investigator
            blender_path = kwargs.get("blender_path", None)
            save_dir = kwargs.get("save_dir", None)
            result = await self.tool_client.call_tool("scene", "initialize_investigator", {
                "thoughtprocess_save": save_dir,
                "blender_path": blender_path,
            })
            return result
        return {"status": "success", "message": "No executor setup needed for this mode."}
        
    async def verify_scene(self, code: str, render_path: str, round_num: int) -> Dict[str, Any]:
        await self._ensure_tools_connected()
        
        # define current round
        current_round = 0
        self.current_round = round_num
        
        # build memory
        if self.mode == "blendergym":
            verify_message = {"role": "user", "content": [{"type": "text", "text": f"Please analyze the current state:\nCode: {code}"}]}
            if os.path.isdir(render_path):
                view1_path = os.path.join(render_path, 'render1.png')
                view2_path = os.path.join(render_path, 'render2.png')
            else:
                view1_path = render_path
                view2_path = None
            scene_content = []
            if os.path.exists(view1_path):
                self.current_image_path = os.path.abspath(view1_path)
                scene_content.extend([
                    {"type": "text", "text": f"Current scene (View 1):"},
                    {"type": "image_url", "image_url": {"url": self._get_image_base64(view1_path)}}
                ])
            if os.path.exists(view2_path):
                scene_content.extend([
                    {"type": "text", "text": f"Current scene (View 2):"},
                    {"type": "image_url", "image_url": {"url": self._get_image_base64(view2_path)}}
                ])
            verify_message["content"].extend(scene_content)
            verify_message["content"].append({"type": "text", "text": prompts_dict[self.mode]['format']['verifier']})
            self.memory.append(verify_message)
        elif self.mode == "autopresent":
            verify_message = {"role": "user", "content": [{"type": "text", "text": f"Please analyze the current code and generated slide:\nCode: {code}"}]}
            # add slide screenshot
            if os.path.exists(render_path):
                verify_message["content"].append({"type": "text", "text": f"Generated slide screenshot:"})
                verify_message["content"].append({"type": "image_url", "image_url": {"url": self._get_image_base64(render_path)}})
            verify_message["content"].append({"type": "text", "text": prompts_dict[self.mode]['format']['verifier']})
            self.memory.append(verify_message)
        elif self.mode == "blendergym-hard" or self.mode == "demo":
            pass
        else:
            raise NotImplementedError("Mode not implemented")
        
        # start verification
        try:
            for i in range(self.max_rounds):
                response = self.client.chat.completions.create(
                    model=self.vision_model,
                    messages=self.memory,
                    tools=self._get_tools(),
                    tool_choice="auto"
                )
                message = response.choices[0].message
                self.memory.append(message.model_dump())
                # Handle tool calls
                if message.tool_calls:
                    for tool_call in message.tool_calls:
                        tool_response = await self._handle_tool_call(tool_call)
                        self.memory.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_call.function.name,
                            "content": tool_response['text']
                        })
                        if tool_response.get('image'):
                            self.memory.append({
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": "Generated image:"},
                                    {"type": "image_url", "image_url": {"url": self._get_image_base64(tool_response['image'])}}
                                ]
                            })
                else:
                    if "OK" in message.content and "Code Localization" not in message.content:
                        result = {"status": "end", "output": message.content}
                    else:
                        result = {"status": "continue", "output": message.content}
                    break
            current_round += 1
            self.save_thought_process()
            return result
        except Exception as e:
            logging.error(f"Verification failed: {e}")
            return {"status": "error", "error": str(e), "round": current_round}
        
    def _get_tools(self) -> List[Dict]:
        if self.mode == "blendergym" or self.mode == "autopresent":
            return [{
                "type": "function",
                "function": {
                    "name": "compare_images",
                    "description": "A tool for comparing current images and the target images, and identifying their visual differences. This tool will automatically select suitable images for comparison, please always call this tool first."
                }
            }]
        elif self.mode == "blendergym-hard" or self.mode == "demo":
            return [{
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
            }]
        
    async def _handle_tool_call(self, tool_call) -> Dict[str, Any]:
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)
        try:
            if function_name == "investigate_3d":
                op = function_args['operation']
                if op == 'focus':
                    output = await self.tool_client.call_tool("scene", "focus", {
                        "object_name": function_args.get("object_name", "")
                    })
                    return {'text': f"Focused camera on object: {function_args.get('object_name', '')}", 'image': output.get('image')}
                elif op == 'zoom':
                    output = await self.tool_client.call_tool("scene", "zoom", {
                        "direction": function_args.get("direction", "")
                    })
                    return {'text': f"Zoomed {function_args.get('direction', '')}", 'image': output.get('image')}
                elif op == 'move':
                    output = await self.tool_client.call_tool("scene", "move", {
                        "direction": function_args.get("direction", "")
                    })
                    return {'text': f"Moved camera {function_args.get('direction', '')}", 'image': output.get('image')}
                else:
                    return {'text': f"Unknown operation: {op}", 'image': None}
            elif function_name == "compare_images":
                output = await self.tool_client.call_tool("image", "compare_images", {
                    "path1": self.current_image_path,
                    "path2": self.target_image_path
                })
                return {'text': output.get('description', ''), 'image': None}
            else:
                return {'text': f"Unknown tool: {function_name}", 'image': None}
        except Exception as e:
            return {'text': f"Error executing tool: {str(e)}", 'image': None}
        
    def save_thought_process(self):
        try:
            with open(self.thought_save + f"/{self.current_round+1}.json", "w") as f:
                json.dump(self.memory, f, indent=4, ensure_ascii=False)
        except Exception as e:
            logging.error(f"Failed to save thought process: {e}")
            
    async def cleanup(self):
        await self.tool_client.cleanup()

def main():
    mcp = FastMCP("verifier")
    agent_holder = {}
    
    @mcp.tool()
    async def initialize_verifier(
        mode: str,
        vision_model: str,
        api_key: str,
        thought_save: str,
        task_name: str,
        max_rounds: int = 10,
        target_image_path: Optional[str] = None,
        target_descirption: Optional[str] = None,
        image_server_path: Optional[str] = None,
        scene_server_path: Optional[str] = None,
        blender_save: Optional[str] = None, # The new file, we cover it each verification step
        api_base_url: Optional[str] = None,
    ) -> dict:
        
        try:
            agent = VerifierAgent(
                mode=mode,
                vision_model=vision_model,
                api_key=api_key,
                thought_save=thought_save,
                task_name=task_name,
                max_rounds=max_rounds,
                target_image_path=target_image_path,
                target_descirption=target_descirption,
                image_server_path=image_server_path,
                scene_server_path=scene_server_path,
                api_base_url=api_base_url
            )
            agent_holder['agent'] = agent
            # Initialize server executor
            if mode == "blendergym" or mode == "autopresent":
                setup_result = await agent.setup_executor(vision_model=vision_model, api_key=api_key, api_base_url=api_base_url)
                if setup_result.get("status") != "success":
                    return {"status": "error", "error": f"Image server setup failed: {setup_result.get('error', setup_result)}"}
            elif mode == "blendergym-hard":
                setup_result = await agent.setup_executor(blender_path=blender_save, save_dir=thought_save)
                if setup_result.get("status") != "success":
                    return {"status": "error", "error": f"Scene server setup failed: {setup_result.get('error', setup_result)}"}
            await agent._ensure_tools_connected()
            return {"status": "success", "message": "Verifier Agent initialized and tool servers connected"}
        except Exception as e:
            return {"status": "error", "error": str(e)}
        
    @mcp.tool()
    async def verify_scene(code: str, render_path: str, round_num: int) -> dict:
        try:
            agent = agent_holder['agent']
            result = await agent.verify_scene(code, render_path, round_num)
            return result
        except Exception as e:
            return {"status": "error", "error": str(e)}
        
    @mcp.tool()
    def save_thought_process() -> dict:
        try:
            agent = agent_holder['agent']
            agent.save_thought_process()
            return {"status": "success", "message": "Thought process saved successfully"}
        except Exception as e:
            return {"status": "error", "error": str(e)}
        
    @mcp.tool()
    async def cleanup_verifier() -> dict:
        try:
            agent = agent_holder['agent']
            await agent.cleanup()
            return {"status": "success", "message": "Verifier Agent cleaned up successfully"}
        except Exception as e:
            return {"status": "error", "error": str(e)}
        
    mcp.run(transport="stdio")

if __name__ == "__main__":
    main() 