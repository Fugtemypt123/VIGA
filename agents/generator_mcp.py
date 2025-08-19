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
        if mode == "blendergym":
            self.server_type = "blender"
            self.server_path = blender_server_path
        elif mode == "autopresent":
            self.server_type = "slides"
            self.server_path = slides_server_path
        else:
            raise NotImplementedError("Mode not implemented")
        
        # Initialize memory if initial parameters are provided
        if mode == "blendergym":
            self.memory = self._build_blendergym_system_prompt(
                mode, task_name, init_code_path, init_image_path, target_image_path
            )
        elif mode == "autopresent":
            self.memory = self._build_autopresent_system_prompt(
                mode, init_code_path, init_image_path, target_description
            )
    
    async def _ensure_server_connected(self):
        if not self._server_connected and self.server_type and self.server_path:
            await self.tool_client.connect_server(self.server_type, self.server_path)
            self._server_connected = True
    
    async def setup_executor(self, **kwargs):
        await self._ensure_server_connected()
        result = await self.tool_client.initialize_executor(self.server_type, **kwargs)
        return result
    
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
        full_prompt.append({
            "role": "system",
            "content": prompts_dict[mode]['system']['generator']
        })
        
        # Add initial code & code analysis
        init_code = open(init_code_path, 'r').read()
        user_content = [{
            "type": "text",
            "text": f"Initial Code:\n```python\n{init_code}\n```"
        }]
        
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
        user_content.append({
            "type": "text",
            "text": f"Code Analysis:\n{code_analysis}"
        })
        
        # Add initial images
        init_image_path_1 = os.path.join(init_image_path, 'render1.png')
        if os.path.exists(init_image_path_1):
            user_content.append({
                "type": "text",
                "text": "Initial Image (View 1):"
            })
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{self._get_image_base64(init_image_path_1)}"}
            })
        else:
            # At least we need one initial image
            raise ValueError(f"Initial image {init_image_path_1} does not exist!")
        
        init_image_path_2 = os.path.join(init_image_path, 'render2.png')
        if os.path.exists(init_image_path_2):
            user_content.append({
                "type": "text",
                "text": "Initial Image (View 2):"
            })
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{self._get_image_base64(init_image_path_2)}"}
            })
        
        # Add target images (for mode `blendergym`)
        target_image_path_1 = os.path.join(target_image_path, 'render1.png')
        if os.path.exists(target_image_path_1):
            user_content.append({
                "type": "text",
                "text": "Target Image (View 1):"
            })
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{self._get_image_base64(target_image_path_1)}"}
            })
        else:
            logging.error(f"Target image {target_image_path_1} does not exist!")
        
        target_image_path_2 = os.path.join(target_image_path, 'render2.png')
        if os.path.exists(target_image_path_2):
            user_content.append({
                "type": "text",
                "text": "Target Image (View 2):"
            })
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{self._get_image_base64(target_image_path_2)}"}
            })
        
        # Add hints 
        if prompts_dict[mode]['hints']['generator'][task_name] is not None:
            user_content.append({
                "type": "text",
                "text": f"Hints:\n{prompts_dict[mode]['hints']['generator'][task_name]}"
            })
        
        # Add output format
        user_content.append({
            "type": "text",
            "text": prompts_dict[mode]['format']['generator']
        })
        
        # Add all user content
        full_prompt.append({
            "role": "user",
            "content": user_content
        })
        return full_prompt
    
    def _build_autopresent_system_prompt(self, 
                             mode: str, 
                             init_code_path: str = None,
                             init_image_path: str = None, 
                             target_description: str = None) -> List[Dict]:
        full_prompt = []
        # Add system prompt
        full_prompt.append({
            "role": "system",
            "content": prompts_dict[mode]['system']['generator'] + '\n\n' + prompts_dict[mode]['api_library']
        })
        
        # Add user input
        user_content = []
        user_content.append({
            "type": "text",
            "text": f"Now, here is the task package, which includes the initial code, a screenshot of the initial slides, the provided images with filenames used in the slides, and my instruction:"
        })
        
        # Add initial code
        init_code = open(init_code_path, 'r').read()
        user_content.append({
            "type": "text",
            "text": f"Initial Code:\n```python\n{init_code}\n```"
        })
        
        # Add initial images
        if os.path.exists(init_image_path):
            user_content.append({
                "type": "text",
                "text": "Initial Slide Screenshot:"
            })
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{self._get_image_base64(init_image_path)}"}
            })
        else:
            user_content.append({
                "type": "text",
                "text": "Initial code cannot be executed, please check the code and fix the errors."
            })
            
        # Add used images
        user_content.append({
            "type": "text",
            "text": "Provided Images (they might already appear in the code):"
        })
        used_image_dir = os.path.join(os.path.dirname(init_image_path), 'media')
        used_images = os.listdir(used_image_dir)
        for image in used_images:
            if image.endswith('.jpg') or image.endswith('.png') or image.endswith('.jpeg'):
                user_content.append({
                    "type": "text",
                    "text": f"Path: {os.path.join('media', image)}"
                })
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{self._get_image_base64(os.path.join(used_image_dir, image))}"}
                })
        
        # Add target description
        user_content.append({
            "type": "text",
            "text": f"Instruction:\n{target_description}"
        })
        
        # Add hints
        user_content.append({
            "type": "text",
            "text": f"Hints:\n{prompts_dict[mode]['hints']}"
        })
        
        # Add output format
        user_content.append({
            "type": "text",
            "text": prompts_dict[mode]['format']['generator']
        })
        
        # Add all user content
        full_prompt.append({
            "role": "user",
            "content": user_content
        })
        return full_prompt
    
    def _get_image_base64(self, image_path: str) -> str:
        """Convert image to base64 string."""
        image = Image.open(image_path)
        img_byte_array = io.BytesIO()
        image.save(img_byte_array, format='PNG')
        img_byte_array.seek(0) 
        base64enc_image = base64.b64encode(img_byte_array.read()).decode('utf-8') 
        return base64enc_image
    
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
            generate_response = self.client.chat.completions.create(
                model=self.model, 
                messages=self.memory
            ).choices[0].message.content
            
            _, _, full_code = self._parse_generate_response(generate_response)
            self.memory.append({"role": "assistant", "content": generate_response})
            
            self.current_round += 1
            
            # Automatically execute the generated code with configured executor
            execution_result = None
            if self._server_connected:
                try:
                    execution_result = await self.tool_client.exec_script(
                        server_type=self.server_type,
                        code=full_code,
                        round_num=self.current_round,
                    )
                    logging.info(f"{self.server_type.capitalize()} execution completed for round {self.current_round}")
                except Exception as e:
                    logging.error(f"{self.server_type.capitalize()} execution failed: {e}")
                    execution_result = {"status": "error", "error": str(e)}
            
            return {
                "status": "success",
                "code": full_code,
                "response": generate_response,
                "round": self.current_round,
                "execution_result": execution_result
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
            if mode == "blendergym":
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