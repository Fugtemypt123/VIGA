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

class ExternalToolClient:
    """Client for connecting to external MCP tool servers."""
    
    def __init__(self):
        self.blender_session = None
        self.slides_session = None
        self.exit_stack = AsyncExitStack()
        self.connection_timeout = 30  # 30 seconds timeout
    
    async def connect_blender_server(self, blender_server_path: str):
        """Connect to the Blender execution MCP server with timeout."""
        try:
            # Add timeout to prevent hanging
            server_params = StdioServerParameters(
                command="python",
                args=[blender_server_path],
                env=None
            )
            
            # Use asyncio.wait_for to add timeout
            stdio_transport = await asyncio.wait_for(
                self.exit_stack.enter_async_context(stdio_client(server_params)),
                timeout=self.connection_timeout
            )
            stdio, write = stdio_transport
            
            self.blender_session = await asyncio.wait_for(
                self.exit_stack.enter_async_context(ClientSession(stdio, write)),
                timeout=self.connection_timeout
            )
            
            await asyncio.wait_for(
                self.blender_session.initialize(),
                timeout=self.connection_timeout
            )
            
            # List available tools
            response = await asyncio.wait_for(
                self.blender_session.list_tools(),
                timeout=10  # Shorter timeout for tool listing
            )
            tools = response.tools
            print(f"Connected to Blender server with tools: {[tool.name for tool in tools]}")
            
        except asyncio.TimeoutError:
            raise RuntimeError(f"Failed to connect to Blender server: Connection timeout after {self.connection_timeout}s")
        except Exception as e:
            raise RuntimeError(f"Failed to connect to Blender server: {str(e)}")
    
    async def connect_slides_server(self, slides_server_path: str):
        """Connect to the Slides execution MCP server with timeout."""
        try:
            # Add timeout to prevent hanging
            server_params = StdioServerParameters(
                command="python",
                args=[slides_server_path],
                env=None
            )
            
            # Use asyncio.wait_for to add timeout
            stdio_transport = await asyncio.wait_for(
                self.exit_stack.enter_async_context(stdio_client(server_params)),
                timeout=self.connection_timeout
            )
            stdio, write = stdio_transport
            
            self.slides_session = await asyncio.wait_for(
                self.exit_stack.enter_async_context(ClientSession(stdio, write)),
                timeout=self.connection_timeout
            )
            
            await asyncio.wait_for(
                self.slides_session.initialize(),
                timeout=self.connection_timeout
            )
            
            # List available tools
            response = await asyncio.wait_for(
                self.slides_session.list_tools(),
                timeout=10  # Shorter timeout for tool listing
            )
            tools = response.tools
            print(f"Connected to Slides server with tools: {[tool.name for tool in tools]}")
            
        except asyncio.TimeoutError:
            raise RuntimeError(f"Failed to connect to Slides server: Connection timeout after {self.connection_timeout}s")
        except Exception as e:
            raise RuntimeError(f"Failed to connect to Slides server: {str(e)}")
    
    async def initialize_blender_executor(self, blender_command: str, blender_file: str, 
                                        blender_script: str, script_save: str, 
                                        render_save: str, blender_save: Optional[str] = None) -> Dict:
        """Initialize the Blender executor using external server with timeout."""
        if not self.blender_session:
            raise RuntimeError("Blender server not connected")
        
        try:
            result = await asyncio.wait_for(
                self.blender_session.call_tool("initialize_executor", {
                    "blender_command": blender_command,
                    "blender_file": blender_file,
                    "blender_script": blender_script,
                    "script_save": script_save,
                    "render_save": render_save,
                    "blender_save": blender_save
                }),
                timeout=30  # 30 seconds for initialization
            )
            content = json.loads(result.content[0].text) if result.content else {}
            return content
        except asyncio.TimeoutError:
            raise RuntimeError("Blender executor initialization timeout after 30s")
        except Exception as e:
            raise RuntimeError(f"Blender executor initialization failed: {str(e)}")
    
    async def exec_blender_script(self, code: str, round_num: int) -> Dict:
        """Execute Blender script using external server with timeout."""
        if not self.blender_session:
            raise RuntimeError("Blender server not connected")
        
        try:
            result = await asyncio.wait_for(
                self.blender_session.call_tool("exec_script", {
                    "code": code,
                    "round": round_num
                }),
                timeout=60  # 60 seconds for script execution
            )
            content = json.loads(result.content[0].text) if result.content else {}
            return content
        except asyncio.TimeoutError:
            raise RuntimeError("Blender script execution timeout after 60s")
        except Exception as e:
            raise RuntimeError(f"Blender script execution failed: {str(e)}")
    
    async def exec_slides_script(self, code: str, round_num: int, code_save: str) -> Dict:
        """Execute Slides script using external server with timeout."""
        if not self.slides_session:
            raise RuntimeError("Slides server not connected")
        
        try:
            result = await asyncio.wait_for(
                self.slides_session.call_tool("exec_pptx", {
                    "code": code,
                    "round": round_num,
                    "code_save": code_save
                }),
                timeout=60  # 60 seconds for script execution
            )
            content = json.loads(result.content[0].text) if result.content else {}
            return content
        except asyncio.TimeoutError:
            raise RuntimeError("Slides script execution timeout after 60s")
        except Exception as e:
            raise RuntimeError(f"Slides script execution failed: {str(e)}")
    
    async def cleanup(self):
        """Clean up connections with timeout."""
        try:
            await asyncio.wait_for(self.exit_stack.aclose(), timeout=10)
        except asyncio.TimeoutError:
            logging.warning("Cleanup timeout, forcing close")
        except Exception as e:
            logging.warning(f"Cleanup error: {e}")

class GeneratorAgent:
    """
    An MCP agent that takes code modification suggestions and implements them.
    This agent follows the MCP server pattern for better encapsulation and tool integration.
    """
    
    def __init__(self, 
                 vision_model: str,
                 api_key: str,
                 thought_save: str,
                 max_rounds: int = 10,
                 generator_hints: Optional[str] = None,
                 init_code: Optional[str] = None,
                 init_image_path: Optional[str] = None,
                 target_image_path: Optional[str] = None,
                 target_description: Optional[str] = None,
                 blender_server_path: Optional[str] = None,
                 slides_server_path: Optional[str] = None):
        """
        Initialize the Generator Agent.
        
        Args:
            vision_model: The OpenAI vision model to use
            api_key: OpenAI API key
            thought_save: Path to save thought process
            max_rounds: Maximum number of generation rounds
            generator_hints: Hints for code generation
            init_code: Initial code to modify
            init_image_path: Path to initial images
            target_image_path: Path to target images
            target_description: Description of target
            blender_server_path: Path to external Blender MCP server
            slides_server_path: Path to external Slides MCP server
        """
        self.model = vision_model
        self.api_key = api_key
        self.client = OpenAI(api_key=self.api_key)
        self.thought_save = thought_save
        self.max_rounds = max_rounds
        self.memory = []
        self.current_round = 0
        
        # Tool execution setup
        self.tool_client = ExternalToolClient()
        self._blender_connected = False
        self._slides_connected = False
        self.blender_config = {}
        self.slides_config = {}
        
        # Initialize memory if initial parameters are provided
        if all([init_code, init_image_path, target_image_path]):
            self.memory = self._build_system_prompt(
                generator_hints, init_code, init_image_path, 
                target_image_path, target_description
            )
    
    async def _ensure_blender_connected(self):
        """Ensure Blender server is connected."""
        if not self._blender_connected and self.blender_server_path:
            await self.tool_client.connect_blender_server(self.blender_server_path)
            self._blender_connected = True
    
    async def _ensure_slides_connected(self):
        """Ensure Slides server is connected."""
        if not self._slides_connected and self.slides_server_path:
            await self.tool_client.connect_slides_server(self.slides_server_path)
            self._slides_connected = True
    
    async def setup_blender_executor(self, 
                                     blender_server_path: str, 
                                     blender_command: str, 
                                     blender_file: str,
                                     blender_script: str, 
                                     script_save: str,
                                     render_save: str, 
                                     blender_save: Optional[str] = None):
        """Setup the Blender executor with configuration."""
        await self._ensure_blender_connected(blender_server_path)
        
        self.blender_config = {
            "blender_command": blender_command,
            "blender_file": blender_file,
            "blender_script": blender_script,
            "script_save": script_save,
            "render_save": render_save,
            "blender_save": blender_save
        }
        
        result = await self.tool_client.initialize_blender_executor(**self.blender_config)
        return result
    
    async def setup_slides_executor(self, code_save: str):
        """Setup the Slides executor with configuration."""
        await self._ensure_slides_connected()
        
        self.slides_config = {
            "code_save": code_save
        }
        
        return {"status": "success", "message": "Slides executor configured"}
    
    def _build_system_prompt(self, hints: str, init_code: str, init_image_path: str, 
                           target_image_path: str, target_description: str = None) -> List[Dict]:
        """
        Build the system prompt for the generator.
        """
        full_prompt = []
        # Add system prompt
        full_prompt.append({
            "role": "system",
            "content": """You are a code generation agent proficient in Blender Python scripting. Your task is to edit code to transform an initial 3D scene into a target scene. After each code edit, your code will be passed to a validator, which will provide feedback on the result. Based on this feedback, you must iteratively refine your code edits. This process will continue across multiple rounds of dialogue. In each round, you must adhere to a fixed output format."""
        })
        
        # Add initial code & code analysis
        user_content = [{
            "type": "text",
            "text": f"Initial Code:\n```python\n{init_code}\n```"
        }]
        
        code_analysis = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a Blender Python code analysis expert."},
                {"role": "user", "content": f"Please analyze the following Blender Python code line by line, explaining what each part does and how it contributes to the scene:\n```python\n{init_code}\n```"}
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
            logging.error(f"Initial image {init_image_path_1} does not exist!")
        
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
        
        # Add target images
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
        if hints is not None:
            user_content.append({
                "type": "text",
                "text": f"Hints:\n{hints}"
            })
        
        # Add output format
        user_content.append({
            "type": "text",
            "text": """After each code edit, your code will be passed to a validator, which will provide feedback on the result. Based on this feedback, you must iteratively refine your code edits. This process will continue across multiple rounds of dialogue. In each round, you must follow a fixed output format. Output Format (keep this format for each round):
1. Thought: Analyze the current state and provide a clear plan for the required changes.
2. Code Edition: Provide your code modifications in the following format:
   -: [lines to remove]
   +: [lines to add]
3. Full Code: Merge your code changes into the full code:
```python
[full code]
```"""
        })
        
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
            if self.blender_config and self._blender_connected:
                try:
                    execution_result = await self.tool_client.exec_blender_script(full_code, self.current_round)
                    logging.info(f"Blender execution completed for round {self.current_round}")
                except Exception as e:
                    logging.error(f"Blender execution failed: {e}")
                    execution_result = {"status": "error", "error": str(e)}
            elif self.slides_config and self._slides_connected:
                try:
                    execution_result = await self.tool_client.exec_slides_script(
                        full_code, self.current_round, self.slides_config["code_save"]
                    )
                    logging.info(f"Slides execution completed for round {self.current_round}")
                except Exception as e:
                    logging.error(f"Slides execution failed: {e}")
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
        vision_model: str,
        api_key: str,
        thought_save: str,
        task_name: str,
        max_rounds: int = 10,
        init_code_path: str = None,
        init_image_path: str = None,
        target_image_path: str = None,
        target_description: str = None,
        # Blender executor parameters
        blender_server_path: str = None,
        blender_command: str = None,
        blender_file: str = None,
        blender_script: str = None,
        script_save: str = None,
        render_save: str = None,
        blender_save: str = None,
        # Slides executor parameters
        slides_server_path: str = None,
        code_save: str = None
        
    ) -> dict:
        """
        Initialize a new Generator Agent with optional Blender or Slides executor setup.
        """
        try:
            agent = GeneratorAgent(
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
                slides_server_path=slides_server_path
            )
            agent_holder['agent'] = agent
            
            # Determine which executor to setup based on provided parameters
            blender_setup = all([blender_command, blender_file, blender_script, script_save, render_save])
            slides_setup = code_save is not None
            
            setup_results = []
            
            # Setup Blender executor if parameters are provided
            if blender_setup:
                try:
                    setup_result = await agent.setup_blender_executor(
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
            if slides_setup:
                try:
                    setup_result = await agent.setup_slides_executor(code_save=code_save)
                    setup_results.append(("Slides", setup_result))
                except Exception as e:
                    setup_results.append(("Slides", {"status": "error", "error": str(e)}))
            
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