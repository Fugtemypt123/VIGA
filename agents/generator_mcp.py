import asyncio
import os
import json
from openai import OpenAI
from typing import Dict, List, Optional, Any
import logging
from mcp.server.fastmcp import FastMCP
from prompts import prompts_dict
from agents.external_tool_client import ExternalToolClient
from agents.prompt_builder import PromptBuilder
from agents.tool_manager import ToolManager
from agents.tool_handler import ToolHandler
from agents.utils import parse_generate_response, get_blendergym_hard_level, save_thought_process

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
        elif mode == "autopresent":
            self.server_type = "slides"
            self.server_path = slides_server_path
        else:
            raise NotImplementedError("Mode not implemented")
        
        # Initialize prompt builder and tool handler
        self.prompt_builder = PromptBuilder(self.client)
        self.tool_handler = ToolHandler(self.tool_client, self.server_type)
        
        # Initialize memory if initial parameters are provided
        if mode == "blendergym":
            self.memory = self.prompt_builder.build_blendergym_generator_prompt(mode, task_name, init_code_path, init_image_path, target_image_path)
        elif mode == "autopresent":
            self.memory = self.prompt_builder.build_autopresent_generator_prompt(mode, init_code_path, init_image_path, target_description)
        elif mode == "blendergym-hard":
            self.memory = self.prompt_builder.build_blendergym_hard_generator_prompt(mode, task_name, init_code_path, init_image_path, target_image_path)
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
            # Update tool handler with blender file path
            self.tool_handler.blender_file_path = self.blender_file_path
            
            # Initialize investigator for blendergym-hard
            if self.mode == "blendergym-hard":
                try:
                    investigator_result = await self.tool_client.call_tool("blender", "initialize_investigator", {"blender_path": kwargs["blender_file"]})
                    if investigator_result.get("status") == "success":
                        logging.info("Investigator initialized successfully")
                    else:
                        logging.warning(f"Investigator initialization failed: {investigator_result.get('error')}")
                except Exception as e:
                    logging.warning(f"Failed to initialize investigator: {e}")
        
        return result
    
    def _get_tools(self) -> List[Dict]:
        """Get available tools for the generator agent."""
        return ToolManager.get_generator_tools(self.mode, self.task_name)
    
    async def _handle_tool_call(self, tool_call) -> Dict[str, Any]:
        """Handle tool calls from the generator agent."""
        return await self.tool_handler.handle_generator_tool_call(tool_call)

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
            use_tools = self.mode == "blendergym-hard" and self._server_connected
            
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
                                        {"type": "image_url", "image_url": {"url": self.prompt_builder._get_image_base64(tool_response['image'])}}
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
            try:
                _, _, full_code = parse_generate_response(generate_response)
            except:
                full_code = ""
            
            # Auto-execute code if it contains "Full Code" and we're in a mode that supports code execution
            execution_result = None
            try:
                self.current_round += 1
                execution_result = await self.tool_handler.execute_script(
                    code=full_code,
                    round_num=self.current_round,
                )
                logging.info(f"Auto-executed code for round {self.current_round}")
            except Exception as e:
                logging.error(f"Failed to auto-execute code: {e}")
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
        save_thought_process(self.memory, self.thought_save)
    
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
