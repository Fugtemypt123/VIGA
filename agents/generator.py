import asyncio
import os
import json
from openai import OpenAI
from typing import Dict, List, Optional, Any
import logging
import copy
from mcp.server.fastmcp import FastMCP
from agents.external_tool_client import ExternalToolClient
from agents.prompt_builder import PromptBuilder
from agents.tool_manager import ToolManager
from agents.tool_handler import ToolHandler
from agents.utils import parse_generate_response, get_blendergym_hard_level, save_thought_process, get_scene_info, get_image_base64

class GeneratorAgent:
    """
    An MCP agent that takes code modification suggestions and implements them.
    This agent follows the MCP server pattern for better encapsulation and tool integration.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the Generator Agent.
        """
        self.config = kwargs
        
        # Extract commonly used parameters
        self.mode = self.config.get("mode")
        self.model = self.config.get("vision_model")
        self.api_key = self.config.get("api_key")
        self.task_name = self.config.get("task_name")
        self.max_rounds = self.config.get("max_rounds", 10)
        self.current_round = 0
        
        # Handle blendergym-hard level detection
        if self.mode == "blendergym-hard":
            self.level = get_blendergym_hard_level(self.task_name)
        else:
            self.level = None
        
        # Initialize OpenAI client
        api_base_url = self.config.get("api_base_url")
        client_kwargs = {"api_key": self.api_key}
        if api_base_url or os.getenv("OPENAI_BASE_URL"):
            client_kwargs["base_url"] = api_base_url or os.getenv("OPENAI_BASE_URL")
        self.client = OpenAI(**client_kwargs)
        
        # Initialize components
        self.tool_client = ExternalToolClient()
        self._server_connected = False
        
        # Determine server type and path
        if self.mode == "blendergym" or self.mode == "blendergym-hard":
            self.server_type = "blender"
            self.server_path = self.config.get("blender_server_path")
        elif self.mode == "autopresent":
            self.server_type = "slides"
            self.server_path = self.config.get("slides_server_path")
        elif self.mode == "design2code":
            self.server_type = "html"
            self.server_path = self.config.get("html_server_path")
        else:
            raise NotImplementedError("Mode not implemented")
        
        # Initialize prompt builder and tool handler
        self.prompt_builder = PromptBuilder(self.client, self.model)
        self.tool_handler = ToolHandler(self.tool_client, self.server_type)
        
        # Initialize memory using generic prompt builder
        self.system_prompt = self.prompt_builder.build_generator_prompt(self.config)
        self.memory = copy.deepcopy(self.system_prompt)
    
    async def _ensure_server_connected(self):
        if not self._server_connected and self.server_type and self.server_path:
            await self.tool_client.connect_server(self.server_type, self.server_path)
            self._server_connected = True
    
    async def setup_executor(self, **kwargs):
        await self._ensure_server_connected()
        result = await self.tool_client.initialize_executor(self.server_type, **kwargs)
        return result
    
    def _get_tools(self) -> List[Dict]:
        """Get available tools for the generator agent."""
        return ToolManager.get_generator_tools(self.mode, self.task_name)
    
    async def _handle_tool_call(self, tool_call) -> Dict[str, Any]:
        """Handle tool calls from the generator agent."""
        return await self.tool_handler.handle_generator_tool_call(tool_call)

    async def call(self, no_memory: bool = False) -> Dict[str, Any]:
        """
        Generate code based on current memory and optional feedback.
        
        Args:
            feedback: Optional feedback from verifier or executor
            
        Returns:
            Dict containing the generated code and metadata
        """
        if no_memory:
            self.memory = copy.deepcopy(self.system_prompt)
            
        if self.mode == "blendergym-hard" and self.level == "level4":
            self.memory.append({"role": "user", "content": get_scene_info(self.task_name, self.config.get("blender_file_path"))})
        
        try:
            chat_args = {
                "model": self.model,
                "messages": self.memory,
                "tools": self._get_tools()
            }
            if chat_args['tools']:
                if self.model == 'gpt-4o':
                    chat_args['parallel_tool_calls'] = False
                if self.model != 'Qwen2-VL-7B-Instruct':
                    chat_args['tool_choice'] = "auto"

            response = self.client.chat.completions.create(
                model=self.model, 
                messages=self.memory, 
                tools=self._get_tools()
            )
            message = response.choices[0].message
            self.memory.append(message.model_dump())
            
            last_full_code = self.config.get("script_save") + f"/{self.current_round}.py"
            
            if message.tool_calls:
                for i, tool_call in enumerate(message.tool_calls):
                    if i > 0:
                        self.memory.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_call.function.name,
                            "content": "You can only call a tool once per conversation round."
                        })
                        continue
                    tool_response = await self._handle_tool_call(tool_call)
                    self.memory.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.function.name,
                        "content": tool_response['text']
                    })
                    full_code = open(last_full_code).read()
                    # Add output content if available from tool response
                    if tool_response.get('output_content'):
                        full_code += tool_response['output_content']
                    
            else:
                # Parse the response to extract code if needed (only for modes that generate code)
                _, _, full_code = parse_generate_response(message.content)
                
                # If the full code is None, just copy the init code
                if full_code is None:
                    full_code = open(last_full_code).read()
            
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
                "response": message.content,
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
        if os.path.isdir(feedback):
            feedback = get_image_base64(os.path.join(feedback, 'render1.png'))
            self.memory.append({"role": "user", "content": [{"type": "text", "text": "Generated image:"}, {"type": "image_url", "image_url": {"url": feedback}}]})
        elif os.path.isfile(feedback):
            feedback = get_image_base64(feedback)
            self.memory.append({"role": "user", "content": [{"type": "text", "text": "Generated image:"}, {"type": "image_url", "image_url": {"url": feedback}}]})
        else:
            feedback = [{"type": "text", "text": feedback}]
            self.memory.append({"role": "user", "content": feedback})
    
    def save_thought_process(self) -> None:
        """Save the current thought process to file."""
        save_thought_process(self.memory, self.config.get("thought_save"))
    
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
    async def initialize_generator(args: dict) -> dict:
        """
        Initialize a new Generator Agent with optional Blender or Slides executor setup.
        """
        try:
            agent = GeneratorAgent(**args)
            agent_holder['agent'] = agent
            setup_results = []
            
            # Setup executor based on mode
            if args.get("mode") == "blendergym" or args.get("mode") == "blendergym-hard":
                try:
                    setup_result = await agent.setup_executor(**args)
                    setup_results.append(("Blender", setup_result))
                except Exception as e:
                    setup_results.append(("Blender", {"status": "error", "error": str(e)}))
            
            elif args.get("mode") == "autopresent":
                try:
                    # Add task_dir from init_code_path
                    setup_kwargs = args.copy()
                    setup_kwargs["task_dir"] = os.path.dirname(args.get("init_code_path"))
                    setup_result = await agent.setup_executor(**setup_kwargs)
                    setup_results.append(("Slides", setup_result))
                except Exception as e:
                    setup_results.append(("Slides", {"status": "error", "error": str(e)}))
            
            elif args.get("mode") == "design2code":
                try:
                    setup_result = await agent.setup_executor(**args)
                    setup_results.append(("HTML", setup_result))
                except Exception as e:
                    setup_results.append(("HTML", {"status": "error", "error": str(e)}))
            
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
    async def call(no_memory: bool = False) -> dict:
        """
        Generate code using the initialized Generator Agent.
        """
        try:
            if 'agent' not in agent_holder:
                return {"status": "error", "error": "Generator Agent not initialized. Call initialize_generator first."}
            result = await agent_holder['agent'].call(no_memory=no_memory)
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
