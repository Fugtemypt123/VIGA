import asyncio
import os
import json
from openai import OpenAI
from typing import Dict, List, Optional, Any
import logging
import copy
from mcp.server.fastmcp import FastMCP
from agents.tool_client import ExternalToolClient
from agents.prompt_builder import PromptBuilder
from agents.tool_manager import ToolManager
from agents.tool_handler import ToolHandler
from agents.config_manager import ConfigManager
from agents.utils import save_thought_process, get_scene_info, get_image_base64

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
        
        # Initialize configuration manager
        self.config_manager = ConfigManager(kwargs)
        
        # Validate configuration
        is_valid, error_message = self.config_manager.validate_generator_configuration()
        if not is_valid:
            raise ValueError(f"Invalid configuration: {error_message}")
        
        # Extract commonly used parameters from config manager
        self.mode = self.config_manager.mode
        self.model = self.config_manager.vision_model
        self.api_key = self.config_manager.api_key
        self.task_name = self.config_manager.task_name
        self.max_rounds = self.config_manager.max_rounds
        self.current_round = 0
        self.level = self.config_manager.level
        
        # Initialize OpenAI client
        client_kwargs = {"api_key": self.api_key}
        if self.config_manager.api_base_url or os.getenv("OPENAI_BASE_URL"):
            client_kwargs["base_url"] = self.config_manager.api_base_url or os.getenv("OPENAI_BASE_URL")
        self.client = OpenAI(**client_kwargs)
        
        # Initialize components
        self.tool_client = ExternalToolClient()
        self._server_connected = False
        
        # Determine tool servers and pick a primary server type for execution tools
        self.tool_servers = self.config_manager.get_generator_tool_servers()
        with open('logs/generator.log', 'w') as f:
            f.write(f"tool_servers: {self.tool_servers}")
        if "blender" in self.tool_servers:
            self.server_type = "blender"
        elif "html" in self.tool_servers:
            self.server_type = "html"
        elif "slides" in self.tool_servers:
            self.server_type = "slides"
        else:
            # Fallback to any provided server key
            self.server_type = next(iter(self.tool_servers.keys()), None)
        
        # Initialize prompt builder and tool handler
        self.prompt_builder = PromptBuilder(self.client, self.model)
        self.tool_handler = ToolHandler(self.tool_client)
        
        # Initialize memory using generic prompt builder
        self.system_prompt = self.prompt_builder.build_generator_prompt(kwargs)
        self.memory = copy.deepcopy(self.system_prompt)
        self.conversation_history = []  # Store last 6 chats for sliding window

    async def _ensure_server_connected(self):
        if not self._server_connected:
            await self.tool_client.connect_servers(self.tool_servers, api_key=self.api_key)
            self._server_connected = True
    
    async def _setup_executor_internal(self, **kwargs):
        """Internal method to setup executor - merged into call_tool"""
        await self._ensure_server_connected()
        result = await self.tool_client.call_tool("initialize_executor", tool_args={"args": kwargs})
        return result
    
    def _build_sliding_window_memory(self, current_chat_content=None):
        """Build sliding window memory: system_prompt + [last 6 chats] + current chat"""
        memory = copy.deepcopy(self.system_prompt)
        
        # Add last 6 conversation exchanges
        for chat in self.conversation_history[-12:]:
            memory.append(chat)
        
        # Add current chat if provided
        if current_chat_content:
            memory.append(current_chat_content)
            
        return memory
    
    def _get_tools(self) -> List[Dict]:
        """Get available tools for the generator agent."""
        return ToolManager.get_generator_tools(self.mode, self.task_name)
    
    async def _handle_tool_call(self, tool_call) -> Dict[str, Any]:
        """Handle tool calls from the generator agent."""
        return await self.tool_handler.handle_generator_tool_call(tool_call)

    async def call(self, no_memory: bool = False) -> Dict[str, Any]:
        """
        Generate code based on current memory and optional feedback.
        Now enforces tool calling and returns verifier flag.
        
        Args:
            no_memory: If True, reset memory to system prompt only
            
        Returns:
            Dict containing the generated code, metadata, and verifier flag
        """
        # Setup executor if not connected (merged setup_executor into call_tool)
        if not self._server_connected and self.server_type:
            await self._setup_executor_internal(**self.config)
        
        # Build sliding window memory
        if no_memory:
            self.conversation_history = []
        
        # Add scene info for level4 if needed
        scene_info_content = None
        if self.config_manager.should_add_scene_info():
            scene_config = self.config_manager.get_scene_info_config()
            scene_info_content = {"role": "user", "content": get_scene_info(scene_config["task_name"], scene_config["blender_file"])}
        
        # Build memory with sliding window
        memory = self._build_sliding_window_memory(scene_info_content)
        
        try:
            max_attempts = 3
            for attempt in range(max_attempts):
                chat_args = {
                    "model": self.model,
                    "messages": memory,
                }
                tools = self._get_tools()
                if tools:
                    chat_args['tools'] = tools
                    if 'gpt' in self.model:
                        chat_args['parallel_tool_calls'] = False
                    if self.model != 'Qwen2-VL-7B-Instruct':
                        chat_args['tool_choice'] = "auto"

                response = self.client.chat.completions.create(**chat_args)
                message = response.choices[0].message
                
                # Store assistant message in conversation history
                self.conversation_history.append(message.model_dump())
                
                if message.tool_calls:
                    # Tool was called - this is what we want
                    tool_called = True
                    full_code = None
                    execution_result = None
                    call_verifier = False
                    
                    # Handle tool calls (only first one)
                    for i, tool_call in enumerate(message.tool_calls):
                        if i > 0:
                            self.conversation_history.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "name": tool_call.function.name,
                                "content": "You can only call a tool once per conversation round."
                            })
                            continue
                            
                        tool_response = await self._handle_tool_call(tool_call)
                        
                        # Store tool response in conversation history
                        self.conversation_history.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_call.function.name,
                            "content": tool_response['text']
                        })
                        
                        # Check if this was an execute_and_evaluate tool call
                        if tool_call.function.name == "execute_and_evaluate":
                            call_verifier = True
                            execution_result = tool_response.get('execution_result')
                            full_code = tool_response.get('full_code')
                            # Internally handle feedback and add to memory
                            try:
                                if execution_result:
                                    if execution_result.get("status") == "success":
                                        result_obj = execution_result.get("result", {})
                                        if result_obj.get("status") == "success":
                                            # Provide the render output (path/dir) back into memory
                                            await self.add_feedback(result_obj.get("output"))
                                        else:
                                            await self.add_feedback(f"Execution error: {result_obj.get('output')}")
                                    else:
                                        await self.add_feedback(f"Execution error: {execution_result.get('error', 'Unknown error')}")
                                else:
                                    await self.add_feedback("No execution result available. Please ensure you're calling the execute_and_evaluate tool.")
                            except Exception as e:
                                logging.error(f"Failed to add execution feedback to memory: {e}")
                        else:
                            execution_result = {"status": "success", "result": {"status": "success", "output": tool_response['text']}}
                        
                        break
                    
                    return {
                        "status": "success",
                        "code": full_code,
                        "response": tool_response['text'],
                        "round": self.current_round,
                        "execution_result": execution_result,
                        "call_verifier": call_verifier
                    }
                else:
                    # No tool called - this violates the requirement
                    if attempt < max_attempts - 1:
                        # Add feedback to encourage tool calling
                        self.conversation_history.append({
                            "role": "user",
                            "content": "You must call a tool in each interaction. Please use one of the available tools to proceed with your task."
                        })
                        continue
                    else:
                        # Final attempt failed - return error
                        return {
                            "status": "error",
                            "error": "No tool was called after multiple attempts. Tool calling is mandatory.",
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
        Add feedback to the agent's conversation history.
        
        Args:
            feedback: Feedback from verifier or executor
        """
        if os.path.isdir(feedback):
            feedback_content = get_image_base64(os.path.join(feedback, 'render1.png'))
            self.conversation_history.append({"role": "user", "content": [{"type": "text", "text": "Generated image:"}, {"type": "image_url", "image_url": {"url": feedback_content}}]})
        elif os.path.isfile(feedback):
            feedback_content = get_image_base64(feedback)
            self.conversation_history.append({"role": "user", "content": [{"type": "text", "text": "Generated image:"}, {"type": "image_url", "image_url": {"url": feedback_content}}]})
        else:
            self.conversation_history.append({"role": "user", "content": [{"type": "text", "text": feedback}]})
    
    def save_thought_process(self) -> None:
        """Save the current thought process to file."""
        current_memory = self._build_sliding_window_memory()
        save_thought_process(current_memory, self.config.get("thought_save"))
    
    def get_memory(self) -> List[Dict]:
        """Get the current memory/conversation history."""
        return self._build_sliding_window_memory()
    
    def reset_memory(self) -> None:
        """Reset the agent's memory."""
        self.conversation_history = []
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
            
            # Executor setup is now handled internally in call_tool
            # No need for explicit setup here
            
            return {
                "status": "success",
                "message": "Generator Agent initialized successfully. Executor will be setup automatically when needed."
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
