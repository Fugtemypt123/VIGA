import os
import json
import asyncio
import copy
from typing import Dict, List, Optional, Any
import logging
from openai import OpenAI
from mcp.server.fastmcp import FastMCP
from prompts import prompt_manager
from agents.tool_client import ExternalToolClient
from agents.prompt_builder import PromptBuilder
from agents.tool_manager import ToolManager
from agents.tool_handler import ToolHandler
from agents.config_manager import ConfigManager
from agents.utils import get_scene_info, save_thought_process, get_image_base64

class VerifierAgent:
    def __init__(self, **kwargs):
        self.config = kwargs
        
        with open('logs/verifier.log', 'w') as f:
            f.write(f"stage1: config: {kwargs}\n")
        
        # Initialize configuration manager
        self.config_manager = ConfigManager(kwargs)
        
        # Validate configuration
        is_valid, error_message = self.config_manager.validate_verifier_configuration()
        if not is_valid:
            raise ValueError(f"Invalid configuration: {error_message}")
        
        # Extract commonly used parameters from config manager
        self.mode = self.config_manager.mode
        self.vision_model = self.config_manager.vision_model
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
        
        # Setup thought save directory
        self.thought_save = self.config_manager.thought_save
        os.makedirs(self.thought_save, exist_ok=True)
        
        # Handle target image path using config manager
        if self.config_manager.is_blender_mode:
            target_path = self.config_manager.get_target_image_path_for_mode()
            self.target_image_path = os.path.abspath(target_path) if target_path else None
        else:
            self.target_image_path = None
        self.current_image_path = None
        
        # Initialize components
        self.tool_client = ExternalToolClient()
        self._server_connected = False
        
        # Determine server type and path using config manager
        self.server_type, self.server_path = self.config_manager.get_verifier_server_type_and_path()
        
        # Initialize prompt builder and tool handler
        self.prompt_builder = PromptBuilder(self.client, self.vision_model)
        self.tool_handler = ToolHandler(self.tool_client, self.server_type)
        
        # Initialize system prompt using generic prompt builder
        self.system_prompt = self.prompt_builder.build_verifier_prompt(kwargs)
        self.memory = copy.deepcopy(self.system_prompt)
        self.conversation_history = []  # Store last 6 chats for sliding window
        self.suggestions_initialized = False  # Track if suggestions have been initialized
        
        with open('logs/verifier.log', 'a') as f:
            f.write(f"stage2: system_prompt: {self.system_prompt}\n")
        
    async def _ensure_server_connected(self):
        if not self._server_connected:
            await self.tool_client.connect_server(self.server_type, self.server_path, self.api_key)
            self._server_connected = True
            
    async def setup_investigator(self, **kwargs):
        await self._ensure_server_connected()
        # ExternalToolClient is now a thin forwarder; call the tool directly
        result = await self.tool_client.call_tool(
            server_type=self.server_type,
            tool_name="initialize_investigator",
            tool_args={"args": kwargs}
        )
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
    
    def _initialize_suggestions(self):
        """Initialize comparison suggestions for the verifier"""
        if not self.suggestions_initialized:
            suggestions = {
                "compare_image": [
                    "Compare overall composition and scene layout",
                    "Check object positioning and spatial relationships", 
                    "Verify lighting and shadow consistency",
                    "Examine material properties and textures",
                    "Assess camera angle and perspective"
                ],
                "compare_text": [
                    "Analyze scene description accuracy",
                    "Verify object count and types",
                    "Check spatial positioning details",
                    "Validate material and color specifications",
                    "Confirm lighting setup and effects"
                ]
            }
            self.suggestions_initialized = True
            return suggestions
        return None
        
    async def call(self, code: str, render_path: str, round_num: int) -> Dict[str, Any]:
        """
        Verify the generated scene using CoT reasoning and fixed camera positions.
        Only called when generator uses execute_and_evaluate tool.
        """
        # Setup investigator if needed
        if self.config_manager.is_blendergym_hard_mode:
            setup_result = await self.setup_investigator(**self.config)
            if setup_result.get("status") != "success":
                return {"status": "error", "error": f"Scene server setup failed: {setup_result.get('error', setup_result)}"}
        await self._ensure_server_connected()
        
        self.current_round = round_num
        
        # Initialize suggestions if first time
        suggestions = self._initialize_suggestions()
        
        # Build verification message with code block focus
        current_image_path_ref = [self.current_image_path]
        verify_message = self.prompt_builder.build_verify_message(self.config, code, render_path, current_image_path_ref)
        self.current_image_path = current_image_path_ref[0]
        
        # Add code block focus instruction
        code_focus_message = {
            "role": "user", 
            "content": f"Focus on analyzing the following code block that was just executed:\n\n```python\n{code}\n```\n\nPay special attention to how this code affects the scene representation and consistency."
        }
        
        # Add suggestions if initialized
        if suggestions:
            suggestions_message = {
                "role": "user",
                "content": f"Use these comparison suggestions to guide your analysis:\n\nImage Comparison:\n" + 
                          "\n".join([f"- {s}" for s in suggestions["compare_image"]]) +
                          f"\n\nText Analysis:\n" + 
                          "\n".join([f"- {s}" for s in suggestions["compare_text"]])
            }
        
        # Add scene info for level4 if needed
        scene_info_content = None
        if self.config_manager.should_add_scene_info():
            scene_config = self.config_manager.get_scene_info_config()
            scene_info_content = {"role": "user", "content": get_scene_info(scene_config["task_name"], scene_config["blender_file"])}
        
        # Build memory with sliding window
        memory = self._build_sliding_window_memory()
        
        # Add new messages to memory
        memory.append(verify_message)
        memory.append(code_focus_message)
        if suggestions:
            memory.append(suggestions_message)
        if scene_info_content:
            memory.append(scene_info_content)
        
        # Store in conversation history
        self.conversation_history.append(verify_message)
        self.conversation_history.append(code_focus_message)
        if suggestions:
            self.conversation_history.append(suggestions_message)
        if scene_info_content:
            self.conversation_history.append(scene_info_content)
        
        # Start verification with CoT reasoning
        try:
            for i in range(self.max_rounds):
                chat_args = {
                    "model": self.vision_model,
                    "messages": memory,
                }
                if self._get_tools():
                    chat_args['tools'] = self._get_tools()
                    if 'gpt' in self.vision_model:
                        chat_args['parallel_tool_calls'] = False
                    if self.vision_model != 'Qwen2-VL-7B-Instruct':
                        chat_args['tool_choice'] = "auto"
                        
                response = self.client.chat.completions.create(**chat_args)
                message = response.choices[0].message
                
                # Store assistant message in conversation history
                self.conversation_history.append(message.model_dump())
                memory.append(message.model_dump())
                
                # Handle tool calls
                if message.tool_calls:
                    for i, tool_call in enumerate(message.tool_calls):
                        if i > 0:
                            self.conversation_history.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "name": tool_call.function.name,
                                "content": "You can only call a tool once per conversation round."
                            })
                            memory.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "name": tool_call.function.name,
                                "content": "You can only call a tool once per conversation round."
                            })
                            continue
                            
                        tool_response = await self._handle_tool_call(tool_call, round_num)
                        
                        # Store tool response in conversation history and memory
                        tool_message = {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_call.function.name,
                            "content": tool_response['text']
                        }
                        self.conversation_history.append(tool_message)
                        memory.append(tool_message)
                        
                        # Add image if available
                        if tool_response.get('image'):
                            image_message = {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": "Generated image:"},
                                    {"type": "image_url", "image_url": {"url": get_image_base64(tool_response['image'])}},
                                    {"type": "text", "text": "Camera position:"},
                                    {"type": "text", "text": tool_response['camera_position']}
                                ]
                            }
                            self.conversation_history.append(image_message)
                            memory.append(image_message)
                        
                        # Continue with CoT reasoning
                        result = {"status": "continue", "output": message.content if message.content else "Please continue your Chain of Thought analysis using the tool results."}
                else:
                    # No tool calls, check if verification is complete
                    if "END THE PROCESS" in message.content and "Code Localization" not in message.content:
                        result = {"status": "end", "output": message.content}
                    else:
                        result = {"status": "continue", "output": message.content}
                    break
                    
            self.save_thought_process()
            return result
            
        except Exception as e:
            logging.error(f"Verification failed: {e}")
            return {"status": "error", "error": str(e), "round": self.current_round}
        
    def _get_tools(self) -> List[Dict]:
        """Get available tools for the verifier agent."""
        return ToolManager.get_verifier_tools(self.mode, self.task_name)
        
    async def _handle_tool_call(self, tool_call, round_num) -> Dict[str, Any]:
        """Handle tool calls from the verifier agent."""
        return await self.tool_handler.handle_verifier_tool_call(
            tool_call, 
            self.current_image_path, 
            self.target_image_path,
            round_num
        )
        
    def save_thought_process(self):
        """Save the current thought process to file."""
        current_memory = self._build_sliding_window_memory()
        save_thought_process(current_memory, self.thought_save, self.current_round)
            
    async def cleanup(self):
        await self.tool_client.cleanup()

def main():
    mcp = FastMCP("verifier")
    agent_holder = {}
    
    @mcp.tool()
    async def initialize_verifier(args: dict) -> dict:
        try:
            agent = VerifierAgent(**args)
            agent_holder['agent'] = agent
            # Initialize server executor
            with open('logs/verifier.log', 'a') as f:
                f.write(f"stage3: setup_investigator: {args}\n")
            setup_result = await agent.setup_investigator(**args)
            if setup_result.get("status") != "success":
                return {"status": "error", "error": f"Server setup failed: {setup_result.get('error', setup_result)}"}
            with open('logs/verifier.log', 'a') as f:
                f.write(f"stage4: setup_result: {setup_result}\n")
            await agent._ensure_server_connected()
            with open('logs/verifier.log', 'a') as f:
                f.write(f"stage5: server connected\n")
            return {"status": "success", "message": "Verifier Agent initialized and tool servers connected"}
        except Exception as e:
            return {"status": "error", "error": str(e)}
        
    @mcp.tool()
    async def call(code: str, render_path: str, round_num: int) -> dict:
        try:
            agent = agent_holder['agent']
            result = await agent.call(code, render_path, round_num)
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
