import os
import json
import asyncio
import copy
from typing import Dict, List, Optional, Any
import logging
from openai import OpenAI
from mcp.server.fastmcp import FastMCP
from prompts import prompts_dict
from agents.external_tool_client import ExternalToolClient
from agents.prompt_builder import PromptBuilder
from agents.tool_manager import ToolManager
from agents.tool_handler import ToolHandler
from agents.utils import save_thought_process

class VerifierAgent:
    def __init__(self, **kwargs):
        self.config = kwargs
        
        # Extract commonly used parameters
        self.mode = self.config.get("mode")
        self.vision_model = self.config.get("vision_model")
        self.api_key = self.config.get("api_key")
        self.task_name = self.config.get("task_name")
        self.max_rounds = self.config.get("max_rounds", 10)
        self.current_round = 0
        
        # Initialize OpenAI client
        api_base_url = self.config.get("api_base_url")
        client_kwargs = {"api_key": self.api_key}
        if api_base_url or os.getenv("OPENAI_BASE_URL"):
            client_kwargs["base_url"] = api_base_url or os.getenv("OPENAI_BASE_URL")
        self.client = OpenAI(**client_kwargs)
        
        # Setup thought save directory
        self.thought_save = self.config.get("thought_save")
        os.makedirs(self.thought_save, exist_ok=True)
        
        # Handle target image path
        target_image_path = self.config.get("target_image_path")
        if self.mode == "blendergym":
            self.target_image_path = os.path.abspath(os.path.join(target_image_path, 'render1.png'))
        else:
            self.target_image_path = None
        self.current_image_path = None
        
        # Initialize components
        self.tool_client = ExternalToolClient()
        self._tools_connected = False
        
        # Determine server type and path
        if self.mode == "blendergym" or self.mode == "autopresent" or self.mode == "design2code":
            self.server_type = "image"
            self.server_path = self.config.get("image_server_path")
        elif self.mode == "blendergym-hard":
            self.server_type = "scene"
            self.server_path = self.config.get("scene_server_path")
        else:
            raise NotImplementedError("Mode not implemented")
        
        # Initialize prompt builder and tool handler
        self.prompt_builder = PromptBuilder(self.client, self.vision_model)
        self.tool_handler = ToolHandler(self.tool_client, self.server_type)
        
        # Initialize system prompt using generic prompt builder
        self.system_prompt = self.prompt_builder.build_verifier_prompt(self.config)
        self.memory = copy.deepcopy(self.system_prompt)
        
    async def _ensure_tools_connected(self):
        if not self._tools_connected:
            if self.server_type == "image":
                await self.tool_client.connect_server("image", self.server_path, self.api_key)
            elif self.server_type == "scene":
                await self.tool_client.connect_server("scene", self.server_path)
            elif self.server_type == "web":
                await self.tool_client.connect_server("web", self.server_path, self.api_key)
            self._tools_connected = True
            
    async def setup_executor(self, **kwargs):
        await self._ensure_tools_connected()
        if self.server_type == "image":
            result = await self.tool_client.initialize_executor("image", **kwargs)
            return result
        elif self.server_type == "scene":
            # Initialize scene investigator
            blender_file = kwargs.get("blender_file", None)
            save_dir = kwargs.get("save_dir", None)
            result = await self.tool_client.call_tool("scene", "initialize_investigator", {
                "thoughtprocess_save": save_dir,
                "blender_path": blender_file,
            })
            return result
        elif self.server_type == "web":
            result = await self.tool_client.initialize_executor("web", **kwargs)
            return result
        return {"status": "success", "message": "No executor setup needed for this mode."}
        
    async def call(self, code: str, render_path: str, round_num: int) -> Dict[str, Any]:
        # reload investigator each time
        if self.mode == "blendergym-hard":
            setup_result = await self.setup_executor(blender_file=self.config.get("blender_file"), save_dir=self.thought_save)
            if setup_result.get("status") != "success":
                return {"status": "error", "error": f"Scene server setup failed: {setup_result.get('error', setup_result)}"}
        await self._ensure_tools_connected()
        
        # define current round
        current_round = 0
        self.current_round = round_num
        # clear the memory at each time to enable long conversation
        self.memory = copy.deepcopy(self.system_prompt)
        
        # build memory using generic prompt builder
        current_image_path_ref = [self.current_image_path]  # Use list for reference passing
        verify_message = self.prompt_builder.build_verify_message(self.config, code, render_path, current_image_path_ref)
        self.current_image_path = current_image_path_ref[0]  # Update the reference
        self.memory.append(verify_message)
        
        # start verification
        try:
            for i in range(self.max_rounds):
                chat_args = {
                    "model": self.vision_model,
                    "messages": self.memory,
                    "tools": self._get_tools()
                }
                if chat_args['tools']:
                    if self.vision_model == 'gpt-4o':
                        chat_args['parallel_tool_calls'] = False
                    if self.vision_model != 'Qwen2-VL-7B-Instruct':
                        chat_args['tool_choice'] = "auto"
                response = self.client.chat.completions.create(**chat_args)
                message = response.choices[0].message
                
                self.memory.append(message.model_dump())
                # Handle tool calls
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
                        tool_response = await self._handle_tool_call(tool_call, round_num)
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
                                    {"type": "image_url", "image_url": {"url": self.prompt_builder._get_image_base64(tool_response['image'])}},
                                    {"type": "text", "text": "Camera position:"},
                                    {"type": "text", "text": tool_response['camera_position']}
                                ]
                            })
                        result = {"status": "continue", "output": message.content if message.content else "Please continue to use the tool to observe the scene, or summarize the existing content and give feedback."}
                else:
                    # No tool calls, check if verification is complete
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
        save_thought_process(self.memory, self.thought_save, self.current_round)
            
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
            if args.get("mode") == "blendergym" or args.get("mode") == "autopresent" or args.get("mode") == "design2code":
                setup_result = await agent.setup_executor(**args)
                if setup_result.get("status") != "success":
                    return {"status": "error", "error": f"Server setup failed: {setup_result.get('error', setup_result)}"}
            elif args.get("mode") == "blendergym-hard":
                # For scene mode, we need to map save_dir to thoughtprocess_save
                setup_kwargs = args.copy()
                setup_kwargs["save_dir"] = args.get("thought_save")
                setup_result = await agent.setup_executor(**setup_kwargs)
                if setup_result.get("status") != "success":
                    return {"status": "error", "error": f"Scene server setup failed: {setup_result.get('error', setup_result)}"}
            await agent._ensure_tools_connected()
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
