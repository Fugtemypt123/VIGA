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
    def __init__(self, 
                 mode: str, 
                 vision_model: str, 
                 api_key: str, 
                 thought_save: str, 
                 task_name: str,
                 max_rounds: int = 10, 
                 target_image_path: Optional[str] = None,
                 target_description: Optional[str] = None,
                 image_server_path: Optional[str] = None,
                 scene_server_path: Optional[str] = None,
                 api_base_url: Optional[str] = None,
                 blender_file: Optional[str] = None,
                 web_server_path: Optional[str] = None):
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
        if mode == "blendergym":
            self.target_image_path = os.path.abspath(os.path.join(target_image_path, 'render1.png'))
        else:
            self.target_image_path = None
        self.current_image_path = None
        self.current_round = 0
        self.tool_client = ExternalToolClient()
        self._tools_connected = False
        self.task_name = task_name
        self.blender_file = blender_file
        
        if mode == "blendergym" or mode == "autopresent" or mode == "design2code":
            self.server_type = "image"
            self.server_path = image_server_path
        elif mode == "blendergym-hard":
            self.server_type = "scene"
            self.server_path = scene_server_path
        else:
            raise NotImplementedError("Mode not implemented")
        
        # Initialize prompt builder and tool handler
        self.prompt_builder = PromptBuilder(self.client, self.vision_model)
        self.tool_handler = ToolHandler(self.tool_client, self.server_type)
        
        if mode == "blendergym":
            self.system_prompt = self.prompt_builder.build_blendergym_verifier_prompt(mode, task_name, target_image_path)
        elif mode == "autopresent":
            self.system_prompt = self.prompt_builder.build_autopresent_verifier_prompt(mode, target_description)
        elif mode == "blendergym-hard":
            self.system_prompt = self.prompt_builder.build_blendergym_hard_verifier_prompt(mode, task_name, target_image_path, blender_file, target_description)
        elif mode == "design2code":
            self.system_prompt = self.prompt_builder.build_design2code_verifier_prompt(mode, target_image_path)
        else:
            raise NotImplementedError("Mode not implemented")
        
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
        
    async def verify_scene(self, code: str, render_path: str, round_num: int) -> Dict[str, Any]:
        # reload investigator each time
        if self.mode == "blendergym-hard":
            setup_result = await self.setup_executor(blender_file=self.blender_file, save_dir=self.thought_save)
            if setup_result.get("status") != "success":
                return {"status": "error", "error": f"Scene server setup failed: {setup_result.get('error', setup_result)}"}
        await self._ensure_tools_connected()
        
        # define current round
        current_round = 0
        self.current_round = round_num
        # clear the memory at each time to enable long conversation
        self.memory = copy.deepcopy(self.system_prompt)
        
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
                    {"type": "image_url", "image_url": {"url": self.prompt_builder._get_image_base64(view1_path)}}
                ])
            if os.path.exists(view2_path):
                scene_content.extend([
                    {"type": "text", "text": f"Current scene (View 2):"},
                    {"type": "image_url", "image_url": {"url": self.prompt_builder._get_image_base64(view2_path)}}
                ])
            verify_message["content"].extend(scene_content)
            verify_message["content"].append({"type": "text", "text": prompts_dict[self.mode]['format']['verifier']})
            self.memory.append(verify_message)
        elif self.mode == "autopresent":
            verify_message = {"role": "user", "content": [{"type": "text", "text": f"Please analyze the current code and generated slide:\nCode: {code}"}]}
            # add slide screenshot
            if os.path.exists(render_path):
                verify_message["content"].append({"type": "text", "text": f"Generated slide screenshot:"})
                verify_message["content"].append({"type": "image_url", "image_url": {"url": self.prompt_builder._get_image_base64(render_path)}})
            verify_message["content"].append({"type": "text", "text": prompts_dict[self.mode]['format']['verifier']})
            self.memory.append(verify_message)
        elif self.mode == "blendergym-hard":
            level = self.task_name.split('-')[0]
            verify_message = {"role": "user", "content": [{"type": "text", "text": f"Please analyze the current state:\n"}]}
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
                    {"type": "image_url", "image_url": {"url": self.prompt_builder._get_image_base64(view1_path)}}
                ])
            if os.path.exists(view2_path):
                scene_content.extend([
                    {"type": "text", "text": f"Current scene (View 2):"},
                    {"type": "image_url", "image_url": {"url": self.prompt_builder._get_image_base64(view2_path)}}
                ])
            verify_message["content"].extend(scene_content)
            verify_message["content"].append({"type": "text", "text": prompts_dict[self.mode]['format']['verifier'][level]})
            self.memory.append(verify_message)
        elif self.mode == "design2code":
            verify_message = {"role": "user", "content": [{"type": "text", "text": f"Please analyze the generated HTML code and compare it with the target design:\nCode: {code}"}]}
            if os.path.exists(render_path):
                verify_message["content"].append({"type": "text", "text": f"Generated webpage screenshot:"})
                verify_message["content"].append({"type": "image_url", "image_url": {"url": self.prompt_builder._get_image_base64(render_path)}})
            verify_message["content"].append({"type": "text", "text": prompts_dict[self.mode]['format']['verifier']})
            self.memory.append(verify_message)
        else:
            raise NotImplementedError("Mode not implemented")
        
        # start verification
        try:
            for i in range(self.max_rounds):
                chat_args = {
                    "model": self.vision_model,
                    "messages": self.memory,
                }
                tools = self._get_tools()
                if tools:
                    chat_args['tools'] = tools
                    if self.vision_model == 'gpt-4o':
                        chat_args['parallel_tool_calls'] = False
                    if self.vision_model != 'Qwen2-VL-7B-Instruct':
                        chat_args['tool_choice'] = "auto"
                response = self.client.chat.completions.create(**chat_args)
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
                                    {"type": "image_url", "image_url": {"url": self.prompt_builder._get_image_base64(tool_response['image'])}},
                                    {"type": "text", "text": "Camera position:"},
                                    {"type": "text", "text": tool_response['camera_position']}
                                ]
                            })
                        result = {"status": "continue", "output": message.content if message.content else "Nothing to say"}
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
        
    async def _handle_tool_call(self, tool_call) -> Dict[str, Any]:
        """Handle tool calls from the verifier agent."""
        return await self.tool_handler.handle_verifier_tool_call(
            tool_call, 
            self.current_image_path, 
            self.target_image_path
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
    async def initialize_verifier(
        mode: str,
        vision_model: str,
        api_key: str,
        thought_save: str,
        task_name: str,
        max_rounds: int = 10,
        target_image_path: Optional[str] = None,
        target_description: Optional[str] = None,
        image_server_path: Optional[str] = None,
        scene_server_path: Optional[str] = None,
        blender_file: Optional[str] = None,
        api_base_url: Optional[str] = None,
        web_server_path: Optional[str] = None,
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
                target_description=target_description,
                image_server_path=image_server_path,
                scene_server_path=scene_server_path,
                api_base_url=api_base_url,
                blender_file=blender_file,
                web_server_path=web_server_path
            )
            agent_holder['agent'] = agent
            # Initialize server executor
            if mode == "blendergym" or mode == "autopresent":
                setup_result = await agent.setup_executor(vision_model=vision_model, api_key=api_key, api_base_url=api_base_url)
                if setup_result.get("status") != "success":
                    return {"status": "error", "error": f"Image server setup failed: {setup_result.get('error', setup_result)}"}
            elif mode == "blendergym-hard":
                setup_result = await agent.setup_executor(blender_file=blender_file, save_dir=thought_save)
                if setup_result.get("status") != "success":
                    return {"status": "error", "error": f"Scene server setup failed: {setup_result.get('error', setup_result)}"}
            elif mode == "design2code":
                setup_result = await agent.setup_executor(vision_model=vision_model, api_key=api_key, api_base_url=api_base_url)
                if setup_result.get("status") != "success":
                    return {"status": "error", "error": f"Web server setup failed: {setup_result.get('error', setup_result)}"}
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
