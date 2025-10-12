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
from utils.common import get_image_base64

class VerifierAgent:
    def __init__(self, args):
        self.config = args
        self.memory = []
        
        # Initialize chat args
        self.init_chat_args = {}
        if 'gpt' in self.config.get("model"):
            self.init_chat_args['parallel_tool_calls'] = False
        if self.config.get("model") != 'Qwen2-VL-7B-Instruct':
            self.init_chat_args['tool_choice'] = "auto"
            
        # Initialize tool client
        self.tool_client = ExternalToolClient(self.config.get("tool_servers"), self.config)
        
        # Initialize OpenAI client
        client_kwargs = {"api_key": self.config.get("api_key"), 'base_url': self.config.get("api_base_url") or os.getenv("OPENAI_BASE_URL") or 'https://api.openai.com/v1'}
        self.client = OpenAI(**client_kwargs)
        
        # Initialize system prompt
        self.prompt_builder = PromptBuilder(self.client, self.config)
        self.system_prompt = self.prompt_builder.build_prompt("verifier", "system")
        self.memory.append(self.system_prompt)
        
    async def run(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verify the generated scene using CoT reasoning and fixed camera positions.
        Only called when generator uses execute_and_evaluate tool.
        """
        self.tool_client.connect_servers()
        user_message = self.prompt_builder.build_prompt("verifier", "user")
        self.memory.append(user_message)
        for i in range(self.config.get("max_rounds")):
            # Prepare chat args
            memory = ([self.memory[0]] + self.memory[-self.config.get("memory_length")+1:]) if len(self.memory) > self.config.get("memory_length") else self.memory
            tools = self.tool_client.tool_configs
            chat_args = {"model": self.config.get("model"), "messages": memory, "tools": tools, **self.init_chat_args}
            
            # Generate response
            response = self.client.chat.completions.create(**chat_args)
            message = response.choices[0].message
            if not message.tool_calls:
                continue
            
            tool_call = message.tool_calls[0]
            tool_response = await self.tool_client.call_tool(tool_call.function.name, tool_call.function.arguments)
                
            # Update and save memory
            self._update_memory({"assistant": message, "user": tool_response})
            self._save_memory()

            if tool_call.function.name == "end":
                return tool_response
        
        return {"status": "error", "output": {"text": ["No valid response, you are on your own."]}}
    
    def _update_memory(self, message: Dict):
        """Update the memory with the new message"""
        # Add tool calling
        assistant_content = message['assistant'].content
        assistant_tool_calls = message['assistant'].tool_calls[0]
        self.memory.append({"role": "assistant", "content": assistant_content, "tool_calls": [assistant_tool_calls]})
        
        # Add tool response
        tool_call_id = message['assistant'].tool_calls[0]['id']
        tool_call_name = message['assistant'].tool_calls[0]['function']['name']
        tool_response = []
        if 'image' in message['user']:
            for text, image in zip(message['user']['text'], message['user']['image']):
                tool_response.append({"type": "text", "text": text})
                tool_response.append({"type": "image_url", "image_url": {"url": get_image_base64(image)}})
                tool_response.append({"type": "text", "text": f"Image loaded from local path: {image}"})
        else:
            for text in message['user']['text']:
                tool_response.append({"type": "text", "text": text})
            
        self.memory.append({"role": "tool", "content": tool_response, "name": tool_call_name, "tool_call_id": tool_call_id})
    
    def _save_memory(self):
        """Save the memory to the file"""
        output_file = self.config.get("output_dir") + "/verifier_memory.json"
        with open(output_file, "w") as f:
            json.dump(self.memory, f, indent=4, ensure_ascii=False)
            
    async def cleanup(self):
        await self.tool_client.cleanup()