import os
import json
from typing import Dict, Any
from openai import OpenAI
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
            
        # Initialize tool client
        self.tool_client = ExternalToolClient(self.config.get("verifier_tools"), self.config)
        
        # Initialize OpenAI client
        client_kwargs = {"api_key": self.config.get("api_key"), 'base_url': self.config.get("api_base_url") or os.getenv("OPENAI_BASE_URL") or 'https://api.openai.com/v1'}
        self.client = OpenAI(**client_kwargs)
        
        # Initialize system prompt
        self.prompt_builder = PromptBuilder(self.client, self.config)
        self.system_prompt = self.prompt_builder.build_prompt("verifier", "system")
        self.memory.extend(self.system_prompt)
        
    async def run(self, user_message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verify the generated scene using CoT reasoning and fixed camera positions.
        Only called when generator uses execute_and_evaluate tool.
        """
        print("\n=== Running verifier agent ===\n")
        
        # Reload scene if needed
        if "reload_scene" in self.tool_client.tool_to_server:
            print("Reload scene...")
            await self.tool_client.call_tool("reload_scene", {})
        print("Build user message...")
        user_message = self.prompt_builder.build_prompt("verifier", "user", user_message)
        if self.config.get("clear_memory"):
            print("Clear memory...")
            self.memory = self.system_prompt + user_message
        else:
            print("Extend memory...")
            self.memory.extend(user_message)
        print("Save memory...")
        self._save_memory()
        result = None
        
        for i in range(self.config.get("max_rounds")):
            # Prepare chat args
            print(f"=== Round {i} ===\n")
            if len(self.memory) > self.config.get("memory_length")+1:
                if self.memory[-self.config.get("memory_length")+1]['role'] == 'tool':
                    memory = self.memory[:2] + self.memory[-self.config.get("memory_length"):]
                else:
                    memory = self.memory[:2] + self.memory[-self.config.get("memory_length")+1:]
            else:
                memory = self.memory
            
            print("Prepare chat args...")
            tool_configs = self.tool_client.tool_configs
            tool_configs = [x for v in tool_configs.values() for x in v]
            chat_args = {"model": self.config.get("model"), "messages": memory, "tools": tool_configs, "tool_choice": "auto", **self.init_chat_args}
            
            # Generate response
            print("Generate response...")
            response = self.client.chat.completions.create(**chat_args)
            message = response.choices[0].message
            
            # Handle tool call
            print("Handle tool call...")
            if not message.tool_calls:
                self.memory.append({"role": "assistant", "content": message.content})
                self.memory.append({"role": "user", "content": "Each return message must contain a tool call. Your previous message did not contain a tool call. Please reconsider."})
                self._save_memory()
                continue
            else:
                tool_call = message.tool_calls[0]
                print(f"Call tool {tool_call.function.name}...")
                tool_response = await self.tool_client.call_tool(tool_call.function.name, json.loads(tool_call.function.arguments))
                
            # Update and save memory
            print("Update and save memory...")
            self._update_memory({"assistant": message, "user": tool_response})
            self._save_memory()

            if tool_call.function.name == "end":
                result = tool_response
                break
            
        print("\n=== Finish verification process ===\n")
        
        if result:
            return result
        else:
            return {"status": "error", "output": {"text": ["No valid response, you are on your own."]}}
    
    def _update_memory(self, message: Dict):
        """Update the memory with the new message"""
        # Add tool calling
        assistant_content = message['assistant'].content
        assistant_tool_calls = message['assistant'].tool_calls[0].model_dump()
        self.memory.append({"role": "assistant", "content": assistant_content, "tool_calls": [assistant_tool_calls]})
        
        # Add tool response
        tool_call_id = message['assistant'].tool_calls[0].id
        tool_call_name = message['assistant'].tool_calls[0].function.name
        tool_response = []
        user_response = []
        
        if 'image' in message['user']:
            tool_response.append({"type": "text", "text": "The next user message contains the image result of the tool call."})
            for text, image in zip(message['user']['text'], message['user']['image']):
                user_response.append({"type": "text", "text": text})
                user_response.append({"type": "image_url", "image_url": {"url": get_image_base64(image)}})
                user_response.append({"type": "text", "text": f"Image loaded from local path: {image}"})
        else:
            for text in message['user']['text']:
                tool_response.append({"type": "text", "text": text}) 
        
        self.memory.append({"role": "tool", "content": tool_response, "name": tool_call_name, "tool_call_id": tool_call_id})
        if user_response:
            self.memory.append({"role": "user", "content": user_response})
    
    def _save_memory(self):
        """Save the memory to the file"""
        output_file = self.config.get("output_dir") + "/verifier_memory.json"
        with open(output_file, "w") as f:
            json.dump(self.memory, f, indent=4, ensure_ascii=False)
            
    async def cleanup(self):
        await self.tool_client.cleanup()