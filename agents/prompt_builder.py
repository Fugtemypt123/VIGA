
    
import os
import json
from typing import Dict, List, Optional
from openai import OpenAI
from prompts import prompt_manager
from utils.common import get_image_base64

class PromptBuilder:
    """Helper class for building system prompts for generator and verifier agents."""
    
    def __init__(self, client: OpenAI, config: Dict):
        self.client = client
        self.config = config
    
    def build_prompt(self, agent_type: str, prompt_type: str, prompts: dict = None) -> List[Dict]:
        """Generic method to build generator prompts based on mode and config."""
        # Get system prompt only (format/hints embedded in system)
        if not prompts:
            prompts = prompt_manager.get_all_prompts(self.config.get("mode"), agent_type, self.config.get("task_name"), self.config.get("level"), self.config.get("no_tools"))
        
        # Build the prompt based on mode
        if prompt_type == "system":
            return self._build_system_prompt(prompts, agent_type)
        elif prompt_type == "user":
            return self._build_user_prompt(prompts)
        else:
            raise NotImplementedError(f"Mode {self.config.get('mode')} not implemented")
        
    def _build_system_prompt(self, prompts: Dict, agent_type: str) -> List[Dict]:
        """Build generator prompt for static_scene mode using prompt manager."""
        content = []
        
        if self.config.get("init_code_path") and agent_type == "generator":
            with open(self.config.get("init_code_path"), 'r') as f:
                content.append({"type": "text", "text": f"Initial code: {f.read()}"})
                
        if self.config.get("init_image_path") and agent_type == "generator":
            if os.path.isdir(self.config.get("init_image_path")):
                if 'render1.png' in os.listdir(self.config.get("init_image_path")):
                    content.append({"type": "image_url", "image_url": {"url": get_image_base64(os.path.join(self.config.get("init_image_path"), 'render1.png'))}})
                    content.append({"type": "text", "text": f"Initial image loaded from local path: {os.path.join(self.config.get('init_image_path'), 'render1.png')}"})
                else:
                    for i, file in enumerate(os.listdir(self.config.get("init_image_path"))):
                        content.append({"type": "image_url", "image_url": {"url": get_image_base64(os.path.join(self.config.get("init_image_path"), file))}})
                        content.append({"type": "text", "text": f"Initial image {i+1} loaded from local path: {os.path.join(self.config.get('init_image_path'), file)}"})
            else:
                content.append({"type": "image_url", "image_url": {"url": get_image_base64(self.config.get("init_image_path"))}})
                content.append({"type": "text", "text": f"Initial image loaded from local path: {self.config.get('init_image_path')}"})

        if self.config.get("target_image_path"):
            if os.path.isdir(self.config.get("target_image_path")):
                if 'visprompt1.png' in os.listdir(self.config.get("target_image_path")):
                    content.append({"type": "image_url", "image_url": {"url": get_image_base64(os.path.join(self.config.get("target_image_path"), 'visprompt1.png'))}})
                    content.append({"type": "text", "text": f"Target image loaded from local path: {os.path.join(self.config.get('target_image_path'), 'visprompt1.png')}"})
                elif 'style1.png' in os.listdir(self.config.get("target_image_path")):
                    content.append({"type": "image_url", "image_url": {"url": get_image_base64(os.path.join(self.config.get("target_image_path"), 'style1.png'))}})
                    content.append({"type": "text", "text": f"Target image loaded from local path: {os.path.join(self.config.get('target_image_path'), 'style1.png')}"})
                elif 'render1.png' in os.listdir(self.config.get("target_image_path")):
                    content.append({"type": "image_url", "image_url": {"url": get_image_base64(os.path.join(self.config.get("target_image_path"), 'render1.png'))}})
                    content.append({"type": "text", "text": f"Target image loaded from local path: {os.path.join(self.config.get('target_image_path'), 'render1.png')}"})
                else:
                    for i, file in enumerate(os.listdir(self.config.get("target_image_path"))):
                        content.append({"type": "image_url", "image_url": {"url": get_image_base64(os.path.join(self.config.get("target_image_path"), file))}})
                        content.append({"type": "text", "text": f"Target image {i+1} loaded from local path: {os.path.join(self.config.get('target_image_path'), file)}"})
            else:
                content.append({"type": "image_url", "image_url": {"url": get_image_base64(self.config.get("target_image_path"))}})
                content.append({"type": "text", "text": f"Target image loaded from local path: {self.config.get('target_image_path')}"})  
            
        if self.config.get("target_description"):
            content.append({"type": "text", "text": f"Task description: {self.config.get('target_description')}"})
        
        if self.config.get("resource_dir"):
            for file in os.listdir(os.path.join(self.config.get("resource_dir"), "media")):
                content.append({"type": "image_url", "image_url": {"url": get_image_base64(os.path.join(self.config.get("resource_dir"), "media", file))}})
                content.append({"type": "text", "text": f"Resource image loaded from local path: {os.path.join(self.config.get('resource_dir'), 'media', file)}. You can import these images when generating the scene."})
            content.append({"type": "text", "text": f"Please specify the output slide path as output.pptx in the code."})
            
        return [{"role": "system", "content": prompts.get('system', '')}, {"role": "user", "content": content}]
    
    def _build_user_prompt(self, prompts: Dict) -> List[Dict]:
        content = []
        if prompts.get('init_plan'):
            content.append({"type": "text", "text": f"Initial plan: {prompts.get('init_plan')}"})
        for key, value in prompts['argument'].items():
            content.append({"type": "text", "text": f"{key}: {value}"})
        if 'image' in prompts['execution']:
            for text, image in zip(prompts['execution']['text'], prompts['execution']['image']):
                content.append({"type": "image_url", "image_url": {"url": get_image_base64(image)}})
                content.append({"type": "text", "text": f"Current scene render image loaded from local path: {image}"})
        else:
            for text in prompts['execution']['text']:
                content.append({"type": "text", "text": text})
        return [{"role": "user", "content": content}]
    
    def build_memory(self, memory: List[Dict]) -> List[Dict]:
        system_memory = memory[:2]
        reverse_memory = memory[2:][::-1]
        chat_memory = []
        i = 0
        while i < len(reverse_memory):
            if reverse_memory[i]['role'] == 'tool' and reverse_memory[i]['name'] == 'undo-last-step':
                # If role == user, skip 2+3=5 steps
                if reverse_memory[i+2]['role'] == 'user':
                    i += 5
                else:
                    i += 4
            if i >= len(reverse_memory):
                break
            chat_memory.append(reverse_memory[i])
            if len(chat_memory) >= self.config.get("memory_length"):
                break
            i += 1
        if len(chat_memory) > 0 and chat_memory[-1]['role'] == 'tool':
            chat_memory.pop()
        all_memory = system_memory + chat_memory[::-1]
        if self.config.get('explicit_comp'):
            target_image_message = []
            for i in range(len(memory[1]['content'])):
                if memory[1]['content'][i]['type'] == 'text' and 'Target image' in memory[1]['content'][i]['text']:
                    target_image_message.append(memory[1]['content'][i-1])
                    target_image_message.append(memory[1]['content'][i])
                    break
            initial_image_message = []
            for i in range(len(memory[1]['content'])):
                if memory[1]['content'][i]['type'] == 'text' and 'Initial image' in memory[1]['content'][i]['text']:
                    initial_image_message.append(memory[1]['content'][i-1])
                    initial_image_message.append(memory[1]['content'][i])
                    break
            last_image_message = []
            last_id = len(memory)-1
            for i in range(len(memory)-1, 0,-1):
                if memory[i]['role'] == 'user' and memory[i]['content'][1]['type'] == 'image_url':
                    last_image_message.append(memory[i]['content'][1])
                    last_image_message.append(memory[i]['content'][2])
                    last_id = i
                    break
            last_last_image_message = []
            for i in range(last_id-1, 0,-1):
                if memory[i]['role'] == 'user' and memory[i]['content'][1]['type'] == 'image_url':
                    last_last_image_message.append(memory[i]['content'][1])
                    last_last_image_message.append(memory[i]['content'][2])
                    break
            if len(last_last_image_message) > 0:
                all_memory.append({
                    "role": "user", 
                    "content": [{
                        "type": "text", 
                        "text": "Here are three images: target state, last state, and the state before the last state. Please compare these images and determine whether your current operation is effectively approaching the target scenario, thereby determine your next action."
                    }] + target_image_message + last_image_message + last_last_image_message
                })
        return all_memory