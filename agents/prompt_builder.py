
    
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
            prompts = prompt_manager.get_all_prompts(self.config.get("mode"), agent_type, self.config.get("task_name"), self.config.get("level"))
        
        # Build the prompt based on mode
        if prompt_type == "system":
            return self._build_system_prompt(prompts)
        elif prompt_type == "user":
            return self._build_user_prompt(prompts)
        else:
            raise NotImplementedError(f"Mode {self.config.get('mode')} not implemented")
        
    def _build_system_prompt(self, prompts: Dict) -> List[Dict]:
        """Build generator prompt for static_scene mode using prompt manager."""
        content = []
        if self.config.get("target_image_path"):
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
        if prompts['argument'].get('code_edit', '') != '':
            content = [
                {"type": "text", "text": f"Initial plan: {prompts.get('init_plan', '')}"},
                {"type": "text", "text": f"Thought: {prompts['argument'].get('thought', '')}"},
                {"type": "text", "text": f"Code edit: {prompts['argument'].get('code_edit', '')}"},
                {"type": "text", "text": f"Full code: {prompts['argument'].get('full_code', '')}"},
            ]
        else:
            content = [
                {"type": "text", "text": f"Initial plan: {prompts.get('init_plan', '')}"},
                {"type": "text", "text": f"Thought: {prompts['argument'].get('thought', '')}"},
                {"type": "text", "text": f"Full code: {prompts['argument'].get('full_code', '')}"},
            ]
        if 'image' in prompts['execution']:
            for text, image in zip(prompts['execution']['text'], prompts['execution']['image']):
                content.append({"type": "image_url", "image_url": {"url": get_image_base64(image)}})
                content.append({"type": "text", "text": f"Current scene render image loaded from local path: {image}"})
        else:
            for text in prompts['execution']['text']:
                content.append({"type": "text", "text": text})
        return [{"role": "user", "content": content}]