import os
import json
from typing import Dict, List, Optional
from openai import OpenAI
from prompts import prompt_manager
from agents.utils import get_image_base64

class PromptBuilder:
    """Helper class for building system prompts for generator and verifier agents."""
    
    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model
    
    def build_generator_prompt(self, config: Dict) -> List[Dict]:
        """Generic method to build generator prompts based on mode and config."""
        mode = config.get("mode")
        task_name = config.get("task_name")
        
        # Extract level for blendergym-hard mode
        level = None
        if mode == "blendergym-hard":
            level = task_name.split('-')[0] if task_name else None
        
        # Get all prompts using the new prompt manager
        prompts = prompt_manager.get_all_prompts(mode, "generator", task_name, level)
        
        # Build the prompt based on mode
        if mode == "blendergym":
            return self._build_blendergym_generator_prompt(
                config, prompts
            )
        elif mode == "autopresent":
            return self._build_autopresent_generator_prompt(
                config, prompts
            )
        elif mode == "blendergym-hard":
            return self._build_blendergym_hard_generator_prompt(
                config, prompts
            )
        elif mode == "design2code":
            return self._build_design2code_generator_prompt(
                config, prompts
            )
        elif mode == "static_scene":
            return self._build_static_scene_generator_prompt(
                config, prompts
            )
        elif mode == "dynamic_scene":
            return self._build_dynamic_scene_generator_prompt(
                config, prompts
            )
        else:
            raise NotImplementedError(f"Mode {mode} not implemented")
    
    def build_verifier_prompt(self, config: Dict) -> List[Dict]:
        """Generic method to build verifier prompts based on mode and config."""
        mode = config.get("mode")
        task_name = config.get("task_name")
        
        # Extract level for blendergym-hard mode
        level = None
        if mode == "blendergym-hard":
            level = task_name.split('-')[0] if task_name else None
        
        # Get all prompts using the new prompt manager
        prompts = prompt_manager.get_all_prompts(mode, "verifier", task_name, level)
        
        # Build the prompt based on mode
        if mode == "blendergym":
            return self._build_blendergym_verifier_prompt(
                config, prompts
            )
        elif mode == "autopresent":
            return self._build_autopresent_verifier_prompt(
                config, prompts
            )
        elif mode == "blendergym-hard":
            return self._build_blendergym_hard_verifier_prompt(
                config, prompts
            )
        elif mode == "design2code":
            return self._build_design2code_verifier_prompt(
                config, prompts
            )
        elif mode == "static_scene":
            return self._build_static_scene_verifier_prompt(
                config, prompts
            )
        elif mode == "dynamic_scene":
            return self._build_dynamic_scene_verifier_prompt(
                config, prompts
            )
        else:
            raise NotImplementedError(f"Mode {mode} not implemented")
    
    def build_verify_message(self, config: Dict, code: str, render_path: str, current_image_path_ref: List) -> Dict:
        """Generic method to build verify messages based on mode and config."""
        mode = config.get("mode")
        task_name = config.get("task_name")
        
        # Extract level for blendergym-hard mode
        level = None
        if mode == "blendergym-hard":
            level = task_name.split('-')[0] if task_name else None
        
        # Get format prompt using the new prompt manager
        format_prompt = prompt_manager.get_format_prompt(mode, "verifier", level)
        
        if mode == "blendergym":
            return self._build_blendergym_verify_message(code, render_path, current_image_path_ref, format_prompt)
        elif mode == "autopresent":
            return self._build_autopresent_verify_message(code, render_path, format_prompt)
        elif mode == "blendergym-hard":
            return self._build_blendergym_hard_verify_message(config, code, render_path, current_image_path_ref, format_prompt)
        elif mode == "design2code":
            return self._build_design2code_verify_message(code, render_path, format_prompt)
        elif mode == "static_scene":
            return self._build_static_scene_verify_message(config, code, render_path, current_image_path_ref, format_prompt)
        elif mode == "dynamic_scene":
            return self._build_dynamic_scene_verify_message(config, code, render_path, current_image_path_ref, format_prompt)
        else:
            raise NotImplementedError(f"Mode {mode} not implemented")
    
    def _build_blendergym_verify_message(self, code: str, render_path: str, current_image_path_ref: List, format_prompt: str) -> Dict:
        """Build verify message for blendergym mode."""
        verify_message = {"role": "user", "content": [{"type": "text", "text": f"Please analyze the current state:\nCode: {code}"}]}
        
        if os.path.isdir(render_path):
            view1_path = os.path.join(render_path, 'render1.png')
            view2_path = os.path.join(render_path, 'render2.png')
        else:
            view1_path = render_path
            view2_path = None
            
        scene_content = []
        if os.path.exists(view1_path):
            current_image_path_ref[0] = os.path.abspath(view1_path)
            scene_content.extend([
                {"type": "text", "text": f"Current scene (View 1):"},
                {"type": "image_url", "image_url": {"url": get_image_base64(view1_path)}}
            ])
        if os.path.exists(view2_path):
            scene_content.extend([
                {"type": "text", "text": f"Current scene (View 2):"},
                {"type": "image_url", "image_url": {"url": get_image_base64(view2_path)}}
            ])
            
        verify_message["content"].extend(scene_content)
        verify_message["content"].append({"type": "text", "text": format_prompt})
        
        return verify_message
    
    def _build_autopresent_verify_message(self, code: str, render_path: str, format_prompt: str) -> Dict:
        """Build verify message for autopresent mode."""
        verify_message = {"role": "user", "content": [{"type": "text", "text": f"Please analyze the current code and generated slide:\nCode: {code}"}]}
        
        # add slide screenshot
        if os.path.exists(render_path):
            verify_message["content"].append({"type": "text", "text": f"Generated slide screenshot:"})
            verify_message["content"].append({"type": "image_url", "image_url": {"url": get_image_base64(render_path)}})
            
        verify_message["content"].append({"type": "text", "text": format_prompt})
        
        return verify_message
    
    def _build_blendergym_hard_verify_message(self, config: Dict, code: str, render_path: str, current_image_path_ref: List, format_prompt: str) -> Dict:
        """Build verify message for blendergym-hard mode."""
        verify_message = {"role": "user", "content": [{"type": "text", "text": f"Please analyze the current state:\n"}]}
        
        if os.path.isdir(render_path):
            view1_path = os.path.join(render_path, 'render1.png')
            view2_path = os.path.join(render_path, 'render2.png')
        else:
            view1_path = render_path
            view2_path = None
            
        scene_content = []
        if os.path.exists(view1_path):
            current_image_path_ref[0] = os.path.abspath(view1_path)
            scene_content.extend([
                {"type": "text", "text": f"Current scene (View 1):"},
                {"type": "image_url", "image_url": {"url": get_image_base64(view1_path)}}
            ])
        if os.path.exists(view2_path):
            scene_content.extend([
                {"type": "text", "text": f"Current scene (View 2):"},
                {"type": "image_url", "image_url": {"url": get_image_base64(view2_path)}}
            ])
            
        verify_message["content"].extend(scene_content)
        verify_message["content"].append({"type": "text", "text": format_prompt})
        
        return verify_message
    
    def _build_design2code_verify_message(self, code: str, render_path: str, format_prompt: str) -> Dict:
        """Build verify message for design2code mode."""
        verify_message = {"role": "user", "content": [{"type": "text", "text": f"Please analyze the generated HTML code and compare it with the target design:\nCode: {code}"}]}
        
        if os.path.exists(render_path):
            verify_message["content"].append({"type": "text", "text": f"Generated webpage screenshot:"})
            verify_message["content"].append({"type": "image_url", "image_url": {"url": get_image_base64(render_path)}})
            
        verify_message["content"].append({"type": "text", "text": format_prompt})
        
        return verify_message
    
    # Placeholder methods for the new prompt building system
    # These will be implemented to use the prompt_manager
    
    def _build_blendergym_generator_prompt(self, config: Dict, prompts: Dict) -> List[Dict]:
        """Build generator prompt for blendergym mode using prompt manager."""
        # This method needs to be implemented to use the new prompt manager
        # For now, return a placeholder
        return [{"role": "system", "content": "Placeholder - needs implementation"}]
    
    def _build_autopresent_generator_prompt(self, config: Dict, prompts: Dict) -> List[Dict]:
        """Build generator prompt for autopresent mode using prompt manager."""
        return [{"role": "system", "content": "Placeholder - needs implementation"}]
    
    def _build_blendergym_hard_generator_prompt(self, config: Dict, prompts: Dict) -> List[Dict]:
        """Build generator prompt for blendergym-hard mode using prompt manager."""
        return [{"role": "system", "content": "Placeholder - needs implementation"}]
    
    def _build_design2code_generator_prompt(self, config: Dict, prompts: Dict) -> List[Dict]:
        """Build generator prompt for design2code mode using prompt manager."""
        return [{"role": "system", "content": "Placeholder - needs implementation"}]
    
    def _build_blendergym_verifier_prompt(self, config: Dict, prompts: Dict) -> List[Dict]:
        """Build verifier prompt for blendergym mode using prompt manager."""
        return [{"role": "system", "content": "Placeholder - needs implementation"}]
    
    def _build_autopresent_verifier_prompt(self, config: Dict, prompts: Dict) -> List[Dict]:
        """Build verifier prompt for autopresent mode using prompt manager."""
        return [{"role": "system", "content": "Placeholder - needs implementation"}]
    
    def _build_blendergym_hard_verifier_prompt(self, config: Dict, prompts: Dict) -> List[Dict]:
        """Build verifier prompt for blendergym-hard mode using prompt manager."""
        return [{"role": "system", "content": "Placeholder - needs implementation"}]
    
    def _build_design2code_verifier_prompt(self, config: Dict, prompts: Dict) -> List[Dict]:
        """Build verifier prompt for design2code mode using prompt manager."""
        return [{"role": "system", "content": "Placeholder - needs implementation"}]
    
    def _build_static_scene_generator_prompt(self, config: Dict, prompts: Dict) -> List[Dict]:
        """Build generator prompt for static_scene mode using prompt manager."""
        return [{"role": "system", "content": "Placeholder - needs implementation"}]
    
    def _build_dynamic_scene_generator_prompt(self, config: Dict, prompts: Dict) -> List[Dict]:
        """Build generator prompt for dynamic_scene mode using prompt manager."""
        return [{"role": "system", "content": "Placeholder - needs implementation"}]
    
    def _build_static_scene_verifier_prompt(self, config: Dict, prompts: Dict) -> List[Dict]:
        """Build verifier prompt for static_scene mode using prompt manager."""
        return [{"role": "system", "content": "Placeholder - needs implementation"}]
    
    def _build_dynamic_scene_verifier_prompt(self, config: Dict, prompts: Dict) -> List[Dict]:
        """Build verifier prompt for dynamic_scene mode using prompt manager."""
        return [{"role": "system", "content": "Placeholder - needs implementation"}]
    
    def _build_static_scene_verify_message(self, config: Dict, code: str, render_path: str, current_image_path_ref: List, format_prompt: str) -> Dict:
        """Build verify message for static_scene mode."""
        # Similar to blendergym_hard_verify_message but for static scenes
        verify_message = {"role": "user", "content": [{"type": "text", "text": f"Please analyze the current state:\n"}]}
        
        if os.path.isdir(render_path):
            view1_path = os.path.join(render_path, 'render1.png')
            view2_path = os.path.join(render_path, 'render2.png')
        else:
            view1_path = render_path
            view2_path = None
            
        if os.path.exists(view1_path):
            verify_message["content"].append({"type": "text", "text": "Current scene view 1:"})
            verify_message["content"].append({"type": "image_url", "image_url": {"url": get_image_base64(view1_path)}})
            
        if view2_path and os.path.exists(view2_path):
            verify_message["content"].append({"type": "text", "text": "Current scene view 2:"})
            verify_message["content"].append({"type": "image_url", "image_url": {"url": get_image_base64(view2_path)}})
            
        # Add target image
        target_image_path = config.get("target_image_path")
        if target_image_path and os.path.exists(target_image_path):
            verify_message["content"].append({"type": "text", "text": "Target scene:"})
            verify_message["content"].append({"type": "image_url", "image_url": {"url": get_image_base64(target_image_path)}})
            
        verify_message["content"].append({"type": "text", "text": f"Code: {code}"})
        verify_message["content"].append({"type": "text", "text": format_prompt})
        
        return verify_message
    
    def _build_dynamic_scene_verify_message(self, config: Dict, code: str, render_path: str, current_image_path_ref: List, format_prompt: str) -> Dict:
        """Build verify message for dynamic_scene mode."""
        # Similar to static_scene_verify_message but for dynamic scenes with animation
        verify_message = {"role": "user", "content": [{"type": "text", "text": f"Please analyze the current dynamic scene state:\n"}]}
        
        if os.path.isdir(render_path):
            view1_path = os.path.join(render_path, 'render1.png')
            view2_path = os.path.join(render_path, 'render2.png')
        else:
            view1_path = render_path
            view2_path = None
            
        if os.path.exists(view1_path):
            verify_message["content"].append({"type": "text", "text": "Current dynamic scene view 1:"})
            verify_message["content"].append({"type": "image_url", "image_url": {"url": get_image_base64(view1_path)}})
            
        if view2_path and os.path.exists(view2_path):
            verify_message["content"].append({"type": "text", "text": "Current dynamic scene view 2:"})
            verify_message["content"].append({"type": "image_url", "image_url": {"url": get_image_base64(view2_path)}})
            
        # Add target image
        target_image_path = config.get("target_image_path")
        if target_image_path and os.path.exists(target_image_path):
            verify_message["content"].append({"type": "text", "text": "Target dynamic scene:"})
            verify_message["content"].append({"type": "image_url", "image_url": {"url": get_image_base64(target_image_path)}})
            
        verify_message["content"].append({"type": "text", "text": f"Code: {code}"})
        verify_message["content"].append({"type": "text", "text": format_prompt})
        
        return verify_message