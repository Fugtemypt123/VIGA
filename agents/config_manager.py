"""
Configuration Manager for AgenticVerifier
Centralizes configuration logic and provides clear boolean flags for complex conditions.
"""
import os
from typing import Dict, Any, Optional


class ConfigManager:
    """
    Centralized configuration manager that provides clear boolean flags
    and configuration logic for complex conditions.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Basic configuration
        self.mode = config.get("mode")
        self.task_name = config.get("task_name")
        self.level = self._extract_level()
        
        # Server configuration flags
        self.is_blender_mode = self.mode in ["blendergym", "blendergym-hard", "static_scene", "dynamic_scene"]
        self.is_slides_mode = self.mode == "autopresent"
        self.is_html_mode = self.mode == "design2code"
        self.is_blendergym_hard_mode = self.mode == "blendergym-hard"
        self.is_static_scene_mode = self.mode == "static_scene"
        self.is_dynamic_scene_mode = self.mode == "dynamic_scene"
        
        # Level-specific flags for blendergym-hard
        self.is_level4 = self.is_blendergym_hard_mode and self.level == "level4"
        self.requires_scene_info = self.is_level4
        
        # Tool configuration flags
        self.has_meshy_tools = self.is_blendergym_hard_mode and self.is_level4
        self.has_execute_tools = self.mode in ["blendergym", "autopresent", "design2code", "static_scene", "dynamic_scene"]
        self.has_verifier_tools = self.mode in ["blendergym", "blendergym-hard", "static_scene", "dynamic_scene"]
        
        # Server paths
        self.blender_server_path = config.get("blender_server_path")
        self.slides_server_path = config.get("slides_server_path")
        self.html_server_path = config.get("html_server_path")
        self.image_server_path = config.get("image_server_path")
        self.scene_server_path = config.get("scene_server_path")
        
        # File paths
        self.init_code_path = config.get("init_code_path")
        self.init_image_path = config.get("init_image_path")
        self.target_image_path = config.get("target_image_path")
        self.target_description = config.get("target_description")
        self.blender_file = config.get("blender_file")
        
        # API keys
        self.api_key = config.get("api_key")
        self.meshy_api_key = config.get("meshy_api_key")
        self.va_api_key = config.get("va_api_key")
        
        # Model configuration
        self.vision_model = config.get("vision_model")
        self.api_base_url = config.get("api_base_url")
        
        # Output configuration
        self.output_dir = config.get("output_dir")
        self.thought_save = config.get("thought_save")
        self.max_rounds = config.get("max_rounds", 10)
        
        # GPU configuration
        self.gpu_devices = config.get("gpu_devices")
        
        # Save configuration
        self.save_blender_file = config.get("save_blender_file", False)
        
        # Tool server paths
        self.generator_script = config.get("generator_script", "agents/generator_mcp.py")
        self.verifier_script = config.get("verifier_script", "agents/verifier_mcp.py")
    
    def _extract_level(self) -> Optional[str]:
        """Extract level from task_name for blendergym-hard mode."""
        if self.is_blendergym_hard_mode and self.task_name:
            return self.task_name.split('-')[0]
        return None
    
    def get_server_type_and_path(self) -> tuple[Optional[str], Optional[str]]:
        """Get server type and path based on mode."""
        if self.is_blender_mode:
            return "blender", self.blender_server_path
        elif self.is_slides_mode:
            return "slides", self.slides_server_path
        elif self.is_html_mode:
            return "html", self.html_server_path
        else:
            return None, None
    
    def get_verifier_server_type_and_path(self) -> tuple[Optional[str], Optional[str]]:
        """Get verifier server type and path based on mode."""
        if self.mode in ["blendergym", "autopresent", "design2code"]:
            return "image", self.image_server_path
        elif self.mode in ["blendergym-hard", "static_scene", "dynamic_scene"]:
            return "scene", self.scene_server_path
        else:
            return None, None
    
    def get_target_image_path_for_mode(self) -> Optional[str]:
        """Get the appropriate target image path based on mode and level."""
        if not self.target_image_path:
            return None
            
        if self.is_blendergym_hard_mode:
            if os.path.isdir(self.target_image_path):
                if self.level == "level1":
                    return os.path.join(self.target_image_path, 'style1.png')
                else:
                    return os.path.join(self.target_image_path, 'visprompt1.png')
            else:
                return self.target_image_path
        elif self.is_blender_mode:
            if os.path.isdir(self.target_image_path):
                return os.path.join(self.target_image_path, 'render1.png')
            else:
                return self.target_image_path
        elif self.is_static_scene_mode or self.is_dynamic_scene_mode:
            # For static_scene and dynamic_scene modes, return the target image path as-is
            return self.target_image_path
        else:
            return self.target_image_path
    
    def get_init_image_path_for_mode(self) -> Optional[str]:
        """Get the appropriate initial image path based on mode."""
        if not self.init_image_path:
            return None
            
        if self.is_blender_mode:
            if os.path.isdir(self.init_image_path):
                return os.path.join(self.init_image_path, 'render1.png')
            else:
                return self.init_image_path
        else:
            return self.init_image_path
    
    def should_add_scene_info(self) -> bool:
        """Check if scene info should be added to the prompt."""
        return self.requires_scene_info
    
    def get_scene_info_config(self) -> Dict[str, Any]:
        """Get configuration for scene info if needed."""
        if self.should_add_scene_info():
            return {
                "task_name": self.task_name,
                "blender_file": self.blender_file
            }
        return {}
    
    def get_executor_setup_config(self) -> Dict[str, Any]:
        """Get configuration for executor setup."""
        setup_config = {
            "mode": self.mode,
            "api_key": self.api_key,
            "task_name": self.task_name,
            "max_rounds": self.max_rounds,
            "init_code_path": self.init_code_path,
            "init_image_path": self.init_image_path,
            "target_image_path": self.target_image_path,
            "target_description": self.target_description,
            "api_base_url": self.api_base_url,
            "thought_save": self.thought_save,
            "output_dir": self.output_dir,
            "gpu_devices": self.gpu_devices,
        }
        
        # Add mode-specific configurations
        if self.is_blender_mode:
            setup_config.update({
                "blender_server_path": self.blender_server_path,
                "blender_command": self.config.get("blender_command"),
                "blender_file": self.blender_file,
                "blender_script": self.config.get("blender_script"),
                "render_save": os.path.join(self.output_dir, "renders"),
                "script_save": os.path.join(self.output_dir, "scripts"),
                "blender_save": os.path.join(self.output_dir, "blender_file.blend") if self.save_blender_file else None,
                "meshy_api_key": self.meshy_api_key,
                "va_api_key": self.va_api_key,
            })
        elif self.is_slides_mode:
            setup_config.update({
                "slides_server_path": self.slides_server_path,
            })
        elif self.is_html_mode:
            setup_config.update({
                "html_server_path": self.html_server_path,
            })
        
        return setup_config
    
    def get_verifier_setup_config(self) -> Dict[str, Any]:
        """Get configuration for verifier setup."""
        verifier_config = {
            "mode": self.mode,
            "vision_model": self.vision_model,
            "api_key": self.api_key,
            "max_rounds": self.max_rounds,
            "task_name": self.task_name,
            "target_image_path": self.target_image_path,
            "target_description": self.target_description,
            "thought_save": self.thought_save,
            "api_base_url": self.api_base_url,
            "image_server_path": self.image_server_path,
            "scene_server_path": self.scene_server_path,
            "blender_file": os.path.join(self.output_dir, "blender_file.blend") if self.save_blender_file else None,
            "web_server_path": None,  # Not used in current implementation
        }
        
        return verifier_config
    
    def validate_configuration(self) -> tuple[bool, Optional[str]]:
        """Validate the configuration and return (is_valid, error_message)."""
        # Check required fields
        if not self.mode:
            return False, "Mode is required"
        
        if not self.api_key:
            return False, "API key is required"
        
        if not self.vision_model:
            return False, "Vision model is required"
        
        # Check mode-specific requirements
        if self.is_blender_mode and not self.blender_server_path:
            return False, "Blender server path is required for blender modes"
        
        if self.is_slides_mode and not self.slides_server_path:
            return False, "Slides server path is required for autopresent mode"
        
        if self.is_html_mode and not self.html_server_path:
            return False, "HTML server path is required for design2code mode"
        
        if self.has_verifier_tools and not self.image_server_path and not self.scene_server_path:
            return False, "Verifier server path is required"
        
        return True, None
