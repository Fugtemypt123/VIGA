"""
Configuration Manager for AgenticVerifier
Centralizes configuration logic and provides clear boolean flags for complex conditions.
"""
import os
from typing import Dict, Any, Optional, List


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
        
        # Server configuration flags (must be set before _extract_level)
        self.is_blender_mode = self.mode in ["blendergym", "blendergym-hard", "static_scene", "dynamic_scene"]
        self.is_slides_mode = self.mode == "autopresent"
        self.is_html_mode = self.mode == "design2code"
        self.is_blendergym_hard_mode = self.mode == "blendergym-hard"
        self.is_static_scene_mode = self.mode == "static_scene"
        self.is_dynamic_scene_mode = self.mode == "dynamic_scene"
        
        # Extract level after flags are set
        self.level = self._extract_level()
        
        # Level-specific flags for blendergym-hard
        self.is_level4 = self.is_blendergym_hard_mode and self.level == "level4"
        self.requires_scene_info = self.is_level4
        
        # Tool configuration flags
        self.has_meshy_tools = self.is_blendergym_hard_mode and self.is_level4
        self.has_execute_tools = self.mode in ["blendergym", "autopresent", "design2code", "static_scene", "dynamic_scene"]
        self.has_verifier_tools = self.mode in ["blendergym", "blendergym-hard", "static_scene", "dynamic_scene"]
        
        # Tool server scripts (lists)
        self.generator_tools = config.get("generator_tools", [])
        self.verifier_tools = config.get("verifier_tools", [])
        
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
    
    def get_generator_server_type_and_path(self) -> tuple[Optional[str], Optional[str]]:
        """Deprecated single-server helper: return first generator tool if provided, else None."""
        if self.generator_tools:
            return "blender", self.generator_tools[0]
        return None, None
    
    def get_verifier_server_type_and_path(self) -> tuple[Optional[str], Optional[str]]:
        """Deprecated single-server helper: return first verifier tool if provided, else None."""
        if self.verifier_tools:
            return "image", self.verifier_tools[0]
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
    
    def get_assets_path(self) -> Optional[str]:
        """Get the assets directory path for static_scene and dynamic_scene modes."""
        if self.is_static_scene_mode or self.is_dynamic_scene_mode:
            # Extract task path from target_image_path first, then output_dir
            if self.target_image_path:
                # target_image_path format: data/static_scene/task_name/target.png
                task_path = os.path.dirname(self.target_image_path)
                return os.path.join(task_path, "assets")
            elif self.output_dir:
                # output_dir format: output/static_scene/timestamp/task_name
                # For output_dir, we need to look for assets in the original data directory
                # This is a fallback and may not work in all cases
                return None
        return None
    
    def get_available_assets(self) -> List[str]:
        """Get list of available .glb assets for static_scene and dynamic_scene modes."""
        assets_path = self.get_assets_path()
        if not assets_path or not os.path.exists(assets_path):
            return []
        
        assets = []
        for file in os.listdir(assets_path):
            if file.endswith('.glb'):
                assets.append(file)
        return sorted(assets)
    
    def get_scene_info_config(self) -> Dict[str, Any]:
        """Get configuration for scene info if needed."""
        if self.should_add_scene_info():
            return {
                "task_name": self.task_name,
                "blender_file": self.blender_file
            }
        return {}

    # ===== Tool servers (generator/verifier) mapping helpers =====
    def _infer_server_type(self, script_path: str) -> Optional[str]:
        """Infer server type key from a tool script path."""
        if not script_path:
            return None
        name = os.path.basename(script_path)
        if "exec_blender" in name:
            return "blender"
        if "meshy" in name:
            # Expose meshy tools under blender server_type to match current handlers
            return "blender"
        if "init_generate" in name or "init_verify" in name or "image" in name:
            return "image"
        if "investigator" in name or "scene" in name:
            return "scene"
        if "rag" in name or "web" in name:
            return "web"
        if "exec_html" in name or "html" in name:
            return "html"
        if "exec_slides" in name or "slides" in name:
            return "slides"
        return None

    def get_generator_tool_servers(self) -> Dict[str, str]:
        """Map generator tool server_type -> script path (first occurrence wins)."""
        servers: Dict[str, str] = {}
        for p in self.generator_tools:
            server_type = self._infer_server_type(p)
            if server_type and server_type not in servers:
                servers[server_type] = p
        return servers

    def get_verifier_tool_servers(self) -> Dict[str, str]:
        """Map verifier tool server_type -> script path (first occurrence wins)."""
        servers: Dict[str, str] = {}
        for p in self.verifier_tools:
            server_type = self._infer_server_type(p)
            if server_type and server_type not in servers:
                servers[server_type] = p
        return servers
    
    def get_generator_setup_config(self) -> Dict[str, Any]:
        """Get configuration for executor setup."""
        setup_config = {
            "mode": self.mode,
            "vision_model": self.vision_model,
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
            # Tool servers map for multi-server connection
            "tool_servers": self.get_generator_tool_servers(),
        }
        
        # Add mode-specific configurations
        if self.is_blender_mode:
            blender_config = {
                "blender_server_path": None,
                "blender_command": self.config.get("blender_command"),
                "blender_file": self.blender_file,
                "blender_script": self.config.get("blender_script"),
                "render_save": os.path.join(self.output_dir, "renders"),
                "script_save": os.path.join(self.output_dir, "scripts"),
                "blender_save": os.path.join(self.output_dir, "blender_file.blend") if self.save_blender_file else None,
                "meshy_api_key": self.meshy_api_key,
                "va_api_key": self.va_api_key,
            }
            
            # Add task assets directory for static_scene and dynamic_scene modes
            if self.is_static_scene_mode or self.is_dynamic_scene_mode:
                assets_path = self.get_assets_path()
                if assets_path:
                    blender_config["task_assets_dir"] = assets_path
            
            setup_config.update(blender_config)
        # Multi-server provided via generator_tools/verifier_tools lists in main config_dict
        
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
            "image_server_path": None,
            "scene_server_path": None,
            "blender_file": self.blender_file,
            "web_server_path": None,  # Not used in current implementation
            # Tool servers map for multi-server connection
            "tool_servers": self.get_verifier_tool_servers(),
        }
        
        return verifier_config
    
    def validate_generator_configuration(self) -> tuple[bool, Optional[str]]:
        """Validate the configuration and return (is_valid, error_message)."""
        # Check required fields
        if not self.mode:
            return False, "Mode is required"
        
        if not self.api_key:
            return False, "API key is required"
        
        if not self.vision_model:
            return False, "Vision model is required"
        
        # No server path requirements; tools are connected on-demand
        
        return True, None
    
    def validate_verifier_configuration(self) -> tuple[bool, Optional[str]]:
        """Validate the configuration and return (is_valid, error_message)."""
        # Check required fields
        if not self.mode:
            return False, "Mode is required"
        
        if not self.api_key:
            return False, "API key is required"
        
        if not self.vision_model:
            return False, "Vision model is required"
        
        # No server path requirements for verifier
        
        return True, None
