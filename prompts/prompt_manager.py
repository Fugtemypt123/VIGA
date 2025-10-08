"""
Unified Prompt Manager for AgenticVerifier
Centralizes all prompt loading and management logic.
"""
import os
from typing import Dict, List, Optional, Any


class PromptManager:
    """
    Centralized prompt manager that handles all prompt loading and organization.
    Replaces dynamic prompt building with static, modular prompt loading.
    """
    
    def __init__(self):
        # Initialize prompts as None, will be set later
        self.prompts = None
    
    def _ensure_prompts_loaded(self):
        """Ensure prompts are loaded, avoiding circular import."""
        if self.prompts is None:
            # Import prompts_dict here to avoid circular import
            from . import prompts_dict
            self.prompts = prompts_dict
    
    def get_system_prompt(self, mode: str, agent_type: str, level: Optional[str] = None) -> str:
        """Get system prompt for specified mode, agent type, and optional level."""
        self._ensure_prompts_loaded()
        if mode not in self.prompts:
            raise ValueError(f"Mode {mode} not supported")
        
        mode_prompts = self.prompts[mode]
        
        if agent_type not in mode_prompts.get('system', {}):
            raise ValueError(f"Agent type {agent_type} not supported for mode {mode}")
        
        system_prompts = mode_prompts['system'][agent_type]
        
        # Handle level-specific prompts (for blendergym-hard)
        if isinstance(system_prompts, dict) and level:
            if level not in system_prompts:
                raise ValueError(f"Level {level} not supported for mode {mode}")
            return system_prompts[level]
        elif isinstance(system_prompts, str):
            return system_prompts
        else:
            raise ValueError(f"Invalid system prompt format for mode {mode}, agent {agent_type}")
    
    def get_format_prompt(self, mode: str, agent_type: str, level: Optional[str] = None) -> str:
        """Get format prompt for specified mode, agent type, and optional level."""
        self._ensure_prompts_loaded()
        if mode not in self.prompts:
            return None
        
        mode_prompts = self.prompts[mode]
        
        if agent_type not in mode_prompts.get('format', {}):
            return None
        
        format_prompts = mode_prompts['format'][agent_type]
        
        # Handle level-specific prompts (for blendergym-hard)
        if isinstance(format_prompts, dict) and level:
            if level not in format_prompts:
                return None
            return format_prompts[level]
        elif isinstance(format_prompts, str):
            return format_prompts
        else:
            return None
    
    def get_hints(self, mode: str, agent_type: str, task_name: Optional[str] = None, level: Optional[str] = None) -> Optional[str]:
        """Get hints for specified mode, agent type, task name, and optional level."""
        self._ensure_prompts_loaded()
        if mode not in self.prompts:
            return None
        
        mode_prompts = self.prompts[mode]
        
        if 'hints' not in mode_prompts:
            return None
        
        hints = mode_prompts['hints']
        
        # Handle different hint structures
        if isinstance(hints, dict):
            if agent_type in hints:
                agent_hints = hints[agent_type]
                
                # Handle level-specific hints (for blendergym-hard)
                if isinstance(agent_hints, dict) and level:
                    return agent_hints.get(level)
                # Handle task-specific hints (for blendergym)
                elif isinstance(agent_hints, dict) and task_name:
                    return agent_hints.get(task_name)
                # Handle simple string hints
                elif isinstance(agent_hints, str):
                    return agent_hints
            else:
                # Handle mode-level hints (for autopresent, design2code)
                if isinstance(hints, str):
                    return hints
        elif isinstance(hints, str):
            return hints
        
        return None
    
    def get_api_library(self, mode: str) -> Optional[str]:
        """Get API library documentation for specified mode."""
        self._ensure_prompts_loaded()
        if mode not in self.prompts:
            return None
        
        return self.prompts[mode].get('api_library')
    
    def get_tool_example(self, mode: str) -> Optional[str]:
        """Get tool usage examples for specified mode."""
        self._ensure_prompts_loaded()
        if mode not in self.prompts:
            return None
        
        return self.prompts[mode].get('tool_example')
    
    def get_all_prompts(self, mode: str, agent_type: str, task_name: Optional[str] = None, level: Optional[str] = None) -> Dict[str, Any]:
        """Get all prompts for a given configuration in a single call."""
        return {
            'system': self.get_system_prompt(mode, agent_type, level),
            'format': self.get_format_prompt(mode, agent_type, level),
            'hints': self.get_hints(mode, agent_type, task_name, level),
            'api_library': self.get_api_library(mode),
            'tool_example': self.get_tool_example(mode)
        }
    
    def is_mode_supported(self, mode: str) -> bool:
        """Check if a mode is supported."""
        self._ensure_prompts_loaded()
        return mode in self.prompts
    
    def get_supported_modes(self) -> List[str]:
        """Get list of all supported modes."""
        self._ensure_prompts_loaded()
        return list(self.prompts.keys())
    
    def get_supported_agent_types(self, mode: str) -> List[str]:
        """Get list of supported agent types for a given mode."""
        self._ensure_prompts_loaded()
        if mode not in self.prompts:
            return []
        
        system_prompts = self.prompts[mode].get('system', {})
        return list(system_prompts.keys())


# Global instance for easy access
prompt_manager = PromptManager()
