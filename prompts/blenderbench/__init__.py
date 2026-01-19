"""BlenderBench prompts module."""

from .generator import blenderstudio_generator_system
from .verifier import blenderstudio_verifier_system
from .generator import blenderstudio_generator_system_no_tools
from .verifier import blenderstudio_verifier_system_no_tools

__all__ = ['blenderstudio_generator_system', 'blenderstudio_verifier_system', 'blenderstudio_generator_system_no_tools', 'blenderstudio_verifier_system_no_tools']

