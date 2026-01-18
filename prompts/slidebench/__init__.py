"""SlideBench prompts module."""

from .generator import autopresent_generator_system
from .verifier import autopresent_verifier_system
from .generator import autopresent_generator_system_no_tools
from .verifier import autopresent_verifier_system_no_tools

__all__ = ['autopresent_generator_system', 'autopresent_verifier_system', 'autopresent_generator_system_no_tools', 'autopresent_verifier_system_no_tools']

