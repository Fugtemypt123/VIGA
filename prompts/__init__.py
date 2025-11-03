from .autopresent import *
from .blendergym import *
from .blenderstudio import *
from .design2code import *
from .static_scene import *
from .dynamic_scene import *
from .prompt_manager import PromptManager, prompt_manager

# Legacy prompts_dict for backward compatibility
prompts_dict = {
    'static_scene': {
        'system': {
            'generator': static_scene_generator_system,
            'verifier': static_scene_verifier_system
        }
    },
    'dynamic_scene': {
        'system': {
            'generator': dynamic_scene_generator_system,
            'verifier': dynamic_scene_verifier_system
        }
    },
    'autopresent': {
        'system': {
            'generator': autopresent_generator_system,
            'verifier': autopresent_verifier_system
        }
    },
    'blendergym': {
        'system': {
            'generator': blendergym_generator_system,
            'verifier': blendergym_verifier_system
        }
    },
    'design2code': {
        'system': {
            'generator': design2code_generator_system,
            'verifier': design2code_verifier_system
        }
    },
    'blenderstudio': {
        'system': {
            'generator': blendergym_generator_system,
            'verifier': blendergym_verifier_system
        }
    }
}

# Export the new prompt manager as the primary interface
__all__ = ['prompt_manager', 'PromptManager', 'prompts_dict']