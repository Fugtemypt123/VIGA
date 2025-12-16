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
            'verifier': static_scene_verifier_system,
            'generator_procedural': static_scene_generator_system_procedural,
            'generator_scene_graph': static_scene_generator_system_scene_graph,
            'verifier_procedural': static_scene_verifier_system_procedural,
            'verifier_scene_graph': static_scene_verifier_system_scene_graph,
            'generator_get_asset': static_scene_generator_system_get_asset,
            'verifier_get_asset': static_scene_verifier_system
        }
    },
    'dynamic_scene': {
        'system': {
            'generator': dynamic_scene_generator_system,
            'verifier': dynamic_scene_verifier_system,
            'generator_init': dynamic_scene_generator_system_init,
            'verifier_init': dynamic_scene_verifier_system,
        }
    },
    'autopresent': {
        'system': {
            'generator': autopresent_generator_system,
            'verifier': autopresent_verifier_system,
            'generator_no_tools': autopresent_generator_system_no_tools,
            'verifier_no_tools': autopresent_verifier_system_no_tools
        }
    },
    'blendergym': {
        'system': {
            'generator': blendergym_generator_system,
            'verifier': blendergym_verifier_system,
            'generator_no_tools': blendergym_generator_system_no_tools,
            'verifier_no_tools': blendergym_verifier_system_no_tools
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
            'generator': blenderstudio_generator_system,
            'verifier': blenderstudio_verifier_system,
            'generator_no_tools': blenderstudio_generator_system_no_tools,
            'verifier_no_tools': blenderstudio_verifier_system_no_tools
        }
    }
}

# Export the new prompt manager as the primary interface
__all__ = ['prompt_manager', 'PromptManager', 'prompts_dict']