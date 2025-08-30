from .autopresent import *
from .blendergym import *
from .blendergym_hard import *
from .demo import *

prompts_dict = {
    'blendergym': {
        'hints': {
            'generator': blendergym_generator_hints,
            'verifier': blendergym_verifier_hints
        },
        'system': {
            'generator': blendergym_generator_system,
            'verifier': blendergym_verifier_system
        },
        'format':{
            'generator': blendergym_generator_format,
            'verifier': blendergym_verifier_format
        }
    },
    'autopresent': {
        'system': {
            'generator': autopresent_generator_system,
            'verifier': autopresent_verifier_system
        },
        'format':{
            'generator': autopresent_generator_format,
            'verifier': autopresent_verifier_format
        },
        'api_library': autopresent_api_library,
        'hints': autopresent_hints
    },
    'blendergym-hard': {
        'hints': blendergym_hard_hints
    },
    'demo': {
        'hints': demo_hints
    }
}