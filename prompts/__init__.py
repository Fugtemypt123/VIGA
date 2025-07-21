from autopresent import *
from blendergym import *

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
        'hints': {
            'generator': autopresent_generator_hints,
            'verifier': autopresent_verifier_hints
        },
        'system': {
            'generator': autopresent_generator_system,
            'verifier': autopresent_verifier_system
        },
        'format':{
            'generator': autopresent_generator_format,
            'verifier': autopresent_verifier_format
        },
        'api_library': autopresent_api_library
    }
}