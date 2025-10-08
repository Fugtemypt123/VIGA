from typing import Dict, List

class ToolManager:
    """Helper class for managing tool definitions and configurations."""
    
    @staticmethod
    def get_generator_tools(mode: str, task_name: str) -> List[Dict]:
        """Get available tools for the generator agent based on mode and task.
        Policy:
        - All modes: include init_generate, execute_and_evaluate, rag_query
        - Only static_scene and dynamic_scene: additionally include generate_and_download_3d_asset (Meshy)
        - No other tools included
        """
        if mode in ["blendergym", "autopresent", "design2code", "static_scene", "dynamic_scene", "blendergym-hard"]:
            # Add execute_and_evaluate tool for code execution modes
            exec_evaluate_tool = {
                "type": "function",
                "function": {
                    "name": "execute_and_evaluate",
                    "description": "Execute code modifications and trigger verifier evaluation. This tool combines code execution with automatic verification. Always use this tool when you want to execute your code changes.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "thought": {
                                "type": "string",
                                "description": "Analyze the current state and provide a clear plan for the required changes. Consider scene representation consistency and infinigen optimization opportunities."
                            },
                            "code_edition": {
                                "type": "string", 
                                "description": "Provide your code modifications in the following format:\n-: [lines to remove]\n+: [lines to add]\nFocus on scene consistency and use infinigen functions when appropriate."
                            },
                            "full_code": {
                                "type": "string",
                                "description": "Merge your code changes into the full code with proper formatting. Ensure consistent scene representation."
                            }
                        },
                        "required": ["thought", "code_edition", "full_code"]
                    }
                }
            }
            tools: List[Dict] = [exec_evaluate_tool]

            # init_plan tool
            tools.insert(0, {
                "type": "function",
                "function": {
                    "name": "init_plan",
                    "description": "Store a detailed scene plan for subsequent actions.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "detailed_description": {"type": "string", "description": "Comprehensive scene description including objects, relations, and initial layout"}
                        },
                        "required": ["detailed_description"]
                    }
                }
            })

            # RAG tool (query Blender/Infinigen knowledge and examples)
            rag_tool = {
                "type": "function",
                "function": {
                    "name": "rag_query",
                    "description": "Query Blender/Infinigen RAG to fetch related APIs/snippets and optional enhanced examples.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "instruction": {"type": "string", "description": "Instruction, e.g., '物理规律地放置一个立方体'"},
                            "use_enhanced": {"type": "boolean", "description": "Use OpenAI-enhanced generation if available", "default": False}
                        },
                        "required": ["instruction"]
                    }
                }
            }
            tools.append(rag_tool)

            # init_generate tools (image-based initialization helpers)
            init_generate_tool = {
                "type": "function",
                "function": {
                    "name": "initialize_generator",
                    "description": "Initialize image generation helper with API key and base url.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "vision_model": {"type": "string", "description": "OpenAI-compatible model name"},
                            "api_key": {"type": "string", "description": "OpenAI API key"},
                            "api_base_url": {"type": "string", "description": "OpenAI-compatible base URL (optional)"}
                        }
                    }
                }
            }
            tools.append(init_generate_tool)

            exec_pil_tool = {
                "type": "function",
                "function": {
                    "name": "exec_pil_code",
                    "description": "Execute PIL Python code and return base64 image or result.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {"type": "string", "description": "Python code using PIL; set `result` in code"}
                        },
                        "required": ["code"]
                    }
                }
            }
            tools.append(exec_pil_tool)

            gen_scene_desc_tool = {
                "type": "function",
                "function": {
                    "name": "generate_scene_description",
                    "description": "Generate a detailed scene description from an image.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "image_path": {"type": "string", "description": "Path to an image file"}
                        },
                        "required": ["image_path"]
                    }
                }
            }
            tools.append(gen_scene_desc_tool)

            gen_init_suggest_tool = {
                "type": "function",
                "function": {
                    "name": "generate_initialization_suggestions",
                    "description": "Generate initialization suggestions from an image.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "image_path": {"type": "string", "description": "Path to an image file"}
                        },
                        "required": ["image_path"]
                    }
                }
            }
            tools.append(gen_init_suggest_tool)

            # Meshy tool ONLY for static_scene and dynamic_scene
            if mode in ["static_scene", "dynamic_scene"]:
                meshy_tool = {
                    "type": "function",
                    "function": {
                        "name": "generate_and_download_3d_asset",
                        "description": "Generate and download a 3D asset using Meshy Text-to-3D API or load from local assets dir if available.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "object_name": {"type": "string", "description": "Asset/object name, e.g., 'table', 'chair'"},
                                "reference_type": {"type": "string", "enum": ["text", "image"], "description": "Reference type for generation"},
                                "object_description": {"type": "string", "description": "Detailed description when using text reference"}
                            },
                            "required": ["object_name", "reference_type"]
                        }
                    }
                }
                tools.append(meshy_tool)

            return tools
        else:
            return []
    
    @staticmethod
    def get_verifier_tools(mode: str, task_name: str) -> List[Dict]:
        """Get available tools for the verifier agent based on mode.
        Policy:
        - All modes: include investigator and scene utility tools only (no image comparison tool)
        - Only blendergym-hard, static_scene, dynamic_scene: investigator is applicable per prompts
        - No other tools included
        """
        # Base tools for ALL modes per prompt: setup_camera, investigate, set_object_visibility, set_key_frame, end
        tools: List[Dict] = [
            {
                "type": "function",
                "function": {
                    "name": "setup_camera",
                    "description": "Setup an observer camera to a canonical view.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "view": {"type": "string", "enum": ["top","front","side","oblique"]}
                        },
                        "required": ["view"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "investigate",
                    "description": "Investigate scene with a unified tool: focus, zoom, move.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "operation": {"type": "string", "enum": ["focus","zoom","move"]},
                            "object_name": {"type": "string"},
                            "direction": {"type": "string", "enum": ["in","out","up","down","left","right"]}
                        },
                        "required": ["operation"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "set_object_visibility",
                    "description": "Toggle visibility of specific scene objects to isolate elements.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "show_object_list": {"type": "array", "items": {"type": "string"}},
                            "hide_object_list": {"type": "array", "items": {"type": "string"}}
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "set_key_frame",
                    "description": "Jump to a specific keyframe index and render a view.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "target_frame": {"type": "integer"}
                        },
                        "required": ["target_frame"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "end",
                    "description": "End the current review round and return structured results.",
                    "parameters": {"type": "object", "properties": {}}
                }
            }
        ]

        # Investigator tools only for specified modes
        # investigator tools already included above per new prompts

        return tools
