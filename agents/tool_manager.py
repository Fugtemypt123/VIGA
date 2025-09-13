from typing import Dict, List

class ToolManager:
    """Helper class for managing tool definitions and configurations."""
    
    @staticmethod
    def get_generator_tools(mode: str, task_name: str) -> List[Dict]:
        """Get available tools for the generator agent based on mode and task."""
        return []
        # if mode == "blendergym-hard":
        #     # For blendergym-hard mode, determine tools based on level
        #     level = task_name.split('-')[0]
        #     tools = []
            
        #     # Define tool definitions
        #     meshy_tool = {
        #         "type": "function",
        #         "function": {
        #             "name": "generate_3d_asset",
        #             "description": "Generate and import a 3D asset into the Blender scene using Meshy Text-to-3D API. This tool can create objects based on text descriptions and automatically import them into the current scene.",
        #             "parameters": {
        #                 "type": "object",
        #                 "properties": {
        #                     "description": {
        #                         "type": "string", 
        #                         "description": "Text description of the 3D asset to generate (e.g., 'a wooden chair', 'a modern table', 'a decorative plant')"
        #                     },
        #                     "location": {
        #                         "type": "string", 
        #                         "description": "Position where to place the asset in the scene, format: 'x,y,z' (e.g., '2,0,0')",
        #                         "default": "0,0,0"
        #                     },
        #                     "scale": {
        #                         "type": "number", 
        #                         "description": "Scale factor for the asset (e.g., 1.0 for normal size, 2.0 for double size)",
        #                         "default": 1.0
        #                     },
        #                     "refine": {
        #                         "type": "boolean", 
        #                         "description": "Whether to apply texture refinement after initial generation (takes longer but produces better quality)",
        #                         "default": True
        #                     }
        #                 },
        #                 "required": ["description"]
        #             }
        #         }
        #     }
            
        #     investigator_tool = {
        #         "type": "function",
        #         "function": {
        #             "name": "investigate_3d",
        #             "description": "A tool for detailed 3D scene investigation with the following operations: focus, zoom, move.",
        #             "parameters": {
        #                 "type": "object",
        #                 "properties": {
        #                     "operation": {"type": "string", "enum": ["focus", "zoom", "move"], "description": "The operation to perform on the 3D scene."},
        #                     "object_name": {"type": "string", "description": "The name of the object to focus on (only for focus operation)."},
        #                     "direction": {"type": "string", "enum": ["in", "out", "up", "down", "left", "right"], "description": "The direction to move the camera (only for zoom and move operation)."}
        #                 },
        #                 "required": ["operation"]
        #             }
        #         }
        #     }
            
        #     # Add tools based on level
        #     if level == "level1":
        #         # Only investigator tool (tool 3)
        #         tools.append(investigator_tool)
        #     elif level == "level2":
        #         # Only blender code executor (tool 2) - no tools needed as it's handled by exec_script
        #         pass
        #     elif level == "level3":
        #         # Blender code executor (tool 2) + investigator tool (tool 3)
        #         tools.append(investigator_tool)
        #     elif level == "level4":
        #         # All tools: meshy (tool 1) + blender code executor (tool 2) + investigator tool (tool 3)
        #         tools.append(meshy_tool)
        #         tools.append(investigator_tool)
            
        #     return tools
        # else:
        #     return []
    
    @staticmethod
    def get_verifier_tools(mode: str, task_name: str) -> List[Dict]:
        """Get available tools for the verifier agent based on mode."""
        if mode == "blendergym":
            return [{
                "type": "function",
                "function": {
                    "name": "compare_images",
                    "description": "A tool for comparing current images and the target images, and identifying their visual differences. This tool will automatically select suitable images for comparison, please always call this tool first."
                }
            }]
        elif mode == "blendergym-hard":
            return [{
                "type": "function",
                "function": {
                    "name": "investigate_3d",
                    "description": "A tool for detailed 3D scene investigation with the following operations: focus, zoom, move.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "operation": {"type": "string", "enum": ["focus", "zoom", "move"], "description": "The operation to perform on the 3D scene."},
                            "object_name": {"type": "string", "description": "The name of the object to focus on (only for focus operation)."},
                            "direction": {"type": "string", "enum": ["in", "out", "up", "down", "left", "right"], "description": "The direction to move the camera (only for zoom and move operation)."}
                        },
                        "required": ["operation"]
                    }
                }
            }]
        else:
            return []
