from typing import Dict, List

verifier_tool_hints = """About how to use the tool: Our investigator_3d tool provides three operations: 
(1) focus on an object: Let your camera focus on a key object, suitable for situations where a certain object of interest is known. Always call (1) to obtain a key object before calling (2)(3).
(2) zoom in/out: Zoom the camera in/out. Generally speaking, zooming in is suitable for observing a small part (such as the object you want to move in a small corner of the scene), while zooming out is suitable for observing the whole picture (such as observing the relative position of the object in the scene); 
(3) move: Rotate the camera up/down/left/right. Please note: This rotation is performed on a sphere with a fixed radius of the distance from the current camera to the target object. If you want to adjust the distance from the current camera to the target object, please do not use this operation."""

class ToolManager:
    """Helper class for managing tool definitions and configurations."""
    
    @staticmethod
    def get_generator_tools(mode: str, task_name: str) -> List[Dict]:
        """Get available tools for the generator agent based on mode and task."""
        if mode == "blendergym-hard":
            # For blendergym-hard mode, determine tools based on level
            level = task_name.split('-')[0]
            if level == "level4":
                # Define tool definitions
                meshy_tool = {
                    "type": "function",
                    "function": {
                        "name": "generate_and_import_3d_asset",
                        "description": "Generate and import a 3D asset into the Blender scene using Meshy Text-to-3D API. This tool can create objects based on text descriptions and automatically import them into the current scene. The default coordinates for import are (0,0,0). You can adjust these coordinates in subsequent code.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "object_name": {
                                    "type": "string",
                                    "description": "The name of the object you want to generate. This is usually a brief description of two words or less. For example, 'Christmas tree', 'snowman', etc."
                                },
                                "reference_type": {
                                    "type": "string",
                                    "enum": ["text", "image"],
                                    "description": "You can choose to generate using text or images. If you use text, you need to provide a detailed description of the generated object. If you use an image, I will automatically crop the object in the image based on the object name."
                                },
                                "object_description": {
                                    "type": "string", 
                                    "description": "A detailed description of the object you want to generate. Include specific information such as color, shape, etc. The clearer the better. For example: 'a wooden tea table with four legs', 'a snowman with a black hat and a red scarf'. Only needed when you use text as reference."
                                }
                                
                            },
                            "required": ["object_name", "reference_type"]
                        }
                    }
                } 
                return [meshy_tool]
        else:
            return []
    
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
                    "description": "A tool for detailed 3D scene investigation with the following operations: focus, zoom, move." + verifier_tool_hints,
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
