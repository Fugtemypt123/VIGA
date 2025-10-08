import json
import logging
from typing import Dict, Any

class ToolHandler:
    """Helper class for handling tool calls in generator and verifier agents."""
    
    def __init__(self, tool_client):
        self.tool_client = tool_client
        self.blender_file_path = None
    
    async def handle_generator_tool_call(self, tool_call) -> Dict[str, Any]:
        """Handle tool calls from the generator agent."""
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)
        
        try:
            if function_name == "init_plan":
                result = await self.tool_client.call_tool(
                    tool_name="init_plan",
                    tool_args={"detailed_description": function_args.get("detailed_description", "")}
                )
                ok = result.get('status') == 'success'
                return {'text': json.dumps(result, ensure_ascii=False), 'success': ok}

            if function_name == "generate_and_download_3d_asset":
                result = await self.tool_client.call_tool("generate_and_download_3d_asset", {
                    "object_name": function_args.get("object_name", ""),
                    "reference_type": function_args.get("reference_type", ""),
                    "object_description": function_args.get("object_description", "")
                })
                
                if result.get("status") == "success":
                    object_name = function_args.get('object_name', '')
                    # output_content = f"# import a 3D asset: {object_name}\n# To edit this asset, please use `bpy.data.objects['{object_name}']`\n# To copy this asset (if you think you'll need more than one of it in the target image), please use `new_object = bpy.data.objects['{object_name}'].copy()\n# To delete this object (if you think the quality of this asset is really bad), please use `bpy.data.objects.remove(bpy.data.objects['{object_name}'])`\n"
                    return {
                        'text': f"Successfully generated and downloaded 3D asset: {object_name}. {result.get('message', '')}",
                        'success': True,
                        # 'output_content': output_content
                    }
                else:
                    return {
                        'text': f"Failed to generate and download 3D asset: {result.get('error', 'Unknown error')}",
                        'success': False,
                        # 'output_content': None
                    }

            elif function_name == "rag_query":
                result = await self.tool_client.call_tool(
                    tool_name="rag_query_tool",
                    tool_args={
                        "instruction": function_args.get("instruction", ""),
                        "use_enhanced": function_args.get("use_enhanced", False),
                        "use_doc_search": True
                    }
                )
                return {
                    'text': result.get('code_example') or json.dumps(result, ensure_ascii=False),
                    'success': result.get('status') == 'success'
                }

            elif function_name == "initialize_generator":
                result = await self.tool_client.call_tool(
                    tool_name="initialize_generator",
                    tool_args={
                        "args": {
                            "vision_model": function_args.get("vision_model"),
                            "api_key": function_args.get("api_key"),
                            "api_base_url": function_args.get("api_base_url")
                        }
                    }
                )
                return {'text': result.get('message', ''), 'success': result.get('status') == 'success'}

            elif function_name == "exec_pil_code":
                result = await self.tool_client.call_tool(
                    tool_name="exec_pil_code",
                    tool_args={"code": function_args.get("code", "")}
                )
                return {'text': json.dumps(result, ensure_ascii=False), 'success': result.get('status') == 'success'}

            elif function_name == "generate_scene_description":
                result = await self.tool_client.call_tool(
                    tool_name="generate_scene_description",
                    tool_args={"image_path": function_args.get("image_path", "")}
                )
                return {'text': result.get('description', ''), 'success': result.get('status') == 'success'}

            elif function_name == "generate_initialization_suggestions":
                # Single-image (init_generate) or dual-image (init_verify)
                if "image_path" in function_args:
                    result = await self.tool_client.call_tool(
                        tool_name="generate_initialization_suggestions",
                        tool_args={"image_path": function_args.get("image_path", "")}
                    )
                    return {'text': result.get('suggestions', ''), 'success': result.get('status') == 'success'}
                else:
                    result = await self.tool_client.call_tool(
                        tool_name="generate_initialization_suggestions",
                        tool_args={
                            "target_path": function_args.get("target_path", ""),
                            "current_path": function_args.get("current_path", "")
                        }
                    )
                    return {'text': result.get('suggestions', ''), 'success': result.get('status') == 'success'}

            elif function_name == "investigate_3d":
                op = function_args.get('operation')
                if op == 'focus':
                    result = await self.tool_client.call_tool("investigate", {
                        "operation": "focus",
                        "object_name": function_args.get("object_name", "")
                    })
                    if result.get("status") == "success":
                        return {
                            'text': f"Focused camera on object: {function_args.get('object_name', '')}. {result.get('message', '')}",
                            'success': True
                        }
                    else:
                        return {
                            'text': f"Failed to focus: {result.get('error', 'Unknown error')}",
                            'success': False
                        }
                elif op == 'zoom':
                    result = await self.tool_client.call_tool("investigate", {
                        "operation": "zoom",
                        "direction": function_args.get("direction", "")
                    })
                    if result.get("status") == "success":
                        return {
                            'text': f"Zoomed {function_args.get('direction', '')}. {result.get('message', '')}",
                            'success': True
                        }
                    else:
                        return {
                            'text': f"Failed to zoom: {result.get('error', 'Unknown error')}",
                            'success': False
                        }
                elif op == 'move':
                    result = await self.tool_client.call_tool("investigate", {
                        "operation": "move",
                        "direction": function_args.get("direction", "")
                    })
                    if result.get("status") == "success":
                        return {
                            'text': f"Moved camera {function_args.get('direction', '')}. {result.get('message', '')}",
                            'success': True
                        }
                    else:
                        return {
                            'text': f"Failed to move: {result.get('error', 'Unknown error')}",
                            'success': False
                        }
                else:
                    return {
                        'text': f"Unknown operation: {op}",
                        'success': False
                    }
            
            elif function_name == "execute_and_evaluate":
                thought = function_args.get("thought", "")
                code_edition = function_args.get("code_edition", "")
                full_code = function_args.get("full_code", "")
                
                try:
                    result = await self.tool_client.call_tool(
                        tool_name="exec_script",
                        tool_args={
                            "code": full_code,
                            "round_num": 1
                        }
                    )
                    
                    # Format the response to include all three components
                    response_text = f"Thought: {thought}\n\nCode Edition: {code_edition}\n\nExecution Result: {result.get('text', 'No output')}"
                    
                    return {
                        'text': response_text,
                        'success': result.get('success', True),
                        'thought': thought,
                        'code_edition': code_edition,
                        'full_code': full_code,
                        'execution_result': result
                    }
                except Exception as e:
                    return {
                        'text': f"Error executing script: {str(e)}",
                        'success': False,
                        'thought': thought,
                        'code_edition': code_edition,
                        'full_code': full_code
                    }
            
            else:
                return {'text': f"Unknown tool: {function_name}", 'success': False}
        except Exception as e:
            return {'text': f"Error executing tool: {str(e)}", 'success': False}
    
    async def handle_verifier_tool_call(self, tool_call, current_image_path: str = None, target_image_path: str = None, round_num: int = None) -> Dict[str, Any]:
        """Handle tool calls from the verifier agent."""
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)
        
        try:
            if function_name == "setup_camera":
                output = await self.tool_client.call_tool("setup_camera", {
                    "view": function_args.get("view", "top"),
                    "round_num": round_num
                })
                return {'text': f"Observer camera set to {function_args.get('view', 'top')}", 'image': output.get('image'), 'camera_position': output.get('camera_position')}
            elif function_name == "investigate_3d":
                op = function_args['operation']
                if op == 'focus':
                    output = await self.tool_client.call_tool("investigate", {
                        "operation": "focus",
                        "object_name": function_args.get("object_name", ""),
                        "round_num": round_num
                    })
                    return {'text': f"Focused camera on object: {function_args.get('object_name', '')}", 'image': output.get('image'), 'camera_position': output.get('camera_position')}
                elif op == 'zoom':
                    output = await self.tool_client.call_tool("investigate", {
                        "operation": "zoom",
                        "direction": function_args.get("direction", ""),
                        "round_num": round_num
                    })
                    return {'text': f"Zoomed {function_args.get('direction', '')}", 'image': output.get('image'), 'camera_position': output.get('camera_position')}
                elif op == 'move':
                    output = await self.tool_client.call_tool("investigate", {
                        "operation": "move",
                        "direction": function_args.get("direction", ""),
                        "round_num": round_num
                    })
                    return {'text': f"Moved camera {function_args.get('direction', '')}", 'image': output.get('image'), 'camera_position': output.get('camera_position')}
                else:
                    return {'text': f"Unknown operation: {op}", 'image': None, 'camera_position': None}
            
            elif function_name == "set_object_visibility":
                output = await self.tool_client.call_tool("set_object_visibility", {
                    "show_object_list": function_args.get("show_object_list", []),
                    "hide_object_list": function_args.get("hide_object_list", []),
                    "round_num": round_num
                })
                return {'text': 'Updated object visibility', 'image': output.get('image'), 'camera_position': output.get('camera_position')}
            elif function_name == "set_key_frame":
                output = await self.tool_client.call_tool("set_key_frame", {
                    "target_frame": function_args.get("target_frame", 0),
                    "round_num": round_num
                })
                return {'text': 'Jumped to key frame', 'image': output.get('image'), 'camera_position': output.get('camera_position')}
            elif function_name == "compare_designs":
                output = await self.tool_client.call_tool("compare_designs", {
                    "generated_path": current_image_path,
                    "target_path": target_image_path
                })
                if output.get("status") == "success":
                    result = output.get("result", {})
                    return {'text': result.get('comparison', ''), 'image': None, 'camera_position': None}
                else:
                    return {'text': f"Comparison failed: {output.get('error', 'Unknown error')}", 'image': None, 'camera_position': None}
            elif function_name == "analyze_html_structure":
                # Extract HTML code from the current context (this would need to be passed in)
                # For now, return a placeholder
                return {'text': "HTML structure analysis not yet implemented", 'image': None, 'camera_position': None}
            else:
                return {'text': f"Unknown tool: {function_name}", 'image': None, 'camera_position': None}
        except Exception as e:
            return {'text': f"Error executing tool: {str(e)}", 'image': None, 'camera_position': None}
