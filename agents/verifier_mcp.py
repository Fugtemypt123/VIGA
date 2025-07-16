import os
import json
import uuid
from PIL import Image, ImageChops, ImageEnhance
import io
import base64
import numpy as np
import math
import tempfile
import sys
import traceback
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from openai import OpenAI
from mcp.server.fastmcp import FastMCP

@dataclass
class VerificationSession:
    """Represents a verification session with all its state."""
    session_id: str
    vision_model: str
    api_key: str
    thoughtprocess_save: str
    max_rounds: int
    verifier_hints: Optional[str]
    target_image_path: Optional[str]
    blender_save: Optional[str]
    memory: List[Dict]
    current_round: int
    created_at: str
    last_updated: str
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'VerificationSession':
        return cls(**data)

class PILExecutor:
    """PIL code execution tool for image processing."""
    
    def __init__(self):
        self._setup_environment()

    def _setup_environment(self):
        self.globals = {
            'Image': Image,
            'io': io,
            'base64': base64,
            'current_image': None,
            'result': None
        }

    def _image_to_base64(self, image: Image.Image) -> str:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    def execute(self, code: str) -> Dict:
        stdout_capture = StringIO()
        stderr_capture = StringIO()
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = stdout_capture, stderr_capture

        try:
            exec(code, self.globals)
            result = self.globals.get('result', None)
            if isinstance(result, Image.Image):
                result = self._image_to_base64(result)
            return {
                'success': True,
                'result': result,
                'stdout': stdout_capture.getvalue(),
                'stderr': stderr_capture.getvalue()
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'stdout': stdout_capture.getvalue(),
                'stderr': stderr_capture.getvalue() + traceback.format_exc()
            }
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr

class ImageDifferentiationTool:
    """Tool for comparing and analyzing image differences."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = OpenAI(api_key=self.api_key)

    def pil_to_base64(self, image: Image.Image) -> str:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _highlight_differences(self, img1, img2, diff, threshold=50):
        img1_array = np.array(img1)
        img2_array = np.array(img2)
        diff_array = np.array(diff)
        mask = np.any(diff_array > threshold, axis=2)

        highlight = np.array([255, 0, 0])
        img1_high = img1_array.copy()
        img2_high = img2_array.copy()

        img1_high[mask] = ((img1_high[mask] * 0.5 + highlight * 0.5)).astype(np.uint8)
        img2_high[mask] = ((img2_high[mask] * 0.5 + highlight * 0.5)).astype(np.uint8)

        return Image.fromarray(img1_high), Image.fromarray(img2_high)

    def describe_difference(self, path1: str, path2: str) -> str:
        img1 = Image.open(path1).convert("RGB")
        img2 = Image.open(path2).convert("RGB")
        if img1.size != img2.size:
            img2 = img2.resize(img1.size)

        diff = ImageChops.difference(img1, img2)
        img1_high, img2_high = self._highlight_differences(img1, img2, diff)

        enhancer = ImageEnhance.Brightness(diff)
        diff_bright = enhancer.enhance(4.0)

        b64s = [self.pil_to_base64(im) for im in [img1, img2, img1_high, img2_high]]

        messages = [
            {"role": "system", "content": "You are an expert in image comparison."},
            {"role": "user", "content": [
                {"type": "text", "text": "Compare the two original images and describe the highlighted red difference."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64s[0]}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64s[1]}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64s[2]}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64s[3]}"}},
            ]}
        ]

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=512
        )
        return response.choices[0].message.content

class Investigator3D:
    """3D scene investigation tool for Blender scenes."""
    
    def __init__(self, thoughtprocess_save: str, blender_path: str, round_num: int):
        # Note: This would need to be run in a Blender environment
        # For now, we'll create a mock implementation
        self.base = Path(thoughtprocess_save) / f"investigator_{round_num}"
        self.base.mkdir(parents=True, exist_ok=True)
        self.blender_path = blender_path
        self.round_num = round_num
        self.count = 0

    def focus_on_object(self, object_name: str) -> str:
        """Focus camera on a specific object."""
        # Mock implementation - in real usage this would use Blender Python API
        output_path = str(self.base / f"{self.count:03d}_focus_{object_name}.png")
        self.count += 1
        return output_path

    def zoom(self, direction: str) -> str:
        """Zoom camera in or out."""
        output_path = str(self.base / f"{self.count:03d}_zoom_{direction}.png")
        self.count += 1
        return output_path

    def move_camera(self, direction: str) -> str:
        """Move camera in specified direction."""
        output_path = str(self.base / f"{self.count:03d}_move_{direction}.png")
        self.count += 1
        return output_path

class MCPVerifierAgent:
    """
    An MCP agent that verifies visual scenes and provides feedback.
    This agent follows the MCP server pattern with session management and tool integration.
    """
    
    def __init__(self):
        """Initialize the MCP Verifier Agent."""
        self.sessions: Dict[str, VerificationSession] = {}
        self.logger = logging.getLogger(__name__)
    
    def create_session(self, 
                      vision_model: str,
                      api_key: str,
                      thoughtprocess_save: str,
                      max_rounds: int = 10,
                      verifier_hints: Optional[str] = None,
                      target_image_path: Optional[str] = None,
                      blender_save: Optional[str] = None) -> str:
        """
        Create a new verification session.
        
        Returns:
            str: Session ID
        """
        session_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        
        # Initialize memory if target images are provided
        memory = []
        if target_image_path:
            memory = self._build_system_prompt(
                vision_model, api_key, verifier_hints, target_image_path
            )
        
        session = VerificationSession(
            session_id=session_id,
            vision_model=vision_model,
            api_key=api_key,
            thoughtprocess_save=thoughtprocess_save,
            max_rounds=max_rounds,
            verifier_hints=verifier_hints,
            target_image_path=target_image_path,
            blender_save=blender_save,
            memory=memory,
            current_round=0,
            created_at=now,
            last_updated=now
        )
        
        self.sessions[session_id] = session
        self.logger.info(f"Created new verification session: {session_id}")
        return session_id
    
    def _build_system_prompt(self, vision_model: str, api_key: str, hints: str, 
                           target_image_path: str) -> List[Dict]:
        """Build the system prompt for the verifier."""
        full_prompt = []
        
        # Add system prompt
        full_prompt.append({
            "role": "system",
            "content": """You're a 3D visual feedback assistant tasked with providing revision suggestions to a 3D scene designer. At the beginning, you will be given several images that describe the target 3D scene. In each subsequent interaction, you will receive a few images of the current 3D scene along with the code that generated it.
Your responsibilities include:
1. **Visual Difference Identification**: Identify differences between the target scene and the current scene. You may use visual tools to assist in this process and should pay close attention to detail. Only answer the most obvious 1-2 differences at a time, don't answer too many.
2. **Code Localization**: Pinpoint locations in the code that could be modified to reduce or eliminate these differences. This may require counterfactual reasoning and inference from the visual discrepancies."""
        })
        
        user_content = []
        
        # Add target images
        target_image_path_1 = os.path.join(target_image_path, 'render1.png')
        if os.path.exists(target_image_path_1):
            user_content.extend([
                {
                    "type": "text",
                    "text": f"Target Image (View 1): {target_image_path_1}"
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{self._get_image_base64(target_image_path_1)}"}
                }
            ])
        
        target_image_path_2 = os.path.join(target_image_path, 'render2.png')
        if os.path.exists(target_image_path_2):
            user_content.extend([
                {
                    "type": "text",
                    "text": f"Target Image (View 2): {target_image_path_2}"
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{self._get_image_base64(target_image_path_2)}"}
                }
            ])
        
        # Add hints
        if hints is not None:
            user_content.append({
                "type": "text",
                "text": f"Hints:\n{hints}"
            })
        
        # Add output format
        output_format = """Output Structure:
1. Thought: Analyze the current state and provide a clear plan for the required changes.
2. Visual Difference: Describe the main differences found (between the current and target scene). Only answer the most obvious 1-2 differences at a time, don't answer too many.
3. Code Localization: Pinpoint locations in the code that could be modified to reduce or eliminate these most obvious differences.
If the current scene is very close to the target scene, only output an "OK!" and do not output other characters."""
        
        user_content.append({
            "type": "text",
            "text": output_format
        })
        
        full_prompt.append({
            "role": "user",
            "content": user_content
        })
        return full_prompt
    
    def _get_image_base64(self, image_path: str) -> str:
        """Convert image to base64 string."""
        try:
            image = Image.open(image_path)
            img_byte_array = io.BytesIO()
            image.save(img_byte_array, format='PNG')
            img_byte_array.seek(0) 
            base64enc_image = base64.b64encode(img_byte_array.read()).decode('utf-8') 
            return base64enc_image
        except Exception as e:
            self.logger.error(f"Failed to convert image to base64: {e}")
            return ""
    
    def verify_scene(self, session_id: str, code: str, render_path: str, round_num: int) -> Dict[str, Any]:
        """
        Verify a scene against the target.
        
        Args:
            session_id: The session ID
            code: The code that generated the scene
            render_path: Path to the rendered images
            round_num: Current round number
            
        Returns:
            Dict containing verification result
        """
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.sessions[session_id]
        
        if session.current_round >= session.max_rounds:
            return {
                "status": "max_rounds_reached",
                "message": f"Maximum rounds ({session.max_rounds}) reached",
                "round": session.current_round
            }
        
        # Get paths for both views
        view1_path = os.path.join(render_path, 'render1.png')
        view2_path = os.path.join(render_path, 'render2.png')
        
        # Prepare verification message
        verify_message = {
            "role": "user",
            "content": [{"type": "text", "text": f"Please analyze the current state:\nCode: {code}"}]
        }
        
        scene_content = [
            {"type": "text", "text": f"Current scene (View 1): {view1_path}"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{self._get_image_base64(view1_path)}"}}
        ]
        
        # Add view2 if available
        if os.path.exists(view2_path):
            scene_content.extend([
                {"type": "text", "text": f"Current scene (View 2): {view2_path}"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{self._get_image_base64(view2_path)}"}}
            ])
        
        verify_message["content"].extend(scene_content)
        
        # Add output format
        output_format = """Output Structure:
1. Thought: Analyze the current state and provide a clear plan for the required changes.
2. Visual Difference: Describe the main differences found (between the current and target scene). Only answer the most obvious 1-2 differences at a time, don't answer too many.
3. Code Localization: Pinpoint locations in the code that could be modified to reduce or eliminate these most obvious differences.
If the current scene is very close to the target scene, only output an "OK!" and do not output other characters."""
        
        verify_message["content"].append({"type": "text", "text": output_format})
        
        # Add image differentiation if available
        if session.target_image_path:
            diff_content = "Below I will provide you with some visual differences identified by some tools. These tools are more capable than you are, use their results to improve your visual difference identification. When the tool says 'No difference', it means you can select OK."
            
            try:
                image_diff_tool = ImageDifferentiationTool(session.api_key)
                target_image_path_1 = os.path.join(session.target_image_path, 'render1.png')
                
                if os.path.exists(target_image_path_1):
                    view1_diff = image_diff_tool.describe_difference(view1_path, target_image_path_1)
                    diff_content += f"\nView1: The differences between the current scene (first image) and the target scene (second image) are as follows: {view1_diff}"
                
                target_image_path_2 = os.path.join(session.target_image_path, 'render2.png')
                if os.path.exists(view2_path) and os.path.exists(target_image_path_2):
                    view2_diff = image_diff_tool.describe_difference(view2_path, target_image_path_2)
                    diff_content += f"\nView2: The differences between the current scene (first image) and the target scene (second image) are as follows: {view2_diff}"
                
                verify_message["content"].append({"type": "text", "text": diff_content})
            except Exception as e:
                self.logger.warning(f"Image differentiation failed: {e}")
        
        session.memory.append(verify_message)
        
        # Start verification loop
        try:
            client = OpenAI(api_key=session.api_key)
            
            for i in range(session.max_rounds):
                # Get analysis from model
                response = client.chat.completions.create(
                    model=session.vision_model,
                    messages=session.memory,
                    tools=self._get_tools(),
                    tool_choice="auto"
                )
                
                message = response.choices[0].message
                session.memory.append(message.model_dump())
                
                # Handle tool calls and get analysis
                if message.tool_calls:
                    for tool_call in message.tool_calls:
                        tool_response = self._handle_tool_call(tool_call, session)
                        session.memory.append({
                            "role": "tool", 
                            "tool_call_id": tool_call.id, 
                            "name": tool_call.function.name, 
                            "content": tool_response['text']
                        })
                        
                        if tool_response['image']:
                            session.memory.append({
                                "role": "user", 
                                "content": [
                                    {"type": "text", "text": "Generated image:"}, 
                                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{self._get_image_base64(tool_response['image'])}"}}
                                ]
                            })
                else:
                    # Check if the scene matches the target
                    if "OK" in message.content and "Code Localization" not in message.content:
                        result = {"status": "end", "output": message.content}
                    else:
                        result = {"status": "continue", "output": message.content}
                    break
                
                # Save thought process after each round
                self.save_thought_process(session_id)
            
            session.current_round += 1
            session.last_updated = datetime.now().isoformat()
            
            return result
            
        except Exception as e:
            self.logger.error(f"Verification failed for session {session_id}: {e}")
            return {
                "status": "error",
                "error": str(e),
                "round": session.current_round
            }
    
    def _get_tools(self) -> List[Dict]:
        """Get the tools available for the verifier."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "investigate_3d",
                    "description": """A tool for detailed 3D scene investigation with the following operations:
   - Focus on an object: Set the camera to track a specific object
   - Zoom in/out: Adjust the camera distance from the object (each adjustment changes radius by 1 unit)
   - Move camera: Move the camera on a sphere around the object (each movement covers 1/4 pi r^2 area)
   The camera moves on a sphere around the target object, where zooming changes the sphere's radius.
Focus the tool on the object whose shape or position needs to be adjusted, and adjust one (or more) appropriate angles to observe their specific changes.""",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "operation": {
                                "type": "string",
                                "enum": ["focus", "zoom", "move"],
                                "description": "The operation to perform: focus on object (set the camera to track a specific object), zoom in/out (adjust the camera distance from the object (each adjustment changes radius by 1 unit)), or move camera (move the camera on a sphere around the object (each movement covers 1/4 pi r^2 area))"
                            },
                            "object_name": {
                                "type": "string",
                                "description": "Name of the object to focus on (required for focus operation)"
                            },
                            "direction": {
                                "type": "string",
                                "enum": ["in", "out", "up", "down", "left", "right"],
                                "description": "Direction for zoom (in/out) or camera movement (up/down/left/right)"
                            }
                        },
                        "required": ["operation"]
                    }
                }
            }
        ]
    
    def _handle_tool_call(self, tool_call, session) -> Dict[str, Any]:
        """Handle tool calls from the model."""
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)
        
        try:
            if function_name == "investigate_3d":
                # Initialize the 3D investigator
                investigator = Investigator3D(
                    session.thoughtprocess_save, 
                    session.blender_save or "", 
                    session.current_round
                )
                
                # Handle different operations
                if function_args['operation'] == 'focus':
                    if 'object_name' not in function_args:
                        raise ValueError("object_name is required for focus operation")
                    output = investigator.focus_on_object(object_name=function_args['object_name'])
                    if output:
                        return {'text': f"Focused camera on object: {function_args['object_name']}", 'image': output}
                    else:
                        return {'text': f"Failed to focus camera on object: {function_args['object_name']}", 'image': None}
                
                elif function_args['operation'] == 'zoom':
                    if 'direction' not in function_args:
                        raise ValueError("direction is required for zoom operation")
                    output = investigator.zoom(direction=function_args['direction'])
                    if output:
                        return {'text': f"Zoomed {function_args['direction']}", 'image': output}
                    else:
                        return {'text': f"Failed to zoom {function_args['direction']}", 'image': None}
                
                elif function_args['operation'] == 'move':
                    if 'direction' not in function_args:
                        raise ValueError("direction is required for move operation")
                    output = investigator.move_camera(direction=function_args['direction'])
                    if output:
                        return {'text': f"Moved camera {function_args['direction']}", 'image': output}
                    else:
                        return {'text': f"Failed to move camera {function_args['direction']}", 'image': None}
                
                else:
                    raise ValueError(f"Unknown operation: {function_args['operation']}")
            else:
                raise ValueError(f"Unknown tool: {function_name}")
                
        except Exception as e:
            return {'text': f"Error executing tool: {str(e)}", 'image': None}
    
    def save_thought_process(self, session_id: str) -> None:
        """Save the thought process for a session."""
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.sessions[session_id]
        try:
            with open(session.thoughtprocess_save, "w") as f:
                json.dump(session.memory, f, indent=4, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Failed to save thought process for session {session_id}: {e}")
    
    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """Get information about a session."""
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.sessions[session_id]
        return {
            "session_id": session.session_id,
            "current_round": session.current_round,
            "max_rounds": session.max_rounds,
            "created_at": session.created_at,
            "last_updated": session.last_updated,
            "memory_length": len(session.memory)
        }
    
    def get_memory(self, session_id: str) -> List[Dict]:
        """Get the memory for a session."""
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        
        return self.sessions[session_id].memory
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all active sessions."""
        return [self.get_session_info(session_id) for session_id in self.sessions.keys()]
    
    def delete_session(self, session_id: str) -> None:
        """Delete a session."""
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        
        del self.sessions[session_id]
        self.logger.info(f"Deleted session: {session_id}")
    
    def reset_session_memory(self, session_id: str) -> None:
        """Reset the memory for a session."""
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.sessions[session_id]
        session.memory = []
        session.current_round = 0
        session.last_updated = datetime.now().isoformat()


def main():
    """Main function to run the MCP Verifier Agent as an MCP server."""
    mcp = FastMCP("verifier")
    agent = MCPVerifierAgent()

    @mcp.tool()
    def create_verification_session(
        vision_model: str,
        api_key: str,
        thoughtprocess_save: str,
        max_rounds: int = 10,
        verifier_hints: str = None,
        target_image_path: str = None,
        blender_save: str = None
    ) -> dict:
        """
        Create a new verification session.
        """
        try:
            session_id = agent.create_session(
                vision_model=vision_model,
                api_key=api_key,
                thoughtprocess_save=thoughtprocess_save,
                max_rounds=max_rounds,
                verifier_hints=verifier_hints,
                target_image_path=target_image_path,
                blender_save=blender_save
            )
            return {
                "status": "success",
                "session_id": session_id,
                "message": "Verification session created successfully"
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    @mcp.tool()
    def verify_scene(session_id: str, code: str, render_path: str, round_num: int) -> dict:
        """
        Verify a scene against the target.
        """
        try:
            result = agent.verify_scene(session_id, code, render_path, round_num)
            return result
        except Exception as e:
            return {"status": "error", "error": str(e)}

    @mcp.tool()
    def exec_pil_code(code: str) -> dict:
        """
        Execute PIL code for image processing.
        """
        try:
            tool = PILExecutor()
            result = tool.execute(code)
            return result
        except Exception as e:
            return {"status": "error", "error": str(e)}

    @mcp.tool()
    def compare_images(path1: str, path2: str, api_key: str) -> dict:
        """
        Compare two images and describe differences.
        """
        try:
            tool = ImageDifferentiationTool(api_key)
            result = tool.describe_difference(path1, path2)
            return {"description": result}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    @mcp.tool()
    def save_thought_process(session_id: str) -> dict:
        """
        Save the thought process for a session.
        """
        try:
            agent.save_thought_process(session_id)
            return {
                "status": "success",
                "message": "Thought process saved successfully"
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    @mcp.tool()
    def get_session_info(session_id: str) -> dict:
        """
        Get information about a session.
        """
        try:
            info = agent.get_session_info(session_id)
            return info
        except Exception as e:
            return {"status": "error", "error": str(e)}

    @mcp.tool()
    def get_memory(session_id: str) -> dict:
        """
        Get the memory for a session.
        """
        try:
            memory = agent.get_memory(session_id)
            return {"memory": memory}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    @mcp.tool()
    def list_sessions() -> dict:
        """
        List all active sessions.
        """
        try:
            sessions = agent.list_sessions()
            return {"sessions": sessions}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    @mcp.tool()
    def delete_session(session_id: str) -> dict:
        """
        Delete a session.
        """
        try:
            agent.delete_session(session_id)
            return {
                "status": "success",
                "message": "Session deleted successfully"
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    @mcp.tool()
    def reset_session_memory(session_id: str) -> dict:
        """
        Reset the memory for a session.
        """
        try:
            agent.reset_session_memory(session_id)
            return {
                "status": "success",
                "message": "Session memory reset successfully"
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    mcp.run(transport="stdio")


if __name__ == "__main__":
    main() 