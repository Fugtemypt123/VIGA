import os
import json
from PIL import Image
import io
import base64
from openai import OpenAI
from typing import Dict, List, Optional, Any
import logging
from mcp.server.fastmcp import FastMCP

class GeneratorAgent:
    """
    An MCP agent that takes code modification suggestions and implements them.
    This agent follows the MCP server pattern for better encapsulation and tool integration.
    """
    
    def __init__(self, 
                 vision_model: str,
                 api_key: str,
                 thoughtprocess_save: str,
                 max_rounds: int = 10,
                 generator_hints: Optional[str] = None,
                 init_code: Optional[str] = None,
                 init_image_path: Optional[str] = None,
                 target_image_path: Optional[str] = None,
                 target_description: Optional[str] = None):
        """
        Initialize the Generator Agent.
        
        Args:
            vision_model: The OpenAI vision model to use
            api_key: OpenAI API key
            thoughtprocess_save: Path to save thought process
            max_rounds: Maximum number of generation rounds
            generator_hints: Hints for code generation
            init_code: Initial code to modify
            init_image_path: Path to initial images
            target_image_path: Path to target images
            target_description: Description of target
        """
        self.model = vision_model
        self.api_key = api_key
        self.client = OpenAI(api_key=self.api_key)
        self.thoughtprocess_save = thoughtprocess_save
        self.max_rounds = max_rounds
        self.memory = []
        
        # Initialize memory if initial parameters are provided
        if all([init_code, init_image_path, target_image_path]):
            self.memory = self._build_system_prompt(
                generator_hints, init_code, init_image_path, 
                target_image_path, target_description
            )
    
    def _build_system_prompt(self, hints: str, init_code: str, init_image_path: str, 
                           target_image_path: str, target_description: str = None) -> List[Dict]:
        """
        Build the system prompt for the generator.
        """
        full_prompt = []
        # Add system prompt
        full_prompt.append({
            "role": "system",
            "content": """You are a code generation agent proficient in Blender Python scripting. Your task is to edit code to transform an initial 3D scene into a target scene. After each code edit, your code will be passed to a validator, which will provide feedback on the result. Based on this feedback, you must iteratively refine your code edits. This process will continue across multiple rounds of dialogue. In each round, you must adhere to a fixed output format."""
        })
        
        # Add initial code & code analysis
        user_content = [{
            "type": "text",
            "text": f"Initial Code:\n```python\n{init_code}\n```"
        }]
        
        code_analysis = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a Blender Python code analysis expert."},
                {"role": "user", "content": f"Please analyze the following Blender Python code line by line, explaining what each part does and how it contributes to the scene:\n```python\n{init_code}\n```"}
            ]
        )
        code_analysis = code_analysis.choices[0].message.content
        user_content.append({
            "type": "text",
            "text": f"Code Analysis:\n{code_analysis}"
        })
        
        # Add initial images
        init_image_path_1 = os.path.join(init_image_path, 'render1.png')
        if os.path.exists(init_image_path_1):
            user_content.append({
                "type": "text",
                "text": "Initial Image (View 1):"
            })
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{self._get_image_base64(init_image_path_1)}"}
            })
        
        init_image_path_2 = os.path.join(init_image_path, 'render2.png')
        if os.path.exists(init_image_path_2):
            user_content.append({
                "type": "text",
                "text": "Initial Image (View 2):"
            })
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{self._get_image_base64(init_image_path_2)}"}
            })
        
        # Add target images
        target_image_path_1 = os.path.join(target_image_path, 'render1.png')
        if os.path.exists(target_image_path_1):
            user_content.append({
                "type": "text",
                "text": "Target Image (View 1):"
            })
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{self._get_image_base64(target_image_path_1)}"}
            })
        
        target_image_path_2 = os.path.join(target_image_path, 'render2.png')
        if os.path.exists(target_image_path_2):
            user_content.append({
                "type": "text",
                "text": "Target Image (View 2):"
            })
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{self._get_image_base64(target_image_path_2)}"}
            })
        
        # Add hints 
        if hints is not None:
            user_content.append({
                "type": "text",
                "text": f"Hints:\n{hints}"
            })
        
        # Add output format
        user_content.append({
            "type": "text",
            "text": """After each code edit, your code will be passed to a validator, which will provide feedback on the result. Based on this feedback, you must iteratively refine your code edits. This process will continue across multiple rounds of dialogue. In each round, you must follow a fixed output format. Output Format (keep this format for each round):
1. Thought: Analyze the current state and provide a clear plan for the required changes.
2. Code Edition: Provide your code modifications in the following format:
   -: [lines to remove]
   +: [lines to add]
3. Full Code: Merge your code changes into the full code:
```python
[full code]
```"""
        })
        
        full_prompt.append({
            "role": "user",
            "content": user_content
        })
        return full_prompt
    
    def _get_image_base64(self, image_path: str) -> str:
        """Convert image to base64 string."""
        image = Image.open(image_path)
        img_byte_array = io.BytesIO()
        image.save(img_byte_array, format='PNG')
        img_byte_array.seek(0) 
        base64enc_image = base64.b64encode(img_byte_array.read()).decode('utf-8') 
        return base64enc_image
    
    def _parse_generate_response(self, response: str) -> tuple:
        """
        Parse the generate response.
        Returns: (thought, edit, full_code)
        """
        try:
            full = response.split("Full Code")[1].strip()
        except:
            full = response.strip()
        
        # Remove the ```python and ``` from the full code
        if "```python" in full:
            full = full.split("```python")[1].split("```")[0].strip()
        else:
            full = full.split("```")[0].strip()
        
        return None, None, full
    
    def generate_code(self, feedback: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate code based on current memory and optional feedback.
        
        Args:
            feedback: Optional feedback from verifier or executor
            
        Returns:
            Dict containing the generated code and metadata
        """
        if feedback:
            self.memory.append({"role": "user", "content": feedback})
        
        try:
            generate_response = self.client.chat.completions.create(
                model=self.model, 
                messages=self.memory
            ).choices[0].message.content
            
            _, _, full_code = self._parse_generate_response(generate_response)
            self.memory.append({"role": "assistant", "content": generate_response})
            
            return {
                "status": "success",
                "code": full_code,
                "response": generate_response,
                "round": len([msg for msg in self.memory if msg["role"] == "assistant"])
            }
        except Exception as e:
            logging.error(f"Code generation failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "round": len([msg for msg in self.memory if msg["role"] == "assistant"])
            }
    
    def add_feedback(self, feedback: str) -> None:
        """
        Add feedback to the agent's memory.
        
        Args:
            feedback: Feedback from verifier or executor
        """
        self.memory.append({"role": "user", "content": feedback})
    
    def save_thought_process(self) -> None:
        """Save the current thought process to file."""
        try:
            with open(self.thoughtprocess_save, "w") as f:
                json.dump(self.memory, f, indent=4, ensure_ascii=False)
        except Exception as e:
            logging.error(f"Failed to save thought process: {e}")
    
    def get_memory(self) -> List[Dict]:
        """Get the current memory/conversation history."""
        return self.memory
    
    def reset_memory(self) -> None:
        """Reset the agent's memory."""
        self.memory = []


def main():
    """Main function to run the Generator Agent as an MCP server."""
    mcp = FastMCP("generator")
    
    agent_holder = {}

    @mcp.tool()
    def initialize_generator(
        vision_model: str,
        api_key: str,
        thoughtprocess_save: str,
        max_rounds: int = 10,
        generator_hints: str = None,
        init_code: str = None,
        init_image_path: str = None,
        target_image_path: str = None,
        target_description: str = None
    ) -> dict:
        """
        Initialize a new Generator Agent.
        """
        try:
            agent = GeneratorAgent(
                vision_model=vision_model,
                api_key=api_key,
                thoughtprocess_save=thoughtprocess_save,
                max_rounds=max_rounds,
                generator_hints=generator_hints,
                init_code=init_code,
                init_image_path=init_image_path,
                target_image_path=target_image_path,
                target_description=target_description
            )
            agent_holder['agent'] = agent
            return {
                "status": "success",
                "message": "Generator Agent initialized successfully"
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    @mcp.tool()
    def generate_code(feedback: str = None) -> dict:
        """
        Generate code using the initialized Generator Agent.
        """
        try:
            if 'agent' not in agent_holder:
                return {"status": "error", "error": "Generator Agent not initialized. Call initialize_generator first."}
            result = agent_holder['agent'].generate_code(feedback)
            return result
        except Exception as e:
            return {"status": "error", "error": str(e)}

    @mcp.tool()
    def add_feedback(feedback: str) -> dict:
        """
        Add feedback to the Generator Agent's memory.
        """
        try:
            if 'agent' not in agent_holder:
                return {"status": "error", "error": "Generator Agent not initialized. Call initialize_generator first."}
            agent_holder['agent'].add_feedback(feedback)
            return {"status": "success", "message": "Feedback added successfully"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    @mcp.tool()
    def save_thought_process() -> dict:
        """
        Save the current thought process to file.
        """
        try:
            if 'agent' not in agent_holder:
                return {"status": "error", "error": "Generator Agent not initialized. Call initialize_generator first."}
            agent_holder['agent'].save_thought_process()
            return {"status": "success", "message": "Thought process saved successfully"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    @mcp.tool()
    def get_memory() -> dict:
        """
        Get the current memory/conversation history.
        """
        try:
            if 'agent' not in agent_holder:
                return {"status": "error", "error": "Generator Agent not initialized. Call initialize_generator first."}
            memory = agent_holder['agent'].get_memory()
            return {"memory": memory}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    @mcp.tool()
    def reset_memory() -> dict:
        """
        Reset the agent's memory.
        """
        try:
            if 'agent' not in agent_holder:
                return {"status": "error", "error": "Generator Agent not initialized. Call initialize_generator first."}
            agent_holder['agent'].reset_memory()
            return {"status": "success", "message": "Memory reset successfully"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    mcp.run(transport="stdio")


if __name__ == "__main__":
    main() 