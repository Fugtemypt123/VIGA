import os
import json
from PIL import Image
import io
import base64
from typing import Dict, List, Optional
from openai import OpenAI
from prompts import prompts_dict

class PromptBuilder:
    """Helper class for building system prompts for generator and verifier agents."""
    
    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model
    
    def _get_image_base64(self, image_path: str) -> str:
        """Return a full data URL for the image, preserving original jpg/png format."""
        image = Image.open(image_path)
        img_byte_array = io.BytesIO()
        ext = os.path.splitext(image_path)[1].lower()
        
        # Convert image to appropriate mode for saving
        if ext in ['.jpg', '.jpeg']:
            save_format = 'JPEG'
            mime_subtype = 'jpeg'
            # JPEG doesn't support transparency, convert RGBA to RGB
            if image.mode in ['RGBA', 'LA', 'P']:
                # Convert P mode to RGB first, then handle RGBA
                if image.mode == 'P':
                    image = image.convert('RGBA')
                # Convert RGBA to RGB with white background
                if image.mode == 'RGBA':
                    # Create a white background
                    background = Image.new('RGB', image.size, (255, 255, 255))
                    background.paste(image, mask=image.split()[-1])  # Use alpha channel as mask
                    image = background
                elif image.mode == 'LA':
                    # Convert LA to RGB
                    image = image.convert('RGB')
        elif ext == '.png':
            save_format = 'PNG'
            mime_subtype = 'png'
            # PNG supports transparency, but convert P mode to RGBA
            if image.mode == 'P':
                image = image.convert('RGBA')
        else:
            # Fallback: keep original format if recognizable, else default to PNG
            save_format = image.format or 'PNG'
            mime_subtype = save_format.lower() if save_format.lower() in ['jpeg', 'png'] else 'png'
            # Handle P mode for fallback cases
            if image.mode == 'P':
                if save_format == 'JPEG':
                    image = image.convert('RGB')
                else:
                    image = image.convert('RGBA')
        
        image.save(img_byte_array, format=save_format)
        img_byte_array.seek(0)
        base64enc_image = base64.b64encode(img_byte_array.read()).decode('utf-8')
        return f"data:image/{mime_subtype};base64,{base64enc_image}"
    
    def build_generator_prompt(self, config: Dict) -> List[Dict]:
        """Generic method to build generator prompts based on mode and config."""
        mode = config.get("mode")
        
        if mode == "blendergym":
            return self.build_blendergym_generator_prompt(
                mode, 
                config.get("task_name"), 
                config.get("init_code_path"), 
                config.get("init_image_path"), 
                config.get("target_image_path")
            )
        elif mode == "autopresent":
            return self.build_autopresent_generator_prompt(
                mode, 
                config.get("init_code_path"), 
                config.get("init_image_path"), 
                config.get("target_description")
            )
        elif mode == "blendergym-hard":
            return self.build_blendergym_hard_generator_prompt(
                mode, 
                config.get("task_name"), 
                config.get("init_code_path"), 
                config.get("init_image_path"), 
                config.get("target_image_path"), 
                config.get("blender_file_path"), 
                config.get("target_description")
            )
        elif mode == "design2code":
            return self.build_design2code_generator_prompt(
                mode, 
                config.get("init_code_path"), 
                config.get("target_image_path")
            )
        else:
            raise NotImplementedError(f"Mode {mode} not implemented")
    
    def build_verifier_prompt(self, config: Dict) -> List[Dict]:
        """Generic method to build verifier prompts based on mode and config."""
        mode = config.get("mode")
        
        if mode == "blendergym":
            return self.build_blendergym_verifier_prompt(
                mode, 
                config.get("task_name"), 
                config.get("target_image_path")
            )
        elif mode == "autopresent":
            return self.build_autopresent_verifier_prompt(
                mode, 
                config.get("target_description")
            )
        elif mode == "blendergym-hard":
            return self.build_blendergym_hard_verifier_prompt(
                mode, 
                config.get("task_name"), 
                config.get("target_image_path"), 
                config.get("blender_file"), 
                config.get("target_description")
            )
        elif mode == "design2code":
            return self.build_design2code_verifier_prompt(
                mode, 
                config.get("target_image_path")
            )
        else:
            raise NotImplementedError(f"Mode {mode} not implemented")
    
    def build_verify_message(self, config: Dict, code: str, render_path: str, current_image_path_ref: List) -> Dict:
        """Generic method to build verify messages based on mode and config."""
        mode = config.get("mode")
        
        if mode == "blendergym":
            return self._build_blendergym_verify_message(code, render_path, current_image_path_ref)
        elif mode == "autopresent":
            return self._build_autopresent_verify_message(code, render_path)
        elif mode == "blendergym-hard":
            return self._build_blendergym_hard_verify_message(config, code, render_path, current_image_path_ref)
        elif mode == "design2code":
            return self._build_design2code_verify_message(code, render_path)
        else:
            raise NotImplementedError(f"Mode {mode} not implemented")
    
    def _build_blendergym_verify_message(self, code: str, render_path: str, current_image_path_ref: List) -> Dict:
        """Build verify message for blendergym mode."""
        verify_message = {"role": "user", "content": [{"type": "text", "text": f"Please analyze the current state:\nCode: {code}"}]}
        
        if os.path.isdir(render_path):
            view1_path = os.path.join(render_path, 'render1.png')
            view2_path = os.path.join(render_path, 'render2.png')
        else:
            view1_path = render_path
            view2_path = None
            
        scene_content = []
        if os.path.exists(view1_path):
            current_image_path_ref[0] = os.path.abspath(view1_path)
            scene_content.extend([
                {"type": "text", "text": f"Current scene (View 1):"},
                {"type": "image_url", "image_url": {"url": self._get_image_base64(view1_path)}}
            ])
        if os.path.exists(view2_path):
            scene_content.extend([
                {"type": "text", "text": f"Current scene (View 2):"},
                {"type": "image_url", "image_url": {"url": self._get_image_base64(view2_path)}}
            ])
            
        verify_message["content"].extend(scene_content)
        verify_message["content"].append({"type": "text", "text": prompts_dict["blendergym"]['format']['verifier']})
        
        return verify_message
    
    def _build_autopresent_verify_message(self, code: str, render_path: str) -> Dict:
        """Build verify message for autopresent mode."""
        verify_message = {"role": "user", "content": [{"type": "text", "text": f"Please analyze the current code and generated slide:\nCode: {code}"}]}
        
        # add slide screenshot
        if os.path.exists(render_path):
            verify_message["content"].append({"type": "text", "text": f"Generated slide screenshot:"})
            verify_message["content"].append({"type": "image_url", "image_url": {"url": self._get_image_base64(render_path)}})
            
        verify_message["content"].append({"type": "text", "text": prompts_dict["autopresent"]['format']['verifier']})
        
        return verify_message
    
    def _build_blendergym_hard_verify_message(self, config: Dict, code: str, render_path: str, current_image_path_ref: List) -> Dict:
        """Build verify message for blendergym-hard mode."""
        task_name = config.get("task_name")
        level = task_name.split('-')[0]
        verify_message = {"role": "user", "content": [{"type": "text", "text": f"Please analyze the current state:\n"}]}
        
        if os.path.isdir(render_path):
            view1_path = os.path.join(render_path, 'render1.png')
            view2_path = os.path.join(render_path, 'render2.png')
        else:
            view1_path = render_path
            view2_path = None
            
        scene_content = []
        if os.path.exists(view1_path):
            current_image_path_ref[0] = os.path.abspath(view1_path)
            scene_content.extend([
                {"type": "text", "text": f"Current scene (View 1):"},
                {"type": "image_url", "image_url": {"url": self._get_image_base64(view1_path)}}
            ])
        if os.path.exists(view2_path):
            scene_content.extend([
                {"type": "text", "text": f"Current scene (View 2):"},
                {"type": "image_url", "image_url": {"url": self._get_image_base64(view2_path)}}
            ])
            
        verify_message["content"].extend(scene_content)
        verify_message["content"].append({"type": "text", "text": prompts_dict["blendergym-hard"]['format']['verifier'][level]})
        
        return verify_message
    
    def _build_design2code_verify_message(self, code: str, render_path: str) -> Dict:
        """Build verify message for design2code mode."""
        verify_message = {"role": "user", "content": [{"type": "text", "text": f"Please analyze the generated HTML code and compare it with the target design:\nCode: {code}"}]}
        
        if os.path.exists(render_path):
            verify_message["content"].append({"type": "text", "text": f"Generated webpage screenshot:"})
            verify_message["content"].append({"type": "image_url", "image_url": {"url": self._get_image_base64(render_path)}})
            
        verify_message["content"].append({"type": "text", "text": prompts_dict["design2code"]['format']['verifier']})
        
        return verify_message
    
    def build_blendergym_hard_generator_prompt(self, 
                                             mode: str, 
                                             task_name: str, 
                                             init_code_path: str = None, 
                                             init_image_path: str = None, 
                                             target_image_path: str = None,
                                             blender_file_path: str = None,
                                             target_description: str = None) -> List[Dict]:
        """Build the system prompt for the generator for blendergym-hard mode."""
        level = task_name.split('-')[0]
        idx = int(task_name.split('-')[1])
        full_prompt = []
        
        # Add system prompt
        full_prompt.append({"role": "system", "content": prompts_dict[mode]['system']['generator'][level]})
        user_content = []
        
        # Add initial code
        init_code = open(init_code_path, 'r').read()
        user_content.append({"type": "text", "text": f"Initial Code:\n```python\n{init_code}\n```"})
        
        # Add scene information if blender file path is provided
        if target_description:
            scene_info = target_description
            user_content.append({"type": "text", "text": scene_info})
        
        # Add initial images
        init_image_path = os.path.join(init_image_path, 'render1.png')
        if os.path.exists(init_image_path):
            user_content.append({"type": "text", "text": "Initial Image (View 1):"})
            user_content.append({"type": "image_url", "image_url": {"url": self._get_image_base64(init_image_path)}})
        else:
            # At least we need one initial image
            raise ValueError(f"Initial image {init_image_path} does not exist!")
        # Add target images (for mode `blendergym`)
        if level == 'level1':
            target_image_path = os.path.join(target_image_path, 'style1.png')
        else:
            target_image_path = os.path.join(target_image_path, 'visprompt1.png')
        if os.path.exists(target_image_path):
            user_content.append({"type": "text", "text": "Target Image (View 1):"})
            user_content.append({"type": "image_url", "image_url": {"url": self._get_image_base64(target_image_path)}})
        else:
            raise ValueError(f"Target image {target_image_path} does not exist!") 
        
        # Add hints 
        user_content.append({"type": "text", "text": f"Hints:\n{prompts_dict[mode]['hints']['generator'][level]}"})
        # Add output format
        user_content.append({"type": "text", "text": prompts_dict[mode]['format']['generator'][level]})
        # Add all user content
        full_prompt.append({"role": "user", "content": user_content})
        return full_prompt
    
    def build_blendergym_generator_prompt(self, 
                                        mode: str, 
                                        task_name: str, 
                                        init_code_path: str = None, 
                                        init_image_path: str = None, 
                                        target_image_path: str = None) -> List[Dict]:
        """Build the system prompt for the generator for blendergym mode."""
        full_prompt = []
        # Add system prompt
        full_prompt.append({"role": "system", "content": prompts_dict[mode]['system']['generator']})
        
        # Add initial code & code analysis
        init_code = open(init_code_path, 'r').read()
        user_content = [{"type": "text", "text": f"Initial Code:\n```python\n{init_code}\n```"}]
        
        # Add code analysis
        code_analysis = self.client.chat.completions.create(
            model=self.model,  # Use a default model for code analysis
            messages=[
                {"role": "system", "content": "You are a Blender Python code analysis expert."},
                {"role": "user", "content": f"Please analyze the following Blender Python code line by line, \
                explaining what each part does and how it contributes to the scene:\n```python\n{init_code}\n```"}
            ]
        )
        code_analysis = code_analysis.choices[0].message.content
        user_content.append({"type": "text", "text": f"Code Analysis:\n{code_analysis}"})
        
        # Add initial images
        init_image_path_1 = os.path.join(init_image_path, 'render1.png')
        if os.path.exists(init_image_path_1):
            user_content.append({"type": "text", "text": "Initial Image (View 1):"})
            user_content.append({"type": "image_url", "image_url": {"url": self._get_image_base64(init_image_path_1)}})
        else:
            # At least we need one initial image
            raise ValueError(f"Initial image {init_image_path_1} does not exist!")
        
        init_image_path_2 = os.path.join(init_image_path, 'render2.png')
        if os.path.exists(init_image_path_2):
            user_content.append({"type": "text", "text": "Initial Image (View 2):"})
            user_content.append({"type": "image_url", "image_url": {"url": self._get_image_base64(init_image_path_2)}})
        
        # Add target images (for mode `blendergym`)
        target_image_path_1 = os.path.join(target_image_path, 'render1.png')
        if os.path.exists(target_image_path_1):
            user_content.append({"type": "text", "text": "Target Image (View 1):"})
            user_content.append({"type": "image_url", "image_url": {"url": self._get_image_base64(target_image_path_1)}})
        else:
            raise ValueError(f"Target image {target_image_path_1} does not exist!")
        
        target_image_path_2 = os.path.join(target_image_path, 'render2.png')
        if os.path.exists(target_image_path_2):
            user_content.append({"type": "text", "text": "Target Image (View 2):"})
            user_content.append({"type": "image_url", "image_url": {"url": self._get_image_base64(target_image_path_2)}})
        
        # Add hints 
        if prompts_dict[mode]['hints']['generator'][task_name] is not None:
            user_content.append({"type": "text", "text": f"Hints:\n{prompts_dict[mode]['hints']['generator'][task_name]}"})
        # Add output format
        user_content.append({"type": "text", "text": prompts_dict[mode]['format']['generator']})
        # Add all user content
        full_prompt.append({"role": "user", "content": user_content})
        return full_prompt
    
    def build_autopresent_generator_prompt(self, 
                                         mode: str, 
                                         init_code_path: str = None,
                                         init_image_path: str = None, 
                                         target_description: str = None) -> List[Dict]:
        """Build the system prompt for the generator for autopresent mode."""
        full_prompt = []
        # Add system prompt
        full_prompt.append({"role": "system", "content": prompts_dict[mode]['system']['generator'] + '\n\n' + prompts_dict[mode]['api_library']})
        
        # Add user input
        user_content = []
        user_content.append({"type": "text", "text": f"Now, here is the task package, which includes the initial code, a screenshot of the initial slides, the provided images with filenames used in the slides, and my instruction:"})
        
        # Add initial code
        init_code = open(init_code_path, 'r').read()
        user_content.append({"type": "text", "text": f"Initial Code:\n```python\n{init_code}\n```"})
        
        # Add initial images
        if os.path.exists(init_image_path):
            user_content.append({"type": "text", "text": "Initial Slide Screenshot:"})
            user_content.append({"type": "image_url", "image_url": {"url": self._get_image_base64(init_image_path)}})
        else:
            user_content.append({"type": "text", "text": "Initial code cannot be executed, please check the code and fix the errors."})
            
        # Add used images
        user_content.append({"type": "text", "text": "Provided Images (they might already appear in the code):"})
        used_image_dir = os.path.join(os.path.dirname(init_image_path), 'media')
        used_images = os.listdir(used_image_dir)
        for image in used_images:
            if image.endswith('.jpg') or image.endswith('.png') or image.endswith('.jpeg'):
                user_content.append({"type": "text", "text": f"Path: {os.path.join('media', image)}"})
                user_content.append({"type": "image_url", "image_url": {"url": self._get_image_base64(os.path.join(used_image_dir, image))}})
        
        # Add target description
        user_content.append({"type": "text", "text": f"Instruction:\n{target_description}"})
        # Add hints
        user_content.append({"type": "text", "text": f"Hints:\n{prompts_dict[mode]['hints']}"})
        # Add output format
        user_content.append({"type": "text", "text": prompts_dict[mode]['format']['generator']})
        # Add all user content
        full_prompt.append({"role": "user", "content": user_content})
        return full_prompt
    
    def build_blendergym_hard_verifier_prompt(self, 
                                            mode: str,
                                            task_name: str,
                                            target_image_path: str,
                                            blender_file_path: str,
                                            target_description: str) -> List[Dict]:
        """Build the system prompt for the verifier for blendergym-hard mode."""
        level = task_name.split('-')[0]
        idx = int(task_name.split('-')[1])
        full_prompt = []
        # System prompt
        full_prompt.append({"role": "system", "content": prompts_dict[mode]['system']['verifier'][level]})
        user_content = []
        # Add scene information if blender file path is provided
        if target_description:
            scene_info = target_description
            user_content.append({"type": "text", "text": scene_info})
        # Add target image/description
        if level == 'level1':
            target_image_path = os.path.join(target_image_path, 'style1.png')
        else:
            target_image_path = os.path.join(target_image_path, 'visprompt1.png')
        if os.path.exists(target_image_path):
            user_content.extend([
                {"type": "text", "text": "Target Image (View 1):"},
                {"type": "image_url", "image_url": {"url": self._get_image_base64(target_image_path)}}
            ])
        else:
            raise ValueError(f"Target image {target_image_path} does not exist!")
        user_content.append({"type": "text", "text": f"Hints:\n{prompts_dict[mode]['hints']['verifier'][level]}"})
        # Add tool example
        user_content.append({"type": "text", "text": f"Tool Example:\n{prompts_dict[mode]['tool_example']}"})
        full_prompt.append({"role": "user", "content": user_content})
        return full_prompt
    
    def build_blendergym_verifier_prompt(self, 
                                       mode: str,
                                       task_name: str,
                                       target_image_path: str) -> List[Dict]:
        """Build the system prompt for the verifier for blendergym mode."""
        full_prompt = []
        # System prompt
        full_prompt.append({"role": "system", "content": prompts_dict[mode]['system']['verifier']})
        user_content = []
        # Add target image/description
        target_image_path_1 = os.path.join(target_image_path, 'render1.png')
        if os.path.exists(target_image_path_1):
            user_content.extend([
                {"type": "text", "text": "Target Image (View 1):"},
                {"type": "image_url", "image_url": {"url": self._get_image_base64(target_image_path_1)}}
            ])
        else:
            raise ValueError(f"Target image {target_image_path_1} does not exist!")
        target_image_path_2 = os.path.join(target_image_path, 'render2.png')
        if os.path.exists(target_image_path_2):
            user_content.extend([
                {"type": "text", "text": "Target Image (View 2):"},
                {"type": "image_url", "image_url": {"url": self._get_image_base64(target_image_path_2)}}
            ])
        # Add hints
        if prompts_dict[mode]['hints']['verifier'][task_name] is not None:
            user_content.append({"type": "text", "text": f"Hints:\n{prompts_dict[mode]['hints']['verifier'][task_name]}"})            
        full_prompt.append({"role": "user", "content": user_content})
        return full_prompt
    
    def build_autopresent_verifier_prompt(self, 
                                        mode: str,
                                        target_description: str) -> List[Dict]:
        """Build the system prompt for the verifier for autopresent mode."""
        full_prompt = []
        # System prompt
        full_prompt.append({"role": "system", "content": prompts_dict[mode]['system']['verifier']})
        user_content = []
        
        # Add target description
        user_content.append({"type": "text", "text": f"Task Instruction:\n{target_description}"})
        
        # Add hint
        if prompts_dict[mode]['hints'] is not None:
            user_content.append({"type": "text", "text": f"Hints:\n{prompts_dict[mode]['hints']}"})
            
        full_prompt.append({"role": "user", "content": user_content})
        return full_prompt
    
    def build_design2code_generator_prompt(self, 
                                         mode: str,
                                         init_code_path: str,
                                         target_image_path: str) -> List[Dict]:
        """Build the system prompt for the generator for design2code mode."""
        full_prompt = []
        # Add system prompt
        full_prompt.append({"role": "system", "content": prompts_dict[mode]['system']['generator']})
        
        user_content = []
        
        # Add initial code
        init_code = open(init_code_path, 'r').read()
        user_content.append({"type": "text", "text": f"Initial Code:\n```python\n{init_code}\n```"})
        
        # Add design screenshot
        if os.path.exists(target_image_path):
            user_content.append({"type": "text", "text": "Design Screenshot:"})
            user_content.append({"type": "image_url", "image_url": {"url": self._get_image_base64(target_image_path)}})
        else:
            raise ValueError(f"Design image {target_image_path} does not exist!")
        
        # Add hints
        user_content.append({"type": "text", "text": f"Hints:\n{prompts_dict[mode]['hints']}"})
        
        # Add output format
        user_content.append({"type": "text", "text": prompts_dict[mode]['format']['generator']})
        
        # Add all user content
        full_prompt.append({"role": "user", "content": user_content})
        return full_prompt
    
    def build_design2code_verifier_prompt(self, 
                                        mode: str,
                                        target_image_path: str) -> List[Dict]:
        """Build the system prompt for the verifier for design2code mode."""
        full_prompt = []
        # System prompt
        full_prompt.append({"role": "system", "content": prompts_dict[mode]['system']['verifier']})
        user_content = []
        
        # Add target design image
        if os.path.exists(target_image_path):
            user_content.extend([
                {"type": "text", "text": "Target Design:"},
                {"type": "image_url", "image_url": {"url": self._get_image_base64(target_image_path)}}
            ])
        else:
            raise ValueError(f"Target design image {target_image_path} does not exist!")
        
        # Add hints
        user_content.append({"type": "text", "text": f"Hints:\n{prompts_dict[mode]['hints']}"})
        
        full_prompt.append({"role": "user", "content": user_content})
        return full_prompt