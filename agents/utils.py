import os
import json
from PIL import Image
import io
import base64
from typing import Dict, List, Optional

def get_image_base64(image_path: str) -> str:
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

def parse_generate_response(response: str) -> tuple:
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
    elif "```" in full:
        full = full.split("```")[0].strip()
    else:
        full = None
    
    return None, None, full

def get_blendergym_hard_level(task_name: str) -> str:
    """Extract the level from blendergym-hard task name."""
    # Task name format is expected to be like "task-level1", "task-level2", etc.
    if "level1" in task_name.lower():
        return "level1"
    elif "level2" in task_name.lower():
        return "level2"
    elif "level3" in task_name.lower():
        return "level3"
    elif "level4" in task_name.lower():
        return "level4"
    else:
        # Default to level1 if no level is specified
        return "level1"

def save_thought_process(memory: List[Dict], thought_save: str, current_round: int = None) -> None:
    """Save the current thought process to file."""
    try:
        if current_round is not None:
            filename = f"{thought_save}/{current_round+1}.json"
        else:
            filename = thought_save
        
        with open(filename, "w") as f:
            json.dump(memory, f, indent=4, ensure_ascii=False)
    except Exception as e:
        import logging
        logging.error(f"Failed to save thought process: {e}")
