import os
import json
from PIL import Image
import io
import base64
from typing import Dict, List, Optional
from openai import OpenAI
from _api_keys import OPENAI_API_KEY, OPENAI_BASE_URL, CLAUDE_API_KEY, CLAUDE_BASE_URL, GEMINI_API_KEY, GEMINI_BASE_URL, QWEN_BASE_URL, MESHY_API_KEY, VA_API_KEY

def build_client(model_name: str):
    model_name = model_name.lower()
    if "gpt" in model_name:
        return OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    elif "claude" in model_name:
        return OpenAI(api_key=CLAUDE_API_KEY, base_url=CLAUDE_BASE_URL)
    elif "gemini" in model_name:
        return OpenAI(api_key=GEMINI_API_KEY, base_url=GEMINI_BASE_URL)
    elif "qwen" in model_name:
        return OpenAI(api_key='not_used', base_url=QWEN_BASE_URL)
    else:
        raise ValueError(f"Invalid model name: {model_name}")
    
def get_model_info(model_name: str):
    model_name = model_name.lower()
    if "gpt" in model_name:
        return {"api_key": OPENAI_API_KEY, "base_url": OPENAI_BASE_URL}
    elif "claude" in model_name:
        return {"api_key": CLAUDE_API_KEY, "base_url": CLAUDE_BASE_URL}
    elif "gemini" in model_name:
        return {"api_key": GEMINI_API_KEY, "base_url": GEMINI_BASE_URL}
    elif "qwen" in model_name:
        return {"api_key": 'not_used', "base_url": QWEN_BASE_URL}
    else:
        raise ValueError(f"Invalid model name: {model_name}")
    
def get_meshy_info():
    return {"meshy_api_key": MESHY_API_KEY, "va_api_key": VA_API_KEY}

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
    if base64enc_image.startswith("/9j/"):
        mime_subtype = 'jpeg'
    elif base64enc_image.startswith("iVBOR"):
        mime_subtype = 'png'
    elif base64enc_image.startswith("UklGR"):
        mime_subtype = 'webp'
    return f"data:image/{mime_subtype};base64,{base64enc_image}"


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
        
def extract_code_pieces(text: str, concat: bool = True) -> list[str]:
    """Extract code pieces from a text string.
    Args:
        text: str, model prediciton text.
    Rets:
        code_pieces: list[str], code pieces in the text.
    """
    code_pieces = []
    while "```python" in text:
        st_idx = text.index("```python") + 10
        # end_idx = text.index("```", st_idx)
        if "```" in text[st_idx:]:
            end_idx = text.index("```", st_idx)
        else: 
            end_idx = len(text)
        code_pieces.append(text[st_idx:end_idx].strip())
        text = text[end_idx+3:].strip()
    if concat: return '\n\n'.join(code_pieces)
    return code_pieces
