import os
import json
from PIL import Image
import io
import base64
from typing import Dict, List, Optional
from openai import OpenAI
import time
from utils._api_keys import OPENAI_API_KEY, OPENAI_BASE_URL, CLAUDE_API_KEY, CLAUDE_BASE_URL, GEMINI_API_KEY, GEMINI_BASE_URL, QWEN_BASE_URL, MESHY_API_KEY, VA_API_KEY
from pathlib import Path

def get_model_response(client: OpenAI, chat_args: Dict, num_candidates: int):
    # repeat multiple time to avoid network errors
    # select the best candidate from the responses
    candidate_responses = []
    for idx in range(num_candidates):
        max_retries = 1
        while max_retries > 0:
            try:
                response = client.chat.completions.create(**chat_args)
                candidate_responses.append(response)
                break
            except Exception as e:
                max_retries -= 1
                time.sleep(10)
    if len(candidate_responses) == 0:
        raise Exception("Failed to get model response")
    return candidate_responses

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

def tournament_select_best(candidate_results: List[Dict], target_image_path: str, model: str = "gpt-4o") -> int:
    """
    Run tournament to select the best candidate using VLM comparison.
    
    Args:
        candidate_results: List of dicts with keys 'render_dir' (path to render directory)
        target_image_path: Path to target image
        model: Vision model name
        
    Returns:
        Index of the winning candidate
    """
    if len(candidate_results) == 0:
        return 0
    
    if len(candidate_results) == 1:
        return 0
    
    # Tournament: keep pairing and comparing until one winner
    current_candidates = list(range(len(candidate_results)))
    
    while len(current_candidates) > 1:
        next_round = []
        
        # Pair up candidates
        for i in range(0, len(current_candidates), 2):
            if i + 1 < len(current_candidates):
                idx1 = current_candidates[i]
                idx2 = current_candidates[i + 1]
                
                render1_files = candidate_results[idx1].get('image', [])
                render2_files = candidate_results[idx2].get('image', [])
                
                if not render1_files:
                    # If no renders, default to first candidate
                    next_round.append(idx2)
                    continue
                elif not render2_files:
                    next_round.append(idx1)
                    continue
                
                img1_path = str(render1_files[0])
                img2_path = str(render2_files[0])
                
                # Compare which is closer to target
                winner = vlm_compare_images(img1_path, img2_path, target_image_path, model)
                
                # Winner is 1 or 2, convert to index
                winner_idx = idx1 if winner == 1 else idx2
                next_round.append(winner_idx)
            else:
                # Odd number, last one gets bye
                next_round.append(current_candidates[i])
        
        current_candidates = next_round
    
    return current_candidates[0]

def vlm_compare_images(image1_path: str, image2_path: str, target_path: str, model: str = "gpt-4o") -> int:
    """
    Use VLM to compare two images and determine which is closer to target.
    
    Args:
        image1_path: Path to first image
        image2_path: Path to second image  
        target_path: Path to target image
        model: Vision model to use
        
    Returns:
        1 if image1 is closer to target, 2 if image2 is closer to target
    """
    try:
        # Encode images
        image1_b64 = get_image_base64(image1_path)
        image2_b64 = get_image_base64(image2_path)
        if os.path.isdir(target_path):
            target_new_path = os.path.join(target_path, 'visprompt1.png')
            if not os.path.exists(target_new_path):
                target_new_path = os.path.join(target_path, 'style1.png')
                if not os.path.exists(target_new_path):
                    target_new_path = os.path.join(target_path, 'render1.png')
        else:
            target_new_path = target_path
        target_b64 = get_image_base64(target_new_path)
        
        # Initialize OpenAI client
        client = build_client(model)
        
        # Create messages
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "You are an expert at comparing 3D rendered images. I will show you two rendered images and a target image. Please determine which of the two rendered images is closer to the target image in terms of visual similarity, lighting, materials, geometry, and overall appearance. Respond with only '1' if the first image is closer to the target, or '2' if the second image is closer to the target."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": target_b64
                        }
                    },
                    {
                        "type": "text", 
                        "text": "Target image:"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image1_b64
                        }
                    },
                    {
                        "type": "text",
                        "text": "Image 1:"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image2_b64
                        }
                    },
                    {
                        "type": "text",
                        "text": "Image 2:"
                    }
                ]
            }
        ]
        
        # Make API call
        response = client.chat.completions.create(model=model, messages=messages)
        
        # Parse response
        result = response.choices[0].message.content.strip()
        if result == "1":
            return 1
        elif result == "2":
            return 2
        else:
            # Default to image1 if response is unclear
            print(f"Unexpected VLM response: {result}, defaulting to image1")
            return 1
            
    except Exception as e:
        print(f"VLM comparison failed: {e}, defaulting to image1")
        return 1