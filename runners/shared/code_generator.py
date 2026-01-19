"""Code generation utilities for alchemy runners."""

import os
import re
import sys
from typing import List

# Import from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.common import build_client

from .image_utils import encode_image


def generate_candidate_codes(
    start_image_path: str,
    current_image_path: str,
    current_code: str,
    target_image_path: str,
    task_description: str,
    model: str = "gpt-4o",
    num_candidates: int = 4
) -> List[str]:
    """Use GPT to generate multiple candidate codes to transform current image to target.

    Args:
        start_image_path: Path to starting image.
        current_image_path: Path to current image.
        current_code: Current Blender Python code.
        target_image_path: Path to target image.
        task_description: Task description text.
        model: Model name.
        num_candidates: Number of candidate codes to generate (3-4).

    Returns:
        List of candidate code strings.
    """
    try:
        # Encode images
        start_b64 = encode_image(start_image_path)
        current_b64 = encode_image(current_image_path)
        target_b64 = encode_image(target_image_path)
        client = build_client(model)

        # Create messages
        messages = [
            {
                "role": "system",
                "content": "You are an expert at writing Blender Python code to transform 3D scenes. Given a starting image, current image, current code, and target image, generate multiple candidate code solutions."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""Task description: {task_description}

You are given:
1. Starting image (initial state)
2. Current image (current state after applying current code)
3. Current Blender Python code
4. Target image (desired final state)

Please generate {num_candidates} different candidate Blender Python code solutions that can transform the current image closer to the target image. Each candidate should be a complete, runnable Blender Python script.

Current code:
```python
{current_code}
```

Please output {num_candidates} complete code solutions, separated by "===CANDIDATE_1===", "===CANDIDATE_2===", etc. Each code block should be complete and executable."""
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{start_b64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": "Starting image:"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{current_b64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": "Current image:"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{target_b64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": "Target image:"
                    }
                ]
            }
        ]

        # Make API call
        response = client.chat.completions.create(model=model, messages=messages)

        # Parse response to extract candidate codes
        content = response.choices[0].message.content
        candidates = []

        # Split by candidate markers
        parts = content.split("===CANDIDATE_")
        for i, part in enumerate(parts[1:], 1):  # Skip first empty part
            # Extract code between markers
            if "===" in part:
                code = part.split("===", 1)[1]
                # Remove markdown code blocks if present
                code = code.replace("```python", "").replace("```", "").strip()
                candidates.append(code)
            else:
                # Last candidate or no marker
                code = part.strip()
                code = code.replace("```python", "").replace("```", "").strip()
                if code:
                    candidates.append(code)

        # If no markers found, try to extract code blocks
        if len(candidates) == 0:
            code_blocks = re.findall(r'```(?:python)?\n(.*?)\n```', content, re.DOTALL)
            candidates = code_blocks[:num_candidates]

        # Ensure we have the right number of candidates
        while len(candidates) < num_candidates and len(candidates) > 0:
            candidates.append(candidates[-1])  # Duplicate last candidate if needed

        return candidates[:num_candidates]

    except Exception as e:
        print(f"Error generating candidate codes: {e}")
        # Return empty codes as fallback
        return [current_code] * num_candidates
