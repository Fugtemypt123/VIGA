"""SAM Worker for Automatic Image Segmentation.

This script uses Segment Anything Model (SAM) to automatically generate
segmentation masks for all objects in an image, then uses a VLM to identify
and name each object.
"""

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch
from PIL import Image

ROOT: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, "utils", "sam"))
sys.path.append(os.path.join(ROOT, "utils"))

from common import build_client, get_image_base64
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry


def panic_filtering_process(raw_masks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter and deduplicate SAM masks using greedy selection.

    Applies a multi-step filtering process:
    1. Pre-filtering: Remove masks below minimum area threshold
    2. Sorting: Sort by predicted IoU confidence score
    3. Greedy filling: Select masks that contribute sufficient new area
    4. Final selection: Keep top k masks by area

    Args:
        raw_masks: List of mask dictionaries, each containing:
            - 'segmentation': Binary mask (H, W)
            - 'predicted_iou': Confidence score (float)
            - 'area': Pixel area (int)

    Returns:
        Filtered list of mask dictionaries.
    """
    # Step 1: Pre-filtering
    # Remove masks that are too small (e.g., text on boxes)
    min_area_threshold = 100
    candidates = []
    for m in raw_masks:
        if m['area'] > min_area_threshold:
            candidates.append(m)

    if len(candidates) == 0:
        return []

    # Step 2: Sort by confidence score
    # Addresses inappropriate merging by prioritizing high-confidence masks
    candidates.sort(key=lambda x: x['predicted_iou'], reverse=True)

    # Step 3: Greedy filling
    # Select masks that contribute significant new area
    height, width = candidates[0]['segmentation'].shape
    occupancy_mask = np.zeros((height, width), dtype=bool)

    final_masks = []
    MIN_NEW_AREA_RATIO = 0.6

    for mask_data in candidates:
        current_seg = mask_data['segmentation']
        mask_area = mask_data['area']

        # Calculate overlap with already occupied regions
        intersection = np.logical_and(current_seg, occupancy_mask)
        intersection_area = np.count_nonzero(intersection)

        # Calculate how many new pixels this mask contributes
        new_area = mask_area - intersection_area

        # Calculate freshness ratio
        keep_ratio = new_area / mask_area if mask_area > 0 else 0

        # Decision: keep mask if it contributes enough new area
        if keep_ratio > MIN_NEW_AREA_RATIO:
            final_masks.append(mask_data)
            # Update occupancy map
            occupancy_mask = np.logical_or(occupancy_mask, current_seg)

    # Step 4: Keep only top k masks by area
    final_masks.sort(key=lambda x: x['area'], reverse=True)
    k = 15
    final_masks = final_masks[:k]

    return final_masks


def save_mask_as_png(
    mask_data: Dict[str, Any],
    original_image: np.ndarray,
    output_path: str
) -> str:
    """Save a single mask as a PNG image with transparency.

    Pixels where segmentation=1 retain original image colors.
    Pixels where segmentation=0 are set to transparent.

    Args:
        mask_data: Mask dictionary with 'segmentation' key (boolean array).
        original_image: Original RGB image array (H, W, 3).
        output_path: Path where the PNG will be saved.

    Returns:
        The output path where the image was saved.
    """
    m = mask_data['segmentation']  # Boolean array (H, W)
    h, w = m.shape

    # Resize original image if dimensions don't match
    if original_image.shape[:2] != (h, w):
        original_image = cv2.resize(original_image, (w, h))

    # Create RGBA image
    rgba_image = np.zeros((h, w, 4), dtype=np.uint8)

    # Where segmentation=1: use original colors, alpha=255 (opaque)
    rgba_image[m, :3] = original_image[m]
    rgba_image[m, 3] = 255

    # Where segmentation=0: keep as 0 (transparent)

    # Save as PNG
    pil_image = Image.fromarray(rgba_image, 'RGBA')
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    pil_image.save(output_path)
    return output_path


def sanitize_filename(name: str) -> str:
    """Sanitize a string for use as a filename.

    Replaces illegal characters with underscores, keeps only alphanumeric
    characters, underscores, and hyphens.

    Args:
        name: The original filename string.

    Returns:
        Sanitized filename string.
    """
    # Replace non-alphanumeric characters (except underscore and hyphen) with underscore
    sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
    # Remove consecutive underscores
    sanitized = re.sub(r'_+', '_', sanitized)
    # Remove leading and trailing underscores
    sanitized = sanitized.strip('_')
    # Use default name if empty
    if not sanitized:
        sanitized = "object"
    return sanitized


def get_object_name_from_vlm(
    image_path: str,
    ori_img_path: str,
    model: str = "gpt-4o",
    existing_names: Optional[List[str]] = None
) -> str:
    """Use a VLM to identify the object in an image and return a unique name.

    Args:
        image_path: Path to the segmented object PNG image.
        ori_img_path: Path to the original full image for context.
        model: VLM model name to use.
        existing_names: List of already-used names to avoid duplicates.

    Returns:
        A unique object name string.
    """
    if existing_names is None:
        existing_names = []

    try:
        # Encode images
        image_b64 = get_image_base64(image_path)
        ori_img_b64 = get_image_base64(ori_img_path)
        # Initialize OpenAI client
        client = build_client(model)

        # Build prompt including existing names to avoid duplicates
        existing_names_str = ""
        if existing_names:
            existing_names_str = f"\n\nAlready identified objects (do not use these names): {', '.join(existing_names)}"

        # Create messages
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_b64
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": ori_img_b64
                        }
                    },
                    {
                        "type": "text",
                        "text": (
                            "Look at the first image showing a segmented object, and the second "
                            "image showing the original image that contains this object. Identify "
                            "what this object is and provide a concise, descriptive name for it "
                            "(e.g., 'red_chair', 'wooden_table'). If the first image is not clear, "
                            "check the second image to get the whole context.\n\n"
                            "If you think the first image is not an object, but a background, "
                            "please return 'background' as the object name.\n\n"
                            "Use only lowercase letters, numbers, and underscores. The name should "
                            "be a single word or short phrase (2-3 words max, use underscores to "
                            f"separate words).{existing_names_str}\n\n"
                            "Respond with ONLY the object name, nothing else."
                        )
                    }
                ]
            }
        ]

        # Call API
        response = client.chat.completions.create(model=model, messages=messages)

        # Parse response
        object_name = response.choices[0].message.content.strip()

        # Clean name: remove quotes, extra spaces
        object_name = object_name.strip('"\'')
        object_name = re.sub(r'\s+', '_', object_name)
        object_name = sanitize_filename(object_name)

        # Ensure uniqueness: add numeric suffix if duplicate
        base_name = object_name
        counter = 1
        while object_name in existing_names:
            object_name = f"{base_name}_{counter}"
            counter += 1

        return object_name

    except Exception as e:
        print(f"VLM naming failed: {e}, using fallback name")
        # Use default name if VLM call fails
        base_name = "object"
        counter = 1
        while f"{base_name}_{counter}" in existing_names:
            counter += 1
        return f"{base_name}_{counter}"


def main() -> None:
    """Run the SAM worker to segment and identify objects in an image."""
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True, help="Path to input image")
    p.add_argument("--out", required=True, help="Path for output npy file")
    p.add_argument("--checkpoint", default=None, help="Path to SAM checkpoint")
    p.add_argument("--vlm-model", default="gpt-4o", help="VLM model for object identification")
    args = p.parse_args()

    # Set default checkpoint path
    if args.checkpoint is None:
        args.checkpoint = os.path.join(ROOT, "utils", "sam", "sam_vit_h_4b8939.pth")

    # Load image
    image = cv2.imread(args.image)
    if image is None:
        raise ValueError(f"Failed to load image: {args.image}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Initialize SAM model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_type = "vit_h"

    sam = sam_model_registry[model_type](checkpoint=args.checkpoint)
    sam.to(device=device)

    # Generate all masks
    mask_generator = SamAutomaticMaskGenerator(sam)
    raw_masks = mask_generator.generate(image)

    # Apply panic filtering process
    filtered_masks = panic_filtering_process(raw_masks)

    if len(filtered_masks) == 0:
        print("Warning: No masks after filtering")
        return

    # Determine output directory
    output_dir = os.path.dirname(args.out)
    os.makedirs(output_dir, exist_ok=True)

    # Save PNG for each mask and use VLM for naming
    object_names: List[str] = []
    mask_files: List[Dict[str, Any]] = []

    print(f"Processing {len(filtered_masks)} filtered masks...")
    for idx, mask_data in enumerate(filtered_masks):
        # 1. Save PNG (temporary filename, will be renamed later)
        temp_png_path = os.path.join(output_dir, f"temp_mask_{idx}.png")
        ori_img_path = args.image
        save_mask_as_png(mask_data, image, temp_png_path)

        # 2. Use VLM to identify object and get name
        print(f"  Identifying object {idx+1}/{len(filtered_masks)}...")
        object_name = get_object_name_from_vlm(
            temp_png_path, ori_img_path, model=args.vlm_model, existing_names=object_names
        )
        if object_name == "background":
            os.remove(temp_png_path)
            continue
        object_names.append(object_name)

        # 3. Rename PNG file to object_ID.png
        final_png_path = os.path.join(output_dir, f"{object_name}.png")
        if os.path.exists(final_png_path):
            # Add numeric suffix if file exists
            counter = 1
            while os.path.exists(final_png_path):
                final_png_path = os.path.join(output_dir, f"{object_name}_{counter}.png")
                counter += 1
        os.rename(temp_png_path, final_png_path)
        print(f"    Identified as: {object_name}")

        # 4. Save corresponding npy file as object_ID.npy
        seg = mask_data['segmentation']  # Boolean array
        mask_uint8 = (seg.astype(np.uint8)) * 255  # Convert to 0/255

        npy_path = os.path.join(output_dir, f"{object_name}.npy")
        if os.path.exists(npy_path):
            # Add numeric suffix if file exists
            counter = 1
            while os.path.exists(npy_path):
                npy_path = os.path.join(output_dir, f"{object_name}_{counter}.npy")
                counter += 1
        np.save(npy_path, mask_uint8)

        mask_files.append({
            "object_id": object_name,
            "png_path": final_png_path,
            "npy_path": npy_path,
            "mask_data": mask_uint8
        })

    # 5. Save combined masks array for backward compatibility
    if args.out:
        mask_arrays = [item["mask_data"] for item in mask_files]
        if mask_arrays:
            mask_arrays = np.stack(mask_arrays, axis=0)
            np.save(args.out, mask_arrays)
            print(f"Also saved combined masks to: {args.out}")

        # Save object names mapping for sam_init.py
        object_names_json_path = args.out.replace('.npy', '_object_names.json')
        # Extract final filename (without extension) from npy_path, may include conflict suffix
        object_mapping = []
        for item in mask_files:
            npy_path = item["npy_path"]
            npy_filename = os.path.basename(npy_path)
            object_name_final = os.path.splitext(npy_filename)[0]
            object_mapping.append(object_name_final)

        object_names_info = {
            "object_names": object_names,
            "object_mapping": object_mapping
        }
        with open(object_names_json_path, 'w') as f:
            json.dump(object_names_info, f, indent=2)
        print(f"Also saved object names mapping to: {object_names_json_path}")

    print(f"\nGenerated {len(raw_masks)} raw masks, filtered to {len(filtered_masks)} masks")
    print(f"Identified objects: {', '.join(object_names)}")
    print(f"Saved {len(mask_files)} mask files to: {output_dir}")


if __name__ == "__main__":
    main()
