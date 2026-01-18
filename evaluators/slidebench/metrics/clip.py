"""CLIP image-pair similarity score."""
from typing import List, Dict

import cv2
import clip
import torch
import argparse
import numpy as np
from PIL import Image

# Lazy-loaded model
_device = None
_model = None
_preprocess = None


def _get_model():
    """Get or initialize the CLIP model."""
    global _device, _model, _preprocess
    if _model is None:
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        _model, _preprocess = clip.load("ViT-B/32", device=_device)
    return _device, _model, _preprocess


def mask_bounding_boxes_with_inpainting(image: Image.Image, bounding_boxes: List) -> Image.Image:
    """Mask bounding boxes in image using inpainting.

    Args:
        image: PIL Image to process.
        bounding_boxes: List of bounding boxes as [x_ratio, y_ratio, w_ratio, h_ratio].

    Returns:
        Inpainted PIL Image with masked regions filled.
    """
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    mask = np.zeros(image_cv.shape[:2], dtype=np.uint8)
    height, width = image_cv.shape[:2]

    for bbox in bounding_boxes:
        x_ratio, y_ratio, w_ratio, h_ratio = bbox
        x = int(x_ratio * width)
        y = int(y_ratio * height)
        w = int(w_ratio * width)
        h = int(h_ratio * height)
        mask[y:y+h, x:x+w] = 255

    inpainted_image = cv2.inpaint(image_cv, mask, 3, cv2.INPAINT_TELEA)
    inpainted_image_pil = Image.fromarray(cv2.cvtColor(inpainted_image, cv2.COLOR_BGR2RGB))

    return inpainted_image_pil


def rescale_and_mask(image_path: str, blocks: List) -> Image.Image:
    """Load image, apply masking, and rescale to square.

    Args:
        image_path: Path to the image file.
        blocks: List of blocks to mask (each with bbox coordinates).

    Returns:
        Processed square PIL Image.
    """
    with Image.open(image_path) as img:
        if len(blocks) > 0:
            img = mask_bounding_boxes_with_inpainting(img, blocks)

        width, height = img.size

        if width < height:
            new_size = (width, width)
        else:
            new_size = (height, height)

        img_resized = img.resize(new_size, Image.LANCZOS)

        return img_resized


def get_clip_similarity(
    image_path1: str,
    image_path2: str,
    blocks1: List[Dict],
    blocks2: List[Dict]
) -> float:
    """Calculate CLIP similarity between two images.

    Args:
        image_path1: Path to first image.
        image_path2: Path to second image.
        blocks1: Blocks to mask in first image.
        blocks2: Blocks to mask in second image.

    Returns:
        Cosine similarity score between image features.
    """
    device, model, preprocess = _get_model()

    image1 = preprocess(rescale_and_mask(image_path1, [block['bbox'] for block in blocks1])).unsqueeze(0).to(device)
    image2 = preprocess(rescale_and_mask(image_path2, [block['bbox'] for block in blocks2])).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features1 = model.encode_image(image1)
        image_features2 = model.encode_image(image2)

    image_features1 /= image_features1.norm(dim=-1, keepdim=True)
    image_features2 /= image_features2.norm(dim=-1, keepdim=True)

    similarity = (image_features1 @ image_features2.T).item()

    return similarity


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLIP image-pair similarity score.")
    parser.add_argument("--image_path_1", type=str, help="Path to the first image.")
    parser.add_argument("--image_path_2", type=str, help="Path to the second image.")
    args = parser.parse_args()

    clip_sim = get_clip_similarity(args.image_path_1, args.image_path_2, [], [])
    print("CLIP similarity score:", clip_sim)
