"""Calculate color similarity of two matched blocks using CIEDE2000."""
import argparse
from typing import Tuple

import numpy as np
import pptx
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
from colormath.color_objects import LabColor, sRGBColor
from PIL import Image
from pptx.enum.dml import MSO_COLOR_TYPE, MSO_FILL_TYPE

# Compatibility patch for numpy.asscalar (removed in numpy 1.25+)
if not hasattr(np, "asscalar"):
    np.asscalar = lambda a: a.item()


def rgb_to_lab(rgb: tuple[int, int, int]) -> LabColor:
    """Convert an RGB color to Lab color space.

    Args:
        rgb: RGB tuple with values in range [0, 255].

    Returns:
        Lab color object.
    """
    rgb_color = sRGBColor(rgb[0], rgb[1], rgb[2], is_upscaled=True)
    lab_color = convert_color(rgb_color, LabColor)
    return lab_color


def color_similarity_ciede2000(
    rgb1: tuple[int, int, int], rgb2: tuple[int, int, int]
) -> float:
    """Calculate color similarity between two RGB colors using CIEDE2000.

    Args:
        rgb1: First RGB color tuple.
        rgb2: Second RGB color tuple.

    Returns:
        Similarity score between 0 and 1, where 1 means identical.
    """
    lab1 = rgb_to_lab(rgb1)
    lab2 = rgb_to_lab(rgb2)

    delta_e = delta_e_cie2000(lab1, lab2)

    # Normalize: delta_e of 0 means identical, 100 means completely different
    similarity = max(0, 1 - (delta_e / 100))

    return similarity


def get_color_similarity(
    color1: tuple[int, int, int] | None, color2: tuple[int, int, int] | None
) -> float:
    """Calculate color similarity between two colors.

    Args:
        color1: First color as RGB tuple, or None.
        color2: Second color as RGB tuple, or None.

    Returns:
        Similarity score between 0.0 and 1.0.
    """
    if (color1 is None) and (color2 is None):
        return 1.0
    elif (color1 is None) or (color2 is None):
        return 0.0
    return color_similarity_ciede2000(color1, color2)


def get_shape_fill_similarity(
    shape1: bytes | pptx.dml.fill.FillFormat,
    shape2: bytes | pptx.dml.fill.FillFormat,
) -> float:
    """Calculate fill similarity between two shapes.

    Args:
        shape1: First shape's fill (bytes for image, FillFormat for color).
        shape2: Second shape's fill (bytes for image, FillFormat for color).

    Returns:
        Similarity score: 1.0 for identical, 0.0 for different.
    """
    # If image fill, compare if image blob is the same
    if isinstance(shape1, bytes) or isinstance(shape2, bytes):
        return float(shape1 == shape2)

    # If color fill, compare the color similarity
    if shape1.type == shape2.type == MSO_FILL_TYPE.SOLID:
        if shape1.fore_color.type == shape2.fore_color.type == MSO_COLOR_TYPE.RGB:
            color1 = shape1.fore_color.rgb
            color2 = shape2.fore_color.rgb
            return get_color_similarity(color1, color2)
        elif shape1.fore_color.type == shape2.fore_color.type == MSO_COLOR_TYPE.SCHEME:
            return float(shape1.fore_color.theme_color == shape2.fore_color.theme_color)
        else:
            return float(shape1 == shape2)
    elif shape1.type == shape2.type == MSO_FILL_TYPE.BACKGROUND:
        return 1.0
    else:
        return 0.0


def _test() -> None:
    """Test color similarity calculation."""

    def average_color(image_path: str, coordinates: list) -> Tuple[int, ...]:
        """Calculate the average color at specified coordinates."""
        image_array = np.array(Image.open(image_path).convert("RGB"))
        colors = [image_array[x, y] for x, y in coordinates]
        avg_color = np.mean(colors, axis=0)
        return tuple(avg_color.astype(int))

    image_path_1 = "slides-bench/test/page1/page1.jpg"
    image_path_2 = "slides-bench/test/page1/gpt-4o-mini.png"
    color1 = average_color(image_path_1, [(0, 2)])
    color2 = average_color(image_path_2, [(1, 2)])
    color_sim = get_color_similarity(color1, color2)
    print("Color similarity score:", color_sim)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Color similarity score.")
    parser.add_argument("--image_path_1", type=str, help="Path to the first image.")
    parser.add_argument("--image_path_2", type=str, help="Path to the second image.")
    args = parser.parse_args()

    _test()
