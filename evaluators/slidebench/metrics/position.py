"""Calculate position similarity of two matched blocks."""
from typing import Dict


def calculate_distance_max_1d(x1: float, y1: float, x2: float, y2: float) -> float:
    """Calculate the maximum 1D distance between two points."""
    distance = max(abs(x2 - x1), abs(y2 - y1))
    return distance


def get_x(block: Dict) -> float:
    """Get center x-coordinate of the block.

    Args:
        block: Block dict with 'bbox' key containing [left, top, right, bottom].
    """
    return (block['bbox'][0] + block['bbox'][2]) / 2


def get_y(block: Dict) -> float:
    """Get center y-coordinate of the block.

    Args:
        block: Block dict with 'bbox' key containing [left, top, right, bottom].
    """
    return (block['bbox'][1] + block['bbox'][3]) / 2


def get_position_similarity(block1: Dict, block2: Dict) -> float:
    """Calculate position similarity between two blocks.

    All coordinates are normalized to [0, 1] scale.

    Args:
        block1: First block dict with 'bbox' key.
        block2: Second block dict with 'bbox' key.

    Returns:
        Similarity score between 0.0 and 1.0.
    """
    position_similarity = 1 - calculate_distance_max_1d(
        get_x(block1), get_y(block1), get_x(block2), get_y(block2)
    )
    return max(0.0, position_similarity)


if __name__ == "__main__":
    # Test: all coordinates normalized to [0, 1] scale
    block1 = {"bbox": [0, 0.1, 0.3, 0.8]}
    block2 = {"bbox": [0, 0.5, 0.2, 0.6]}
    print(get_position_similarity(block1, block2))
