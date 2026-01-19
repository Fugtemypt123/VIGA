"""Tournament selection algorithm for alchemy runners."""

from pathlib import Path
from typing import Dict, List

from .image_utils import vlm_compare_images


def tournament_select_best(
    candidate_results: List[Dict],
    target_image_path: str,
    model: str = "gpt-4o"
) -> int:
    """Run tournament to select the best candidate using VLM comparison.

    Args:
        candidate_results: List of dicts with keys 'render_dir' (path to render directory).
        target_image_path: Path to target image.
        model: Vision model name.

    Returns:
        Index of the winning candidate.
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

                render_dir1 = candidate_results[idx1]['render_dir']
                render_dir2 = candidate_results[idx2]['render_dir']

                # Find render1.png in each directory
                render_dir1_path = Path(render_dir1)
                render_dir2_path = Path(render_dir2)

                render1_files = sorted(render_dir1_path.glob("render*.png"))
                render2_files = sorted(render_dir2_path.glob("render*.png"))

                if not render1_files or not render2_files:
                    # If no renders, default to first candidate
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
