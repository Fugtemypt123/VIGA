#!/usr/bin/env python3
import os
import sys
import json
from typing import Dict, Any


def load_json(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_round1_avg(instance_data: Dict[str, Any]) -> float:
    """
    Given one instance's detail dict (value under instance_details[instance_key]),
    return the round 1 average score. If round "1" doesn't exist or lacks average_score, return 0.0.
    """
    if not isinstance(instance_data, dict):
        return 0.0
    round1 = instance_data.get("1")
    if not isinstance(round1, dict):
        return 0.0
    # If there is a direct "average_score" for round1, use it (some structures may store it differently)
    direct_avg = round1.get("average_score")
    if isinstance(direct_avg, (int, float)):
        return float(direct_avg)
    # Otherwise, some structures store per-render dicts with their own average and a round_average at this level
    round_avg = round1.get("round_average")
    if isinstance(round_avg, (int, float)):
        return float(round_avg)
    return 0.0


def main():
    json_path = 'output/blendergym_hard/gpt-4o/_evaluation/reference_free_intermediate_scores.json'
    if not os.path.exists(json_path):
        print(f"File not found: {json_path}")
        sys.exit(1)

    data = load_json(json_path)

    # Traverse top-level keys (e.g., level1, level2, ...), then into instance_details
    results = {}
    total_sum = 0.0
    total_count = 0
    # Track per-level aggregates
    level_sums = {}
    level_counts = {}

    for level_key, level_data in data.items():
        if not isinstance(level_data, dict):
            continue
        instance_details = level_data.get("instance_details")
        if not isinstance(instance_details, dict):
            continue
        for instance_key, instance_data in instance_details.items():
            score = extract_round1_avg(instance_data)
            results[instance_key] = score
            total_sum += score
            total_count += 1
            # Parse level from instance key (expected format: "levelX/...")
            level = instance_key.split('/')[0] if isinstance(instance_key, str) else "unknown"
            level_sums[level] = level_sums.get(level, 0.0) + score
            level_counts[level] = level_counts.get(level, 0) + 1

    # Print per-instance scores
    print("Instance,Round1_AverageScore")
    for instance, score in sorted(results.items()):
        print(f"{instance},{score}")

    # Print level-wise averages (e.g., level1, level2)
    print("\nLevel_Average_Round1")
    for level in sorted(level_sums.keys()):
        avg = (level_sums[level] / level_counts[level]) if level_counts[level] else 0.0
        print(f"{level},{avg}")


if __name__ == "__main__":
    main()


