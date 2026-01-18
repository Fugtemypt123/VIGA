"""BlenderBench evaluation runner combining reference-based and reference-free metrics."""
import argparse
import os
import sys
import subprocess

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate BlenderBench results')
    parser.add_argument('test_id', type=str, help='Test ID (e.g., 20250815_150016)')
    args = parser.parse_args()

    ref_based_eval_path = 'evaluators/blenderbench/ref_based_eval.py'
    ref_free_eval_path = 'evaluators/blenderbench/ref_free_eval.py'

    # Run reference-based evaluation first so we know the best CLIP rounds.
    subprocess.run([
        "python", ref_based_eval_path,
        args.test_id
    ], check=True)

    eval_dir = os.path.join("output", "blenderbench", args.test_id, "_evaluation")
    best_round_source = os.path.join(eval_dir, "intermediate_scores.json")

    # Run reference-free evaluation restricted to the previously determined best rounds.
    subprocess.run([
        "python", ref_free_eval_path,
        args.test_id,
        "--best_round_source", best_round_source
    ], check=True)
