#!/usr/bin/env python3
"""
Gather script to compute overall_scores.json from intermediate_scores.json.
This script extracts the aggregation logic from evaluate.py to allow
post-processing of intermediate results.
"""

import os
import sys
import argparse
import json
from typing import Dict, Any, List


def aggregate_per_round_scores(scores_across_instances: Dict[str, Any], 
                              penalty_factor: float = 2.0,
                              max_rounds: int = 10) -> Dict[str, Any]:
    """
    Aggregate per-round averages across all instances (rounds 1..10).
    
    Args:
        scores_across_instances: The intermediate scores structure
        penalty_factor: Fixed penalty factor for missing earlier rounds
        max_rounds: Maximum number of rounds to consider
        
    Returns:
        Dictionary with per-round summary statistics
    """
    per_round_values = {str(i): {'n_clip': [], 'pl': [], 'penalized_count': 0} for i in range(1, max_rounds + 1)}
    
    for instance_scores in scores_across_instances['instance_details'].values():
        # Collect available round indices for this instance
        available_rounds = sorted(
            [int(r) for r, v in instance_scores.items() if isinstance(v, dict) and 'avg_n_clip' in v and 'avg_pl' in v]
        )
        if not available_rounds:
            continue
        max_available_round = max(available_rounds)

        # Process each round
        for round_idx in range(1, max_rounds + 1):
            key = str(round_idx)
            
            # Case 1: round exists normally
            if key in instance_scores and 'avg_n_clip' in instance_scores[key] and 'avg_pl' in instance_scores[key]:
                per_round_values[key]['n_clip'].append(instance_scores[key]['avg_n_clip'])
                per_round_values[key]['pl'].append(instance_scores[key]['avg_pl'])
                continue

            # Case 2: earlier round missing but later rounds exist -> apply fixed penalty
            if round_idx < max_available_round:
                # Find the next available later round to base the penalty on
                later_rounds = [r for r in available_rounds if r > round_idx]
                if not later_rounds:
                    continue
                next_round = min(later_rounds)
                next_key = str(next_round)
                base_n = instance_scores[next_key]['avg_n_clip']
                base_pl = instance_scores[next_key]['avg_pl']
                # Apply fixed penalty factor
                per_round_values[key]['n_clip'].append(base_n * penalty_factor)
                per_round_values[key]['pl'].append(base_pl * penalty_factor)
                per_round_values[key]['penalized_count'] += 1
                continue
            
            # Case 3: missing because process ended -> apply exponential decay
            if round_idx > max_available_round:
                # Find the last available round to base the decay on
                last_round = max_available_round
                last_key = str(last_round)
                base_n = instance_scores[last_key]['avg_n_clip']
                base_pl = instance_scores[last_key]['avg_pl']
                # Calculate exponential decay: divide by 2 for each round after the last available
                decay_factor = penalty_factor ** (round_idx - last_round)
                per_round_values[key]['n_clip'].append(base_n / decay_factor)
                per_round_values[key]['pl'].append(base_pl / decay_factor)
                per_round_values[key]['penalized_count'] += 1
                continue

    per_round_summary = {}
    for key, vals in per_round_values.items():
        if vals['n_clip'] and vals['pl']:
            per_round_summary[key] = {
                'avg_n_clip': sum(vals['n_clip']) / len(vals['n_clip']),
                'avg_pl': sum(vals['pl']) / len(vals['pl']),
                'num_instances': len(vals['n_clip']),
                'num_penalized': int(vals['penalized_count'])
            }

    return per_round_summary


def compute_last_round_scores(scores_across_instances: Dict[str, Any]) -> tuple:
    """
    Compute last round scores from instance details, similar to evaluate.py logic.
    
    Args:
        scores_across_instances: The intermediate scores structure
        
    Returns:
        tuple: (last_round_n_clip_list, last_round_pl_list)
    """
    last_round_n_clip_list = []
    last_round_pl_list = []
    
    for instance_scores in scores_across_instances['instance_details'].values():
        # Find valid rounds for this instance
        valid_rounds = {k: v for k, v in instance_scores.items() 
                       if isinstance(v, dict) and 'avg_n_clip' in v and 'avg_pl' in v}
        
        if valid_rounds:
            # Get the last round (highest round number)
            last_round_key = str(max(valid_rounds.keys()))
            last_round_n_clip = valid_rounds[last_round_key]['avg_n_clip']
            last_round_pl = valid_rounds[last_round_key]['avg_pl']
            last_round_n_clip_list.append(last_round_n_clip)
            last_round_pl_list.append(last_round_pl)
    
    return last_round_n_clip_list, last_round_pl_list


def compute_overall_scores(intermediates: Dict[str, Any], 
                          penalty_factor: float = 2.0,
                          max_rounds: int = 10) -> Dict[str, Any]:
    """
    Compute overall scores from intermediate scores.
    
    Args:
        intermediates: The intermediate scores loaded from JSON
        penalty_factor: Fixed penalty factor for missing earlier rounds
        max_rounds: Maximum number of rounds to consider
        
    Returns:
        Dictionary with overall scores for each task type
    """
    scores_across_tasks = {}
    
    for task_type, scores_across_instances in intermediates.items():
        print(f"Processing task type: {task_type}")
        
        # Aggregate per-round averages across all instances
        per_round_summary = aggregate_per_round_scores(
            scores_across_instances, penalty_factor, max_rounds
        )
        
        # Compute last round scores if not already present
        if not scores_across_instances.get('last_round_n_clip'):
            last_round_n_clip_list, last_round_pl_list = compute_last_round_scores(scores_across_instances)
            scores_across_instances['last_round_n_clip'] = last_round_n_clip_list
            scores_across_instances['last_round_pl'] = last_round_pl_list
        
        # Aggregate results for this task type
        if scores_across_instances.get('best_n_clip'):
            scores_across_tasks[task_type] = {
                'best_n_clip': sum(scores_across_instances['best_n_clip']) / len(scores_across_instances['best_n_clip']),
                'best_pl': sum(scores_across_instances['best_pl']) / len(scores_across_instances['best_pl']),
                'last_round_n_clip': sum(scores_across_instances['last_round_n_clip']) / len(scores_across_instances['last_round_n_clip']),
                'last_round_pl': sum(scores_across_instances['last_round_pl']) / len(scores_across_instances['last_round_pl']),
                'num_instances': len(scores_across_instances['best_n_clip']),
                'per_round': per_round_summary
            }

            print(f"  Task {task_type} overall scores:")
            print(f"    Average best n_clip: {scores_across_tasks[task_type]['best_n_clip']:.4f}")
            print(f"    Average best pl: {scores_across_tasks[task_type]['best_pl']:.4f}")
            print(f"    Average last round n_clip: {scores_across_tasks[task_type]['last_round_n_clip']:.4f}")
            print(f"    Average last round pl: {scores_across_tasks[task_type]['last_round_pl']:.4f}")
            print(f"    Number of instances: {scores_across_tasks[task_type]['num_instances']}")
        else:
            print(f"  No valid scores for task type {task_type}")
            scores_across_tasks[task_type] = {}
    
    return scores_across_tasks


def main():
    parser = argparse.ArgumentParser(description='Gather overall scores from intermediate scores')
    parser.add_argument('test_id', type=str, help='Test ID (e.g., 20250815_150016)')
    parser.add_argument('--output_file', type=str, default=None, 
                       help='Output file path (default: overall_scores.json in same directory)')
    parser.add_argument('--missing_round_penalty_factor', type=float, default=2.0,
                        help='Fixed penalty factor for missing earlier rounds.')
    parser.add_argument('--max_rounds', type=int, default=10,
                        help='Maximum number of rounds to consider.')
    
    args = parser.parse_args()
    
    input_file = f'output/blendergym/{args.test_id}/_evaluation/intermediate_scores.json'
    
    # Load intermediate scores
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file {input_file} does not exist.")
    
    print(f"Loading intermediate scores from: {input_file}")
    with open(input_file, 'r') as f:
        intermediates = json.load(f)
    
    # Determine output file path
    if args.output_file:
        output_path = args.output_file
    else:
        input_dir = os.path.dirname(input_file)
        output_path = os.path.join(input_dir, 'overall_scores.json')
    
    # Compute overall scores
    print(f"Computing overall scores...")
    scores_across_tasks = compute_overall_scores(
        intermediates, 
        args.missing_round_penalty_factor,
        args.max_rounds
    )
    
    # Save results
    print(f"Saving overall scores to: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(scores_across_tasks, f, indent=4)
    
    # Print summary
    print(f"\n=== Summary ===")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_path}")
    
    for task_type, scores in scores_across_tasks.items():
        if scores:
            print(f"\n{task_type.upper()}:")
            print(f"  Average best n_clip: {scores['best_n_clip']:.4f}")
            print(f"  Average best pl: {scores['best_pl']:.4f}")
            print(f"  Average last round n_clip: {scores['last_round_n_clip']:.4f}")
            print(f"  Average last round pl: {scores['last_round_pl']:.4f}")
            print(f"  Instances evaluated: {scores['num_instances']}")
        else:
            print(f"\n{task_type.upper()}: No valid scores")


if __name__ == "__main__":
    main() 