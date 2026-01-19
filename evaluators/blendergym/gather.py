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

# Task instance counts for different task types
TASK_INSTANCE_COUNT_DICT = {
    'geometry': 50,
    'material': 40,
    'blendshape': 75,
    'placement': 40,
    'lighting': 35
}

TASK_SCALE_DICT = {
    'geometry': 1.0,
    'material': 1.0,
    'blendshape': 0.1,
    'placement': 0.1,
    'lighting': 1.0
}

def aggregate_per_round_scores(scores_across_instances: Dict[str, Any], 
                              max_rounds: int = 10) -> Dict[str, Any]:
    """
    Aggregate per-round averages across all instances (rounds 1..10).
    
    Args:
        scores_across_instances: The intermediate scores structure
        max_rounds: Maximum number of rounds to consider
        
    Returns:
        Dictionary with per-round summary statistics
    """
    per_round_values = {str(i): {'n_clip': [], 'pl': []} for i in range(1, max_rounds + 1)}
    
    for instance_scores in scores_across_instances['instance_details'].values():
        # Process each round
        for round_idx in range(1, max_rounds + 1):
            key = str(round_idx)
            # Only process rounds that exist
            if key in instance_scores and 'avg_n_clip' in instance_scores[key] and 'avg_pl' in instance_scores[key]:
                per_round_values[key]['n_clip'].append(instance_scores[key]['avg_n_clip'])
                per_round_values[key]['pl'].append(instance_scores[key]['avg_pl'])

    per_round_summary = {}
    for key, vals in per_round_values.items():
        if vals['n_clip'] and vals['pl']:
            per_round_summary[key] = {
                'avg_n_clip': sum(vals['n_clip']) / len(vals['n_clip']),
                'avg_pl': sum(vals['pl']) / len(vals['pl']),
                'num_instances': len(vals['n_clip'])
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


def compute_worst_scores(scores_across_instances: Dict[str, Any]) -> tuple:
    """
    Compute worst round scores from instance details, similar to best scores logic but finding max values.
    
    Args:
        scores_across_instances: The intermediate scores structure
        
    Returns:
        tuple: (worst_n_clip_list, worst_pl_list)
    """
    worst_n_clip_list = []
    worst_pl_list = []
    
    for instance_scores in scores_across_instances['instance_details'].values():
        # Find valid rounds for this instance
        valid_rounds = {k: v for k, v in instance_scores.items() 
                       if isinstance(v, dict) and 'avg_n_clip' in v and 'avg_pl' in v}
        
        if valid_rounds:
            # Get the worst round (highest n_clip value, which means worst performance)
            worst_round_key = max(valid_rounds.keys(), key=lambda r: valid_rounds[r]['avg_n_clip'])
            worst_n_clip = valid_rounds[worst_round_key]['avg_n_clip']
            worst_pl = valid_rounds[worst_round_key]['avg_pl']
            worst_n_clip_list.append(worst_n_clip)
            worst_pl_list.append(worst_pl)
    
    return worst_n_clip_list, worst_pl_list


def compute_round1_averages(scores_across_instances: Dict[str, Any]) -> tuple:
    """
    Compute round 1 averages from actual round 1 scores only.
    
    Args:
        scores_across_instances: The intermediate scores structure
        
    Returns:
        tuple: (round1_n_clip_list, round1_pl_list)
    """
    round1_n_clip_list = []
    round1_pl_list = []
    
    for instance_scores in scores_across_instances['instance_details'].values():
        # Only include instances where round 1 exists
        if '1' in instance_scores and isinstance(instance_scores['1'], dict) and 'avg_n_clip' in instance_scores['1']:
            round1_n_clip = instance_scores['1']['avg_n_clip']
            round1_pl = instance_scores['1']['avg_pl']
            round1_n_clip_list.append(round1_n_clip)
            round1_pl_list.append(round1_pl)
    
    return round1_n_clip_list, round1_pl_list


def compute_overall_scores(intermediates: Dict[str, Any], 
                          max_rounds: int = 10) -> Dict[str, Any]:
    """
    Compute overall scores from intermediate scores.
    
    Args:
        intermediates: The intermediate scores loaded from JSON
        max_rounds: Maximum number of rounds to consider
        
    Returns:
        Dictionary with overall scores for each task type
    """
    scores_across_tasks = {}
    
    for task_type, scores_across_instances in intermediates.items():
        print(f"Processing task type: {task_type}")
        
        # Aggregate per-round averages across all instances
        per_round_summary = aggregate_per_round_scores(
            scores_across_instances, max_rounds
        )
        
        # Compute last round scores if not already present
        if not scores_across_instances.get('last_round_n_clip'):
            last_round_n_clip_list, last_round_pl_list = compute_last_round_scores(scores_across_instances)
            scores_across_instances['last_round_n_clip'] = last_round_n_clip_list
            scores_across_instances['last_round_pl'] = last_round_pl_list
        
        # Compute worst scores if not already present
        if not scores_across_instances.get('worst_n_clip'):
            worst_n_clip_list, worst_pl_list = compute_worst_scores(scores_across_instances)
            scores_across_instances['worst_n_clip'] = worst_n_clip_list
            scores_across_instances['worst_pl'] = worst_pl_list
        
        # Compute round 1 averages from actual round 1 scores only
        round1_n_clip_list, round1_pl_list = compute_round1_averages(scores_across_instances)
        scores_across_instances['round1_n_clip'] = round1_n_clip_list
        scores_across_instances['round1_pl'] = round1_pl_list
        
        # Aggregate results for this task type
        if scores_across_instances.get('best_n_clip'):
            scores_across_tasks[task_type] = {
                'best_n_clip': sum(scores_across_instances['best_n_clip']) / len(scores_across_instances['best_n_clip']),
                'best_pl': sum(scores_across_instances['best_pl']) / len(scores_across_instances['best_pl']),
                'worst_n_clip': sum(scores_across_instances['worst_n_clip']) / len(scores_across_instances['worst_n_clip']),
                'worst_pl': sum(scores_across_instances['worst_pl']) / len(scores_across_instances['worst_pl']),
                'last_round_n_clip': sum(scores_across_instances['last_round_n_clip']) / len(scores_across_instances['last_round_n_clip']),
                'last_round_pl': sum(scores_across_instances['last_round_pl']) / len(scores_across_instances['last_round_pl']),
                'round1_n_clip': sum(scores_across_instances['round1_n_clip']) / len(scores_across_instances['round1_n_clip']),
                'round1_pl': sum(scores_across_instances['round1_pl']) / len(scores_across_instances['round1_pl']),
                'num_instances': len(scores_across_instances['best_n_clip']),
                'per_round': per_round_summary
            }

            print(f"  Task {task_type} overall scores:")
            print(f"    Average best n_clip: {scores_across_tasks[task_type]['best_n_clip']}")
            print(f"    Average best pl: {scores_across_tasks[task_type]['best_pl']}")
            print(f"    Average worst n_clip: {scores_across_tasks[task_type]['worst_n_clip']}")
            print(f"    Average worst pl: {scores_across_tasks[task_type]['worst_pl']}")
            print(f"    Average last round n_clip: {scores_across_tasks[task_type]['last_round_n_clip']}")
            print(f"    Average last round pl: {scores_across_tasks[task_type]['last_round_pl']}")
            print(f"    Average round 1 n_clip: {scores_across_tasks[task_type]['round1_n_clip']}")
            print(f"    Average round 1 pl: {scores_across_tasks[task_type]['round1_pl']}")
            print(f"    Number of instances: {scores_across_tasks[task_type]['num_instances']}")
        else:
            print(f"  No valid scores for task type {task_type}")
            scores_across_tasks[task_type] = {}
    
    return scores_across_tasks


def main() -> None:
    """Gather overall scores from intermediate evaluation scores."""
    parser = argparse.ArgumentParser(description='Gather overall scores from intermediate scores')
    parser.add_argument('test_id', type=str, help='Test ID (e.g., 20250815_150016)')
    parser.add_argument('--output_file', type=str, default=None, 
                       help='Output file path (default: overall_scores.json in same directory)')
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
        args.max_rounds
    )
    
    # Save results
    print(f"Saving overall scores to: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(scores_across_tasks, f, indent=4)
        
    n_clip_penalty = 1.0
    pl_penalty = 0.8
    
    for task_type, task_scores in scores_across_tasks.items():
        if 'num_instances' not in task_scores:
            continue
        missing_rounds = TASK_INSTANCE_COUNT_DICT[task_type] - task_scores['num_instances']
        task_scores['final_n_clip'] = (task_scores['best_n_clip'] * task_scores['num_instances'] + n_clip_penalty * missing_rounds * TASK_SCALE_DICT[task_type]) / TASK_INSTANCE_COUNT_DICT[task_type]
        task_scores['final_pl'] = (task_scores['best_pl'] * task_scores['num_instances'] + pl_penalty * missing_rounds * TASK_SCALE_DICT[task_type]) / TASK_INSTANCE_COUNT_DICT[task_type]
        task_scores['failed_instances'] = missing_rounds
        print(f"Task type: {task_type}, Final n_clip: {task_scores['final_n_clip']}, Final pl: {task_scores['final_pl']}")
        
    with open(output_path, 'w') as f:
        json.dump(scores_across_tasks, f, indent=4)
    
    # Print summary
    print(f"\n=== Summary ===")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_path}")
    
    for task_type, scores in scores_across_tasks.items():
        if scores:
            print(f"\n{task_type.upper()}:")
            print(f"  Average best n_clip: {scores['best_n_clip']}")
            print(f"  Average best pl: {scores['best_pl']}")
            print(f"  Final n_clip: {scores['final_n_clip']}")
            print(f"  Final pl: {scores['final_pl']}")
            print(f"  Failed instances: {scores['failed_instances']}")
            print(f"  Instances evaluated: {scores['num_instances']}")
        else:
            print(f"\n{task_type.upper()}: No valid scores")


if __name__ == "__main__":
    main() 