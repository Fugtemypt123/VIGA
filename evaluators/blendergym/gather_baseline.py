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


def main() -> None:
    """Gather overall scores for BlenderGym baseline results."""
    parser = argparse.ArgumentParser(description='Evaluate baseline blendergym results')
    parser.add_argument('--model', type=str, default='gpt-4o', help='Model to evaluate (e.g., gpt-4o)')
    args = parser.parse_args()
    model = args.model.replace('-', '_').replace('.', '_')
    
    score_file = f'data/blendergym/baseline/{model}/_evaluation/overall_scores.json'
    with open(score_file, 'r') as f:
        scores = json.load(f)
        
    n_clip_penalty = 1.0
    pl_penalty = 0.8
    
    for task_type, task_scores in scores.items():
        if 'num_instances' not in task_scores:
            continue
        missing_rounds = TASK_INSTANCE_COUNT_DICT[task_type] - task_scores['num_instances']
        task_scores['final_n_clip'] = (task_scores['avg_n_clip'] * task_scores['num_instances'] + n_clip_penalty * missing_rounds * TASK_SCALE_DICT[task_type]) / TASK_INSTANCE_COUNT_DICT[task_type]
        task_scores['final_pl'] = (task_scores['avg_pl'] * task_scores['num_instances'] + pl_penalty * missing_rounds * TASK_SCALE_DICT[task_type]) / TASK_INSTANCE_COUNT_DICT[task_type]
        task_scores['failed_instances'] = missing_rounds
        print(f"Task type: {task_type}, Final n_clip: {task_scores['final_n_clip']}, Final pl: {task_scores['final_pl']}")
        
    with open(score_file, 'w') as f:
        json.dump(scores, f, indent=4)

if __name__ == "__main__":
    main() 