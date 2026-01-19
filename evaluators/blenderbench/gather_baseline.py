"""Gather and aggregate BlenderBench baseline evaluation results."""
import json
import argparse

TASK_INSTANCE_COUNT_DICT = {
    'level1': 9,
    'level2': 9,
    'level3': 9,
}

TASK_SCALE_DICT = {
    'level1': 1.0,
    'level2': 1.0,
    'level3': 1.0,
}

n_clip_penalty = 1.0
pl_penalty = 0.8
vlm_penalty = 0.0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Gather baseline BlenderStudio results')
    parser.add_argument('--model', type=str, default='gpt-4o', help='Model to evaluate (e.g., gpt-4o)')
    args = parser.parse_args()
    model = args.model.replace('-', '_').replace('.', '_')
    
    score_file = f'data/blenderbench/_evaluation/ref_based_overall_scores_{model}.json'
    with open(score_file, 'r') as f:
        scores = json.load(f)
        
    for task_type, task_scores in scores.items():
        if 'num_instances' not in task_scores:
            continue
        missing_rounds = TASK_INSTANCE_COUNT_DICT[task_type] - task_scores['num_instances']
        task_scores['final_n_clip'] = (task_scores['best_n_clip'] * task_scores['num_instances'] + n_clip_penalty * missing_rounds * TASK_SCALE_DICT[task_type]) / TASK_INSTANCE_COUNT_DICT[task_type]
        task_scores['final_pl'] = (task_scores['best_pl'] * task_scores['num_instances'] + pl_penalty * missing_rounds * TASK_SCALE_DICT[task_type]) / TASK_INSTANCE_COUNT_DICT[task_type]
        task_scores['failed_instances'] = missing_rounds
        print(f"Task type: {task_type}, Final n_clip: {task_scores['final_n_clip']}, Final pl: {task_scores['final_pl']}")
        
    with open(score_file, 'w') as f:
        json.dump(scores, f, indent=4)
        
    vlm_score_file = f'data/blenderbench/_evaluation/ref_free_overall_scores_{model}.json'
    with open(vlm_score_file, 'r') as f:
        vlm_scores = json.load(f)
        
    for task_type, task_scores in vlm_scores.items():
        if 'num_instances' not in task_scores:
            continue
        missing_rounds = TASK_INSTANCE_COUNT_DICT[task_type] - task_scores['num_instances']
        task_scores['final_vlm'] = (task_scores['average_best_score'] * task_scores['num_instances'] + vlm_penalty * missing_rounds * TASK_SCALE_DICT[task_type]) / TASK_INSTANCE_COUNT_DICT[task_type]
        task_scores['failed_instances'] = missing_rounds
        print(f"Task type: {task_type}, Final vlm: {task_scores['final_vlm']}")
        
    with open(vlm_score_file, 'w') as f:
        json.dump(vlm_scores, f, indent=4)