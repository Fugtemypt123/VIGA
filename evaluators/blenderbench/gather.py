"""Gather and aggregate BlenderBench evaluation results."""
import argparse
import json

TASK_INSTANCE_COUNT_DICT = {
    'level1': 9,
    'level2': 9,
    'level3': 9,
}

TASK_SCALE_DICT = {
    'level1': 1.0,
    'level2': 0.5,
    'level3': 1.0,
}

n_clip_penalty = 1.0
pl_penalty = 0.8
vlm_penalty = 0.0


def _get_best_clip_round(instance_rounds: dict):
    best_round = None
    best_clip = None
    best_pl = None

    for round_id, round_scores in instance_rounds.items():
        if not isinstance(round_scores, dict):
            continue
        if 'avg_n_clip' not in round_scores or 'avg_pl' not in round_scores:
            continue
        clip_score = round_scores['avg_n_clip']
        if best_clip is None or clip_score < best_clip:
            best_clip = clip_score
            best_pl = round_scores['avg_pl']
            best_round = round_id

    return best_round, best_clip, best_pl


def _extract_vlm_for_round(instance_rounds: dict, round_id: str):
    """
    Return (round_scores, vlm_score) for the requested round.
    """
    round_scores = instance_rounds.get(round_id, {})
    vlm_score = None
    if isinstance(round_scores, dict):
        if isinstance(round_scores.get('round_average'), (int, float)):
            vlm_score = round_scores['round_average']
        else:
            averages = []
            for render_scores in round_scores.values():
                if isinstance(render_scores, dict) and isinstance(render_scores.get('average_score'), (int, float)):
                    averages.append(render_scores['average_score'])
            if averages:
                vlm_score = sum(averages) / len(averages)

    if vlm_score is None:
        vlm_score = 0.0

    return round_scores, vlm_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Gather baseline BlenderStudio results')
    parser.add_argument('test_id', type=str, help='Test ID (e.g., 20250815_150016)')
    args = parser.parse_args()

    eval_dir = f'output/blenderbench/{args.test_id}/_evaluation'

    score_file = f'{eval_dir}/overall_scores.json'
    with open(score_file, 'r') as f:
        scores = json.load(f)

    intermediate_file = f'{eval_dir}/intermediate_scores.json'
    with open(intermediate_file, 'r') as f:
        intermediate_scores = json.load(f)

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

    vlm_score_file = f'{eval_dir}/reference_free_overall_scores.json'
    with open(vlm_score_file, 'r') as f:
        vlm_scores = json.load(f)

    ref_free_intermediate_file = f'{eval_dir}/reference_free_intermediate_scores.json'
    with open(ref_free_intermediate_file, 'r') as f:
        ref_free_intermediate_scores = json.load(f)

    for task_type, task_scores in vlm_scores.items():
        if 'num_instances' not in task_scores:
            continue

        instance_rounds = intermediate_scores.get(task_type, {}).get('instance_details', {})
        ref_free_instance_rounds = ref_free_intermediate_scores.get(task_type, {}).get('instance_details', {})

        recomputed_best_scores = []
        vlm_sum = 0.0

        for instance_name, rounds in instance_rounds.items():
            best_round, best_clip, best_pl = _get_best_clip_round(rounds)
            if best_round is None:
                continue

            ref_free_rounds = ref_free_instance_rounds.get(instance_name, {})
            round_scores, vlm_score = _extract_vlm_for_round(ref_free_rounds, best_round)

            recomputed_best_scores.append({
                'best_round': str(best_round),
                'best_score': {
                    'clip': best_clip,
                    'pl': best_pl,
                    'vlm_score': vlm_score
                },
                'best_round_scores': round_scores
            })
            vlm_sum += vlm_score

        if recomputed_best_scores:
            task_scores['best_scores'] = recomputed_best_scores
            task_scores['average_best_score'] = vlm_sum / len(recomputed_best_scores)
            task_scores['num_instances'] = len(recomputed_best_scores)
        else:
            task_scores['best_scores'] = []
            task_scores['average_best_score'] = 0.0
            task_scores['num_instances'] = 0

    for task_type, task_scores in vlm_scores.items():
        if 'num_instances' not in task_scores:
            continue
        missing_rounds = TASK_INSTANCE_COUNT_DICT[task_type] - task_scores['num_instances']
        task_scores['final_vlm'] = (task_scores['average_best_score'] * task_scores['num_instances'] + vlm_penalty * missing_rounds * TASK_SCALE_DICT[task_type]) / TASK_INSTANCE_COUNT_DICT[task_type]
        task_scores['failed_instances'] = missing_rounds
        print(f"Task type: {task_type}, Final vlm: {task_scores['final_vlm']}")

    with open(vlm_score_file, 'w') as f:
        json.dump(vlm_scores, f, indent=4)