#!/usr/bin/env python3
"""
Evaluation script for AgenticVerifier blendergym results.
Adapted from the original evaluate.py to work with our codebase structure.
"""

import os
import sys
import argparse
import json
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from transformers import CLIPProcessor, CLIPModel
from concurrent.futures import ThreadPoolExecutor, as_completed

# Task instance counts for different task types
TASK_INSTANCE_COUNT_DICT = {
    'geometry': 50,
    'material': 40,
    'blendshape': 75,
    'placement': 40,
    'lighting': 35
}

# Global CLIP model/processor to share across threads
GLOBAL_CLIP_MODEL = None
GLOBAL_CLIP_PROCESSOR = None


def ensure_clip_loaded():
    """
    Lazily load the global CLIP model and processor once per process.
    """
    global GLOBAL_CLIP_MODEL, GLOBAL_CLIP_PROCESSOR
    if GLOBAL_CLIP_MODEL is None or GLOBAL_CLIP_PROCESSOR is None:
        GLOBAL_CLIP_MODEL = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        GLOBAL_CLIP_PROCESSOR = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


def clip_similarity(image1, image2):
    """
    Compute the CLIP similarity between two PIL images.

    Args:
    image1 (PIL.Image): The first input image.
    image2 (PIL.Image): The second input image.

    Returns:
    float: The CLIP similarity between the two images.
    """
    if image1.size != image2.size:
        image2 = image2.resize(image1.size)

    # Ensure global model is initialized
    ensure_clip_loaded()

    # Preprocess the images
    images = [image1, image2]
    inputs = GLOBAL_CLIP_PROCESSOR(images=images, return_tensors="pt")

    # Compute the features for the images
    with torch.no_grad():
        features = GLOBAL_CLIP_MODEL.get_image_features(**inputs)

    # Compute the cosine similarity between the image features
    sim = torch.nn.functional.cosine_similarity(features[0], features[1], dim=-1)

    return sim.item()

def photometric_loss(image1: Image.Image, image2: Image.Image) -> float:
    """
    Compute the photometric loss between two PIL images.

    Args:
    image1 (PIL.Image): The first input image.
    image2 (PIL.Image): The second input image.

    Returns:
    float: The photometric loss between the two images.
    """
    if image1.size != image2.size:
        image2 = image2.resize(image1.size)
    
    # Convert images to numpy arrays
    img1_array = np.array(image1)[:, :, :3]
    img2_array = np.array(image2)[:, :, :3]

    # Normalize images to [0, 1]
    img1_norm = img1_array.astype(np.float32) / 255.0
    img2_norm = img2_array.astype(np.float32) / 255.0

    # Compute the squared difference between the normalized images
    diff = np.square(img1_norm - img2_norm)

    # Compute the mean squared error
    mse = np.mean(diff)
    return mse


def process_task_instance(output_base_dir: str, task_dir: str):
    """
    Process a single task instance directory and compute metrics across rounds.

    Returns:
        tuple: (task_dir, task_instance_scores, best_n_clip, best_pl)
               where best_* can be None if no valid rounds.
    """
    task_instance_dir = os.path.join(output_base_dir, task_dir)
    
    exist_score = False
    if os.path.exists(os.path.join(task_instance_dir, "scores.json")):
        
        with open(os.path.join(task_instance_dir, "scores.json"), 'r') as f:
            task_instance_scores = json.load(f)
            for key, score in task_instance_scores.items():
                if score != {}:
                    exist_score = True
                    break
    if not exist_score:
        renders_dir = os.path.join(task_instance_dir, "renders")

        if not os.path.exists(renders_dir):
            return task_dir, {}, None, None

        gt_renders_dir = f"data/blendergym/{task_dir}/renders/goal"
        if not os.path.exists(gt_renders_dir):
            return task_dir, {}, None, None

        task_instance_scores = {}

        # Get all round directories (1, 2, 3, 4, etc.)
        round_dirs = [d for d in os.listdir(renders_dir)
                    if os.path.isdir(os.path.join(renders_dir, d))]
        try:
            round_dirs.sort(key=lambda x: int(x))
        except Exception:
            # Fallback to lexical sort if non-numeric dirs exist
            round_dirs.sort()

        if not round_dirs:
            return task_dir, {}, None, None

        for round_dir in round_dirs:
            round_path = os.path.join(renders_dir, round_dir)
            task_instance_scores[round_dir] = {}

            n_clip_views = []
            pl_views = []

            # render1
            render1_path = os.path.join(round_path, "render1.png")
            gt_render1_path = os.path.join(gt_renders_dir, "render1.png")
            if os.path.exists(render1_path) and os.path.exists(gt_render1_path):
                try:
                    proposal_render = Image.open(render1_path)
                    gt_render = Image.open(gt_render1_path)
                    n_clip = float(1 - clip_similarity(proposal_render, gt_render))
                    pl = float(photometric_loss(proposal_render, gt_render))
                    n_clip_views.append(n_clip)
                    pl_views.append(pl)
                    task_instance_scores[round_dir]['render1'] = {'n_clip': n_clip, 'pl': pl}
                except Exception:
                    pass

            # render2
            render2_path = os.path.join(round_path, "render2.png")
            gt_render2_path = os.path.join(gt_renders_dir, "render2.png")
            if os.path.exists(render2_path) and os.path.exists(gt_render2_path):
                try:
                    proposal_render2 = Image.open(render2_path)
                    gt_render2 = Image.open(gt_render2_path)
                    n_clip2 = float(1 - clip_similarity(proposal_render2, gt_render2))
                    pl2 = float(photometric_loss(proposal_render2, gt_render2))
                    n_clip_views.append(n_clip2)
                    pl_views.append(pl2)
                    task_instance_scores[round_dir]['render2'] = {'n_clip': n_clip2, 'pl': pl2}
                except Exception:
                    pass

            if n_clip_views:
                avg_n_clip = sum(n_clip_views) / len(n_clip_views)
                avg_pl = sum(pl_views) / len(pl_views)
                task_instance_scores[round_dir]['avg_n_clip'] = avg_n_clip
                task_instance_scores[round_dir]['avg_pl'] = avg_pl

    # Determine best rounds
    valid_rounds = {k: v for k, v in task_instance_scores.items() if 'avg_n_clip' in v and 'avg_pl' in v}
    best_n_clip = None
    best_pl = None
    if valid_rounds:
        best_n_clip_round = min(valid_rounds.keys(), key=lambda r: valid_rounds[r]['avg_n_clip'])
        best_n_clip = valid_rounds[best_n_clip_round]['avg_n_clip']
        best_pl = valid_rounds[best_n_clip_round]['avg_pl']
        
    # select last round results
    last_round_n_clip = None
    last_round_pl = None
    for round in valid_rounds.keys():
        if round == str(max(valid_rounds.keys())):
            last_round_n_clip = valid_rounds[round]['avg_n_clip']
            last_round_pl = valid_rounds[round]['avg_pl']
            break

    # Save individual instance scores
    instance_scores_path = os.path.join(task_instance_dir, 'scores.json')
    try:
        with open(instance_scores_path, 'w') as f:
            json.dump(task_instance_scores, f, indent=4)
    except Exception:
        pass

    return task_dir, task_instance_scores, best_n_clip, best_pl, last_round_n_clip, last_round_pl

def extract_task_type_and_number(task_dir_name):
    """
    Extract task type and number from directory name like 'placement1', 'material5', etc.
    
    Args:
        task_dir_name (str): Directory name like 'placement1', 'material5'
        
    Returns:
        tuple: (task_type, task_number) or (None, None) if invalid
    """
    for task_type in TASK_INSTANCE_COUNT_DICT.keys():
        if task_dir_name.startswith(task_type):
            try:
                task_number = int(task_dir_name[len(task_type):])
                return task_type, task_number
            except ValueError:
                continue
    return None, None

def main() -> None:
    """Run evaluation for BlenderGym results."""
    parser = argparse.ArgumentParser(description='Evaluate AgenticVerifier blendergym results')
    parser.add_argument('test_id', type=str, help='Test ID (e.g., 20250815_150016)')
    parser.add_argument('--output_dir', type=str, default=None, 
                       help='Output directory for evaluation results (default: output/blendergym/{test_id}/)_evaluation)')
    
    args = parser.parse_args()
    test_id = args.test_id
    MAX_ROUNDS = 10
    
    # Set up paths
    output_base_dir = f"output/blendergym/{test_id}"
    if not os.path.exists(output_base_dir):
        raise ValueError(f"Output directory {output_base_dir} does not exist.")
    
    if args.output_dir:
        eval_output_dir = args.output_dir
    else:
        eval_output_dir = os.path.join(output_base_dir, "_evaluation")
    
    os.makedirs(eval_output_dir, exist_ok=True)
    
    # Get all task directories
    task_dirs = [d for d in os.listdir(output_base_dir) 
                if os.path.isdir(os.path.join(output_base_dir, d)) and d != "evaluation"]
    
    print(f"Found {len(task_dirs)} task directories in {output_base_dir}")
    
    # Group tasks by type
    tasks_by_type = {}
    for task_dir in task_dirs:
        task_type, task_number = extract_task_type_and_number(task_dir)
        if task_type and task_number:
            if task_type not in tasks_by_type:
                tasks_by_type[task_type] = []
            tasks_by_type[task_type].append((task_dir, task_number))
    
    print(f"Grouped tasks by type: {list(tasks_by_type.keys())}")
    
    scores_across_tasks = {}
    intermediates = {}
    
    # Ensure CLIP is loaded once (shared by threads)
    ensure_clip_loaded()

    for task_type, task_instances in tasks_by_type.items():
        print(f"\nProcessing task type: {task_type}")

        # Sort by task number
        task_instances.sort(key=lambda x: x[1])

        scores_across_instances = {
            'best_n_clip': [],
            'best_pl': [],
            'last_round_n_clip': [],
            'last_round_pl': [],
            'instance_details': {}
        }

        # Run per-instance processing in parallel threads
        max_workers = min(8, (os.cpu_count() or 4))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(process_task_instance, output_base_dir, task_dir)
                for task_dir, _ in task_instances
            ]

            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {task_type}"):
                try:
                    task_dir, task_instance_scores, best_n_clip, best_pl, last_round_n_clip, last_round_pl = future.result()
                    scores_across_instances['instance_details'][task_dir] = task_instance_scores
                    if best_n_clip is not None and best_pl is not None:
                        scores_across_instances['best_n_clip'].append(best_n_clip)
                        scores_across_instances['best_pl'].append(best_pl)
                        scores_across_instances['last_round_n_clip'].append(last_round_n_clip)
                        scores_across_instances['last_round_pl'].append(last_round_pl)
                        print(f"    {task_dir}: Best n_clip={best_n_clip:.4f}, Best pl={best_pl:.4f}")
                    else:
                        print(f"    {task_dir}: No valid scores")
                except Exception as e:
                    print(f"    Error processing {task_type} instance: {e}")

        # Aggregate per-round averages across all instances (rounds 1..10)
        per_round_values = {str(i): {'n_clip': [], 'pl': []} for i in range(1, 11)}
        for instance_scores in scores_across_instances['instance_details'].values():
            for round_idx in range(1, 11):
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

        # Store per-round aggregation in intermediates structure too
        scores_across_instances['per_round'] = per_round_summary

        # Aggregate results for this task type
        if scores_across_instances['best_n_clip']:
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

        intermediates[task_type] = scores_across_instances
    
    # Save overall results
    overall_scores_path = os.path.join(eval_output_dir, 'overall_scores.json')
    with open(overall_scores_path, 'w') as f:
        json.dump(scores_across_tasks, f, indent=4)
    
    intermediate_scores_path = os.path.join(eval_output_dir, 'intermediate_scores.json')
    with open(intermediate_scores_path, 'w') as f:
        json.dump(intermediates, f, indent=4)
    
    # Print summary
    print(f"\n=== Evaluation Summary ===")
    print(f"Test ID: {test_id}")
    print(f"Output directory: {eval_output_dir}")
    print(f"Results saved to: {overall_scores_path}")
    
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