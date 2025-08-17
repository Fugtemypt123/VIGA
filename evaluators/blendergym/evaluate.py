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

# Task instance counts for different task types
TASK_INSTANCE_COUNT_DICT = {
    'geometry': 45,
    'material': 40,
    'blendshape': 75,
    'placement': 40,
    'lighting': 40
}

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

    # Load the CLIP model
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Preprocess the images
    images = [image1, image2]
    inputs = processor(images=images, return_tensors="pt")

    # Compute the features for the images
    with torch.no_grad():
        features = model.get_image_features(**inputs)

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

def main():
    parser = argparse.ArgumentParser(description='Evaluate AgenticVerifier blendergym results')
    parser.add_argument('test_id', type=str, help='Test ID (e.g., 20250815_150016)')
    parser.add_argument('--output_dir', type=str, default=None, 
                       help='Output directory for evaluation results (default: output/blendergym/{test_id}/evaluation)')
    
    args = parser.parse_args()
    test_id = args.test_id
    
    # Set up paths
    output_base_dir = f"output/blendergym/{test_id}"
    if not os.path.exists(output_base_dir):
        raise ValueError(f"Output directory {output_base_dir} does not exist.")
    
    if args.output_dir:
        eval_output_dir = args.output_dir
    else:
        eval_output_dir = os.path.join(output_base_dir, "evaluation")
    
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
    
    for task_type, task_instances in tasks_by_type.items():
        print(f"\nProcessing task type: {task_type}")
        
        # Sort by task number
        task_instances.sort(key=lambda x: x[1])
        
        # Initialize scores for this task type
        scores_across_instances = {
            'best_n_clip': [], 
            'best_pl': [],
            'instance_details': {}
        }
        
        for task_dir, task_number in tqdm(task_instances, desc=f"Processing {task_type}"):
            print(f"  Processing {task_dir} (instance {task_number})")
            
            task_instance_dir = os.path.join(output_base_dir, task_dir)
            renders_dir = os.path.join(task_instance_dir, "renders")
            
            if not os.path.exists(renders_dir):
                print(f"    Warning: renders directory not found in {task_dir}")
                continue
            
            # Get ground truth images from data directory
            gt_renders_dir = f"data/blendergym/{task_dir}/renders/goal"
            if not os.path.exists(gt_renders_dir):
                print(f"    Warning: ground truth renders directory not found: {gt_renders_dir}")
                continue
            
            # Store scores for this instance
            task_instance_scores = {}
            
            # Get all round directories (1, 2, 3, 4, etc.)
            round_dirs = [d for d in os.listdir(renders_dir) 
                         if os.path.isdir(os.path.join(renders_dir, d))]
            round_dirs.sort(key=lambda x: int(x))
            
            if not round_dirs:
                print(f"    Warning: no round directories found in {renders_dir}")
                continue
            
            # Process each round
            for round_dir in round_dirs:
                round_path = os.path.join(renders_dir, round_dir)
                task_instance_scores[round_dir] = {}
                
                n_clip_views = []
                pl_views = []
                
                # Check for render1.png in this round
                render1_path = os.path.join(round_path, "render1.png")
                if not os.path.exists(render1_path):
                    print(f"    Warning: render1.png not found in {round_path}")
                    continue
                
                # Check for corresponding ground truth
                gt_render1_path = os.path.join(gt_renders_dir, "render1.png")
                if not os.path.exists(gt_render1_path):
                    print(f"    Warning: ground truth render1.png not found in {gt_renders_dir}")
                    continue
                
                try:
                    # Load images
                    proposal_render = Image.open(render1_path)
                    gt_render = Image.open(gt_render1_path)
                    
                    # Compute scores
                    n_clip = float(1 - clip_similarity(proposal_render, gt_render))
                    pl = float(photometric_loss(proposal_render, gt_render))
                    
                    n_clip_views.append(n_clip)
                    pl_views.append(pl)
                    
                    # Record scores for this render
                    task_instance_scores[round_dir]['render1'] = {
                        'n_clip': n_clip,
                        'pl': pl
                    }
                    
                except Exception as e:
                    print(f"    Error processing {round_path}: {e}")
                    continue
                
                # Check for render2.png if it exists
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
                        
                        task_instance_scores[round_dir]['render2'] = {
                            'n_clip': n_clip2,
                            'pl': pl2
                        }
                        
                    except Exception as e:
                        print(f"    Error processing render2 in {round_path}: {e}")
                        continue
                
                # Compute average scores for this round
                if n_clip_views:
                    avg_n_clip = sum(n_clip_views) / len(n_clip_views)
                    avg_pl = sum(pl_views) / len(pl_views)
                    
                    task_instance_scores[round_dir]['avg_n_clip'] = avg_n_clip
                    task_instance_scores[round_dir]['avg_pl'] = avg_pl
            
            # Find best scores for this instance
            if task_instance_scores:
                # Filter out rounds without scores
                valid_rounds = {k: v for k, v in task_instance_scores.items() 
                              if 'avg_n_clip' in v and 'avg_pl' in v}
                
                if valid_rounds:
                    best_n_clip_round = min(valid_rounds.keys(), 
                                          key=lambda r: valid_rounds[r]['avg_n_clip'])
                    best_pl_round = min(valid_rounds.keys(), 
                                      key=lambda r: valid_rounds[r]['avg_pl'])
                    
                    best_n_clip = valid_rounds[best_n_clip_round]['avg_n_clip']
                    best_pl = valid_rounds[best_pl_round]['avg_pl']
                    
                    # Record best scores
                    task_instance_scores['best_n_clip'] = (best_n_clip_round, best_n_clip)
                    task_instance_scores['best_pl'] = (best_pl_round, best_pl)
                    
                    # Add to overall scores
                    scores_across_instances['best_n_clip'].append(best_n_clip)
                    scores_across_instances['best_pl'].append(best_pl)
                    
                    print(f"    Best n_clip: {best_n_clip:.4f} (round {best_n_clip_round})")
                    print(f"    Best pl: {best_pl:.4f} (round {best_pl_round})")
                else:
                    print(f"    No valid scores found for {task_dir}")
            
            # Save instance scores
            scores_across_instances['instance_details'][task_dir] = task_instance_scores
            
            # Save individual instance scores
            instance_scores_path = os.path.join(task_instance_dir, 'scores.json')
            with open(instance_scores_path, 'w') as f:
                json.dump(task_instance_scores, f, indent=4)
        
        # Compute overall scores for this task type
        if scores_across_instances['best_n_clip']:
            scores_across_tasks[task_type] = {
                'best_n_clip': sum(scores_across_instances['best_n_clip']) / len(scores_across_instances['best_n_clip']),
                'best_pl': sum(scores_across_instances['best_pl']) / len(scores_across_instances['best_pl']),
                'num_instances': len(scores_across_instances['best_n_clip'])
            }
            
            print(f"  Task {task_type} overall scores:")
            print(f"    Average best n_clip: {scores_across_tasks[task_type]['best_n_clip']:.4f}")
            print(f"    Average best pl: {scores_across_tasks[task_type]['best_pl']:.4f}")
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
            print(f"  Instances evaluated: {scores['num_instances']}")
        else:
            print(f"\n{task_type.upper()}: No valid scores")

if __name__ == "__main__":
    main() 