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


def normalize_model_name(model_name: str) -> str:
    """
    Normalize model name by replacing spaces and dots with underscores.
    
    Args:
        model_name: Original model name (e.g., 'gpt-4o', 'gpt 4o')
        
    Returns:
        Normalized model name (e.g., 'gpt_4o')
    """
    return model_name.replace('-', '_').replace('.', '_')


def process_task_instance(task_dir: str, model_name: str):
    """
    Process a single task instance directory for baseline format.
    Baseline format: data/blendergym/{task_dir}/baseline/{model_name}/
    
    Returns:
        tuple: (task_dir, n_clip, pl) where n_clip and pl can be None if no valid renders.
    """
    # Normalize model name
    normalized_model = normalize_model_name(model_name)
    
    # Baseline path: data/blendergym/{task_dir}/baseline/{model_name}/
    baseline_dir = os.path.join("data/blendergym", task_dir, "baseline", normalized_model)
    
    if not os.path.exists(baseline_dir):
        return task_dir, None, None

    gt_renders_dir = f"data/blendergym/{task_dir}/renders/goal"
    if not os.path.exists(gt_renders_dir):
        return task_dir, None, None

    n_clip_views = []
    pl_views = []

    # render1
    render1_path = os.path.join(baseline_dir, "render1.png")
    gt_render1_path = os.path.join(gt_renders_dir, "render1.png")
    if os.path.exists(render1_path) and os.path.exists(gt_render1_path):
        try:
            proposal_render = Image.open(render1_path)
            gt_render = Image.open(gt_render1_path)
            n_clip = float(1 - clip_similarity(proposal_render, gt_render))
            pl = float(photometric_loss(proposal_render, gt_render))
            n_clip_views.append(n_clip)
            pl_views.append(pl)
        except Exception:
            pass

    # render2
    render2_path = os.path.join(baseline_dir, "render2.png")
    gt_render2_path = os.path.join(gt_renders_dir, "render2.png")
    if os.path.exists(render2_path) and os.path.exists(gt_render2_path):
        try:
            proposal_render2 = Image.open(render2_path)
            gt_render2 = Image.open(gt_render2_path)
            n_clip2 = float(1 - clip_similarity(proposal_render2, gt_render2))
            pl2 = float(photometric_loss(proposal_render2, gt_render2))
            n_clip_views.append(n_clip2)
            pl_views.append(pl2)
        except Exception:
            pass

    # Compute average scores
    if n_clip_views:
        avg_n_clip = sum(n_clip_views) / len(n_clip_views)
        avg_pl = sum(pl_views) / len(pl_views)
        return task_dir, avg_n_clip, avg_pl
    else:
        return task_dir, None, None

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
    """Run evaluation for BlenderGym baseline results."""
    parser = argparse.ArgumentParser(description='Evaluate baseline blendergym results')
    parser.add_argument('--model', type=str, default='gpt-4o', help='Model to evaluate (e.g., gpt-4o)')
    args = parser.parse_args()
    model = args.model
    
    # Set up paths
    output_base_dir = f"data/blendergym"
    if not os.path.exists(output_base_dir):
        raise ValueError(f"Output directory {output_base_dir} does not exist.")
    
    # Get all task directories
    task_dirs = [d for d in os.listdir(output_base_dir) 
                if os.path.isdir(os.path.join(output_base_dir, d)) and d != "blender_files"]
    
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
    
    # Ensure CLIP is loaded once (shared by threads)
    ensure_clip_loaded()

    for task_type, task_instances in tasks_by_type.items():
        print(f"\nProcessing task type: {task_type}")

        # Sort by task number
        task_instances.sort(key=lambda x: x[1])

        n_clip_scores = []
        pl_scores = []

        # Run per-instance processing in parallel threads
        max_workers = min(8, (os.cpu_count() or 4))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(process_task_instance, task_dir, model)
                for task_dir, _ in task_instances
            ]

            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {task_type}"):
                try:
                    task_dir, n_clip, pl = future.result()
                    if n_clip is not None and pl is not None:
                        n_clip_scores.append(n_clip)
                        pl_scores.append(pl)
                        print(f"    {task_dir}: n_clip={n_clip:.4f}, pl={pl:.4f}")
                    else:
                        print(f"    {task_dir}: No valid scores")
                except Exception as e:
                    print(f"    Error processing {task_type} instance: {e}")

        # Aggregate results for this task type (baseline has only one round)
        if n_clip_scores and pl_scores:
            scores_across_tasks[task_type] = {
                'avg_n_clip': sum(n_clip_scores) / len(n_clip_scores),
                'avg_pl': sum(pl_scores) / len(pl_scores),
                'num_instances': len(n_clip_scores)
            }

            print(f"  Task {task_type} overall scores:")
            print(f"    Average n_clip: {scores_across_tasks[task_type]['avg_n_clip']:.4f}")
            print(f"    Average pl: {scores_across_tasks[task_type]['avg_pl']:.4f}")
            print(f"    Number of instances: {scores_across_tasks[task_type]['num_instances']}")
        else:
            print(f"  No valid scores for task type {task_type}")
            scores_across_tasks[task_type] = {}
    
    # Save overall results
    normalized_model = normalize_model_name(model)
    eval_output_dir = os.path.join(output_base_dir, "baseline", normalized_model, "_evaluation")
    os.makedirs(eval_output_dir, exist_ok=True)
    
    overall_scores_path = os.path.join(eval_output_dir, 'overall_scores.json')
    with open(overall_scores_path, 'w') as f:
        json.dump(scores_across_tasks, f, indent=4)
    
    # Print summary
    print(f"\n=== Evaluation Summary ===")
    print(f"Model: {model}")
    print(f"Output directory: {eval_output_dir}")
    print(f"Results saved to: {overall_scores_path}")
    
    for task_type, scores in scores_across_tasks.items():
        if scores:
            print(f"\n{task_type.upper()}:")
            print(f"  Average n_clip: {scores['avg_n_clip']:.4f}")
            print(f"  Average pl: {scores['avg_pl']:.4f}")
            print(f"  Instances evaluated: {scores['num_instances']}")
        else:
            print(f"\n{task_type.upper()}: No valid scores")

if __name__ == "__main__":
    main() 