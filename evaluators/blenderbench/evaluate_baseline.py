#!/usr/bin/env python3
"""
Evaluation script for AgenticVerifier blenderbench baseline results.
Combines reference-based (CLIP + photometric loss) and reference-free (GPT) evaluation methods.
Baseline renders are located at: data/blenderbench/{level}/{task}/baseline/{model_name}/render1.png
"""

import argparse
import base64
import json
import os
import re
import sys

from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils._api_keys import OPENAI_API_KEY

# Task instance counts for different task types
TASK_INSTANCE_COUNT_DICT = {
    'level1': 9,
    'level2': 9,
    'level3': 9,
}

# Global CLIP model/processor to share across threads
GLOBAL_CLIP_MODEL = None
GLOBAL_CLIP_PROCESSOR = None

# Evaluation criteria for reference-free evaluation
EVALUATION_CRITERIA = {
    "task_completion": {
        "criteria": "Evaluate how well the image demonstrates completion of the specified task. Consider whether the required objects are in the correct positions, orientations, and states as described in the task.",
        "scale": 5
    },
    "visual_quality": {
        "criteria": "Assess the overall visual quality of the image including lighting, shadows, reflections, and rendering quality. Consider if the scene looks realistic and well-lit.",
        "scale": 5
    },
    "spatial_accuracy": {
        "criteria": "Evaluate the spatial relationships between objects, their positioning relative to each other and the environment, and whether they follow the spatial requirements mentioned in the task.",
        "scale": 5
    },
    "detail_accuracy": {
        "criteria": "Assess the accuracy of object details, materials, textures, and any specific visual characteristics mentioned in the task description.",
        "scale": 5
    }
}

INSTRUCTION_TEMPLATE = """You are evaluating a 3D scene image generated based on a specific task description. You will be shown a TARGET image (may be style-transferred/noisy) and a GENERATED image. Compare them with respect to the task, prioritizing geometric structure, spatial layout, object identity/pose/placement, and lighting intent over stylistic differences (colors, textures, style effects).

{criteria}

Task Description: {task_description}

Instructions:
- Use the TARGET image as a visual reference for intended scene layout and camera/view configuration.
- Ignore style-transfer artifacts (e.g., color grading, texture stylization, artistic filters) when they do not affect task objectives.
- Focus your judgment on whether the GENERATED image meets the task criteria relative to the TARGET reference.

Give an integer score between 0 - {scale}, where higher scores mean the criteria is better met.
First, respond with a score; Then, provide your justification for the score in natural language sentences. Your response should look like this: '4. The cabinet is correctly positioned and the plant is visible in the mirror as required.'
Only evaluate the image based on the specified criteria, and no other aspects. Give scores across the full spectrum (0-5) instead of only good ones (3-5).
"""


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


def encode_image(image_path):
    """Encode image to base64 for GPT API."""
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def evaluate_image_with_gpt(image_path: str, target_image_path: str, task_description: str, 
                            criteria_name: str, criteria_dict: dict, model_name: str = "gpt-4o", 
                            client: OpenAI = None):
    """
    Evaluate a single image using GPT based on specific criteria.
    
    Args:
        image_path: Path to the image to evaluate
        target_image_path: Path to the target image
        task_description: Text description of the task
        criteria_name: Name of the evaluation criteria
        criteria_dict: Dictionary containing criteria and scale
        model_name: GPT model to use
        client: OpenAI client instance
        
    Returns:
        dict: Evaluation result with score and justification
    """
    if client is None:
        client = OpenAI(api_key=OPENAI_API_KEY)
    
    try:
        # Encode image
        image_base64 = encode_image(image_path)
        target_image_base64 = encode_image(target_image_path)
        image_url = f"data:image/png;base64,{image_base64}"
        target_image_url = f"data:image/png;base64,{target_image_base64}"
        
        # Create prompt
        prompt = INSTRUCTION_TEMPLATE.format(
            criteria=criteria_dict["criteria"],
            task_description=task_description,
            scale=criteria_dict["scale"]
        )
        
        # Create messages for GPT
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "text", "text": "Target image:"},
                {"type": "image_url", "image_url": {"url": target_image_url}},
                {"type": "text", "text": "Generated image:"},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        }]
             
        # Get response from GPT
        response = client.chat.completions.create(model=model_name, messages=messages)
        response_text = response.choices[0].message.content.strip()
        
        # Parse response
        try:
            score, justification = response_text.split(".", 1)
            score = float(score.strip())
        except (ValueError, AttributeError):
            score, justification = 0.0, response_text

        return {
            "score": score,
            "justification": justification.strip(),
            "criteria": criteria_name
        }

    except Exception as e:
        print(f"Error evaluating image {image_path}: {e}")
        return {
            "score": 0.0,
            "justification": f"Error during evaluation: {str(e)}",
            "criteria": criteria_name
        }


def load_task_description(task_dir: str) -> str:
    """
    Load task description from task.txt file.
    
    Args:
        task_dir: Directory containing the task
        
    Returns:
        str: Task description text
    """
    task_txt_path = os.path.join(task_dir, 'task.txt')
    if os.path.exists(task_txt_path):
        with open(task_txt_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    return ""


def process_task_instance_ref_based(data_dir: str, task_dir: str, model_name: str):
    """
    Process a single task instance for reference-based evaluation (CLIP + photometric loss).

    Returns:
        tuple: (task_dir, scores_dict) where scores_dict contains n_clip and pl
    """
    baseline_render_path = os.path.join(data_dir, task_dir, "baseline", model_name, "render1.png")
    gt_renders_dir = os.path.join(data_dir, task_dir, "renders", "goal")
    gt_render_path = os.path.join(gt_renders_dir, "render1.png")

    if not os.path.exists(baseline_render_path):
        return task_dir, None

    if not os.path.exists(gt_render_path):
        return task_dir, None

    try:
        baseline_render = Image.open(baseline_render_path)
        gt_render = Image.open(gt_render_path)
        
        n_clip = float(1 - clip_similarity(baseline_render, gt_render))
        pl = float(photometric_loss(baseline_render, gt_render))
        
        return task_dir, {
            'render1': {
                'n_clip': n_clip,
                'pl': pl
            },
            'avg_n_clip': n_clip,
            'avg_pl': pl
        }
    except Exception as e:
        print(f"Error processing {task_dir} for ref-based eval: {e}")
        return task_dir, None


def process_task_instance_ref_free(data_dir: str, task_dir: str, model_name: str, 
                                   gpt_model_name: str = "gpt-4o", client: OpenAI = None):
    """
    Process a single task instance for reference-free evaluation (GPT).

    Returns:
        tuple: (task_dir, scores_dict)
    """
    baseline_render_path = os.path.join(data_dir, task_dir, "baseline", model_name, "render1.png")
    gt_renders_dir = os.path.join(data_dir, task_dir, "renders", "goal")

    if not os.path.exists(baseline_render_path):
        return task_dir, None

    # Load task description
    task_description = load_task_description(os.path.join(data_dir, task_dir))
    if not task_description:
        task_description = "Task description not available"

    # Determine target image path based on level
    # For level1, try style1.png first, then _style1.png
    # For other levels, use visprompt1.png
    if 'level1' in task_dir:
        target_image_name = 'style1.png'
        target_image_path = os.path.join(gt_renders_dir, target_image_name)
        # Check for _style1.png (with underscore) as fallback
        if not os.path.exists(target_image_path):
            target_image_name = '_style1.png'
            target_image_path = os.path.join(gt_renders_dir, target_image_name)
    else:
        target_image_name = 'visprompt1.png'
        target_image_path = os.path.join(gt_renders_dir, target_image_name)

    if not os.path.exists(target_image_path):
        return task_dir, None

    try:
        render_scores = {}
        
        # Evaluate with GPT for each criteria
        for criteria_name, criteria_dict in EVALUATION_CRITERIA.items():
            gpt_result = evaluate_image_with_gpt(
                baseline_render_path, 
                target_image_path,
                task_description, 
                criteria_name, 
                criteria_dict,
                gpt_model_name,
                client
            )
            render_scores[criteria_name] = gpt_result
        
        # Calculate average score across all criteria
        scores = [gpt_result["score"] for gpt_result in render_scores.values()]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        
        render_scores["average_score"] = avg_score
        
        return task_dir, {
            'render1': render_scores,
            'round_average': avg_score,
            'average_score': avg_score
        }
    except Exception as e:
        print(f"Error processing {task_dir} for ref-free eval: {e}")
        return task_dir, None


def extract_task_type_and_number(task_dir_name):
    """
    Extract task type and number from directory name like 'level1/camera1', 'level2/attribute5', etc.
    
    Args:
        task_dir_name (str): Directory name like 'level1/camera1', 'level2/attribute5'
        
    Returns:
        tuple: (task_type, task_number) or (None, None) if invalid
    """
    parts = task_dir_name.split("/")
    if len(parts) < 2:
        return None, None
    
    level_dir_name = parts[0]
    task_name = parts[1]
    
    for task_type in TASK_INSTANCE_COUNT_DICT.keys():
        if level_dir_name == task_type:
            # Extract number from task name (camera1 -> 1, attribute5 -> 5)
            try:
                # Find the last digit sequence in the task name
                numbers = re.findall(r'\d+', task_name)
                if numbers:
                    task_number = int(numbers[-1])
                    return task_type, task_number
            except ValueError:
                continue
    return None, None


def main() -> None:
    """Run evaluation for BlenderBench baseline results."""
    parser = argparse.ArgumentParser(description='Evaluate AgenticVerifier blenderbench baseline results')
    parser.add_argument('--data_dir', type=str, default='data/blenderbench',
                       help='Base directory containing baseline results (default: data/blenderbench)')
    parser.add_argument('--model_name', type=str, default='gpt_4o',
                       help='Model name in baseline directory (default: gpt_4o)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for evaluation results (default: data/blenderbench/_evaluation)')
    parser.add_argument('--eval_type', type=str, choices=['ref_based', 'ref_free', 'both'], default='both',
                       help='Evaluation type: ref_based (CLIP+PL), ref_free (GPT), or both (default: both)')
    parser.add_argument('--gpt_model', type=str, default='gpt-4o',
                       help='GPT model to use for reference-free evaluation (default: gpt-4o)')
    parser.add_argument('--max_workers', type=int, default=8,
                       help='Maximum number of parallel workers')
    
    args = parser.parse_args()
    
    data_dir = args.data_dir
    model_name = args.model_name.replace('-', '_').replace('.', '_')
    
    if not os.path.exists(data_dir):
        raise ValueError(f"Data directory {data_dir} does not exist.")
    
    if args.output_dir:
        eval_output_dir = args.output_dir
    else:
        eval_output_dir = os.path.join(data_dir, "_evaluation")
    
    os.makedirs(eval_output_dir, exist_ok=True)
    
    # Get all task directories
    task_dirs = []
    for level_dir in os.listdir(data_dir):
        level_path = os.path.join(data_dir, level_dir)
        if os.path.isdir(level_path) and level_dir.startswith('level'):
            for task_dir in os.listdir(level_path):
                task_path = os.path.join(level_path, task_dir)
                if os.path.isdir(task_path):
                    baseline_path = os.path.join(task_path, "baseline", model_name)
                    if os.path.exists(baseline_path):
                        task_dirs.append(os.path.join(level_dir, task_dir))
    
    print(f"Found {len(task_dirs)} task directories with baseline results for model {model_name}")
    
    # Group tasks by type
    tasks_by_type = {}
    for task_dir in task_dirs:
        task_type, task_number = extract_task_type_and_number(task_dir)
        if task_type and task_number:
            if task_type not in tasks_by_type:
                tasks_by_type[task_type] = []
            tasks_by_type[task_type].append((task_dir, task_number))
    
    print(f"Grouped tasks by type: {list(tasks_by_type.keys())}")
    
    # Initialize OpenAI client if needed
    client = None
    if args.eval_type in ['ref_free', 'both']:
        client = OpenAI(api_key=OPENAI_API_KEY)
    
    # Ensure CLIP is loaded if needed
    if args.eval_type in ['ref_based', 'both']:
        ensure_clip_loaded()
    
    ref_based_scores = {}
    ref_free_scores = {}
    ref_based_intermediates = {}
    ref_free_intermediates = {}
    
    for task_type, task_instances in tasks_by_type.items():
        print(f"\nProcessing task type: {task_type}")
        
        # Sort by task number
        task_instances.sort(key=lambda x: x[1])
        
        if args.eval_type in ['ref_based', 'both']:
            ref_based_instance_details = {}
            ref_based_best_scores = {'n_clip': [], 'pl': []}
        
        if args.eval_type in ['ref_free', 'both']:
            ref_free_instance_details = {}
            ref_free_best_scores = []
        
        # Run per-instance processing in parallel threads
        max_workers = min(args.max_workers, (os.cpu_count() or 4))
        
        # Process ref-based evaluation
        if args.eval_type in ['ref_based', 'both']:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(process_task_instance_ref_based, data_dir, task_dir, model_name)
                    for task_dir, _ in task_instances
                ]
                
                for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {task_type} (ref-based)"):
                    try:
                        task_dir, scores = future.result()
                        if scores:
                            ref_based_instance_details[task_dir] = scores
                            ref_based_best_scores['n_clip'].append(scores['avg_n_clip'])
                            ref_based_best_scores['pl'].append(scores['avg_pl'])
                            print(f"    {task_dir}: n_clip={scores['avg_n_clip']:.4f}, pl={scores['avg_pl']:.4f}")
                        else:
                            print(f"    {task_dir}: No valid ref-based scores")
                    except Exception as e:
                        print(f"    Error processing {task_type} instance (ref-based): {e}")
        
        # Process ref-free evaluation
        if args.eval_type in ['ref_free', 'both']:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(process_task_instance_ref_free, data_dir, task_dir, model_name, args.gpt_model, client)
                    for task_dir, _ in task_instances
                ]
                
                for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {task_type} (ref-free)"):
                    try:
                        task_dir, scores = future.result()
                        if scores:
                            ref_free_instance_details[task_dir] = scores
                            ref_free_best_scores.append(scores['average_score'])
                            print(f"    {task_dir}: ref-free score={scores['average_score']:.4f}")
                        else:
                            print(f"    {task_dir}: No valid ref-free scores")
                    except Exception as e:
                        print(f"    Error processing {task_type} instance (ref-free): {e}")
        
        # Aggregate results for this task type
        if args.eval_type in ['ref_based', 'both']:
            if ref_based_best_scores['n_clip']:
                ref_based_scores[task_type] = {
                    'best_n_clip': sum(ref_based_best_scores['n_clip']) / len(ref_based_best_scores['n_clip']),
                    'best_pl': sum(ref_based_best_scores['pl']) / len(ref_based_best_scores['pl']),
                    'num_instances': len(ref_based_best_scores['n_clip'])
                }
                print(f"  Task {task_type} ref-based overall scores:")
                print(f"    Average best n_clip: {ref_based_scores[task_type]['best_n_clip']:.4f}")
                print(f"    Average best pl: {ref_based_scores[task_type]['best_pl']:.4f}")
                print(f"    Number of instances: {ref_based_scores[task_type]['num_instances']}")
            else:
                print(f"  No valid ref-based scores for task type {task_type}")
                ref_based_scores[task_type] = {}
            
            ref_based_intermediates[task_type] = {
                'instance_details': ref_based_instance_details,
                'best_scores': ref_based_best_scores
            }
        
        if args.eval_type in ['ref_free', 'both']:
            if ref_free_best_scores:
                ref_free_scores[task_type] = {
                    'average_best_score': sum(ref_free_best_scores) / len(ref_free_best_scores),
                    'num_instances': len(ref_free_best_scores)
                }
                print(f"  Task {task_type} ref-free overall scores:")
                print(f"    Average best score: {ref_free_scores[task_type]['average_best_score']:.4f}")
                print(f"    Number of instances: {ref_free_scores[task_type]['num_instances']}")
            else:
                print(f"  No valid ref-free scores for task type {task_type}")
                ref_free_scores[task_type] = {}
            
            ref_free_intermediates[task_type] = {
                'instance_details': ref_free_instance_details,
                'best_scores': ref_free_best_scores
            }
    
    # Save results
    if args.eval_type in ['ref_based', 'both']:
        ref_based_overall_path = os.path.join(eval_output_dir, f'ref_based_overall_scores_{model_name}.json')
        with open(ref_based_overall_path, 'w') as f:
            json.dump(ref_based_scores, f, indent=4)
        
        ref_based_intermediate_path = os.path.join(eval_output_dir, f'ref_based_intermediate_scores_{model_name}.json')
        with open(ref_based_intermediate_path, 'w') as f:
            json.dump(ref_based_intermediates, f, indent=4)
        
        print(f"\nRef-based evaluation results saved to: {ref_based_overall_path}")
    
    if args.eval_type in ['ref_free', 'both']:
        ref_free_overall_path = os.path.join(eval_output_dir, f'ref_free_overall_scores_{model_name}.json')
        with open(ref_free_overall_path, 'w') as f:
            json.dump(ref_free_scores, f, indent=4)
        
        ref_free_intermediate_path = os.path.join(eval_output_dir, f'ref_free_intermediate_scores_{model_name}.json')
        with open(ref_free_intermediate_path, 'w') as f:
            json.dump(ref_free_intermediates, f, indent=4)
        
        print(f"\nRef-free evaluation results saved to: {ref_free_overall_path}")
    
    # Print summary
    print(f"\n=== Evaluation Summary ===")
    print(f"Model: {model_name}")
    print(f"Output directory: {eval_output_dir}")
    print(f"Evaluation type: {args.eval_type}")
    
    if args.eval_type in ['ref_based', 'both']:
        print(f"\n=== Reference-Based Evaluation ===")
        for task_type, scores in ref_based_scores.items():
            if scores:
                print(f"\n{task_type.upper()}:")
                print(f"  Average best n_clip: {scores['best_n_clip']:.4f}")
                print(f"  Average best pl: {scores['best_pl']:.4f}")
                print(f"  Instances evaluated: {scores['num_instances']}")
            else:
                print(f"\n{task_type.upper()}: No valid scores")
    
    if args.eval_type in ['ref_free', 'both']:
        print(f"\n=== Reference-Free Evaluation ===")
        print(f"GPT Model used: {args.gpt_model}")
        for task_type, scores in ref_free_scores.items():
            if scores:
                print(f"\n{task_type.upper()}:")
                print(f"  Average best score: {scores['average_best_score']:.4f}")
                print(f"  Instances evaluated: {scores['num_instances']}")
            else:
                print(f"\n{task_type.upper()}: No valid scores")


if __name__ == "__main__":
    main()

