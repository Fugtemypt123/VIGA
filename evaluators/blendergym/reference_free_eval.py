#!/usr/bin/env python3
"""
Reference-free evaluation script for AgenticVerifier blendergym results using GPT.
This script evaluates the quality of generated images against task descriptions without requiring ground truth images.
"""

import os
import sys
import argparse
import json
import base64
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from transformers import CLIPProcessor, CLIPModel
from concurrent.futures import ThreadPoolExecutor, as_completed

import openai
openai.api_key = os.environ["OPENAI_API_KEY"]
client = openai.OpenAI()

# Task instance counts for different task types
TASK_INSTANCE_COUNT_DICT = {
    'geometry': 45,
    'material': 40,
    'blendshape': 75,
    'placement': 40,
    'lighting': 40
}

# Global CLIP model/processor to share across threads
GLOBAL_CLIP_MODEL = None
GLOBAL_CLIP_PROCESSOR = None

# Evaluation criteria for different aspects
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

INSTRUCTION_TEMPLATE = """You are evaluating a 3D scene image generated based on a specific task description. Please evaluate the image based on the following criteria:

{criteria}

Task Description: {task_description}

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


def encode_image(image_path):
    """Encode image to base64 for GPT API."""
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def evaluate_image_with_gpt(image_path: str, task_description: str, criteria_name: str, criteria_dict: dict, model_name: str = "gpt-4o", max_tokens: int = 200):
    """
    Evaluate a single image using GPT based on specific criteria.
    
    Args:
        image_path: Path to the image to evaluate
        task_description: Text description of the task
        criteria_name: Name of the evaluation criteria
        criteria_dict: Dictionary containing criteria and scale
        model_name: GPT model to use
        max_tokens: Maximum tokens for response
        
    Returns:
        dict: Evaluation result with score and justification
    """
    try:
        # Encode image
        image_base64 = encode_image(image_path)
        image_url = f"data:image/jpeg;base64,{image_base64}"
        
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
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        }]
        
        # Get response from GPT
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.0,
        )
        
        response_text = response.choices[0].message.content.strip()
        
        # Parse response
        try:
            score, justification = response_text.split(".", 1)
            score = float(score.strip())
        except:
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


def load_task_description(task_dir: str) -> str:
    """
    Load task description from task.txt file.
    
    Args:
        task_dir: Directory containing the task
        
    Returns:
        str: Task description text
    """
    task_file = os.path.join(task_dir, "task.txt")
    if os.path.exists(task_file):
        with open(task_file, 'r', encoding='utf-8') as f:
            return f.read().strip()
    return ""


def process_task_instance_reference_free(output_base_dir: str, task_dir: str, model_name: str = "gpt-4o", max_tokens: int = 200):
    """
    Process a single task instance directory and compute reference-free metrics across rounds.

    Returns:
        tuple: (task_dir, task_instance_scores, best_scores)
    """
    task_instance_dir = os.path.join(output_base_dir, task_dir)
    renders_dir = os.path.join(task_instance_dir, "renders")

    if not os.path.exists(renders_dir):
        return task_dir, {}, {}

    # Load task description
    task_description = load_task_description(task_instance_dir)
    if not task_description:
        print(f"Warning: No task description found for {task_dir}")
        task_description = "Task description not available"

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
        return task_dir, {}, {}

    for round_dir in round_dirs:
        round_path = os.path.join(renders_dir, round_dir)
        task_instance_scores[round_dir] = {}

        # Evaluate each render in the round
        for render_name in ["render1.png", "render2.png"]:
            render_path = os.path.join(round_path, render_name)
            if os.path.exists(render_path):
                render_scores = {}
                
                # Evaluate with GPT for each criteria
                for criteria_name, criteria_dict in EVALUATION_CRITERIA.items():
                    gpt_result = evaluate_image_with_gpt(
                        render_path, 
                        task_description, 
                        criteria_name, 
                        criteria_dict,
                        model_name,
                        max_tokens
                    )
                    render_scores[criteria_name] = gpt_result
                
                # Calculate average score across all criteria
                scores = [gpt_result["score"] for gpt_result in render_scores.values()]
                avg_score = sum(scores) / len(scores) if scores else 0.0
                
                render_scores["average_score"] = avg_score
                task_instance_scores[round_dir][render_name.replace(".png", "")] = render_scores

        # Calculate round average
        round_scores = []
        for render_name, render_data in task_instance_scores[round_dir].items():
            if "average_score" in render_data:
                round_scores.append(render_data["average_score"])
        
        if round_scores:
            task_instance_scores[round_dir]["round_average"] = sum(round_scores) / len(round_scores)

    # Find best round
    best_scores = {}
    valid_rounds = {k: v for k, v in task_instance_scores.items() if "round_average" in v}
    if valid_rounds:
        best_round = max(valid_rounds.keys(), key=lambda r: valid_rounds[r]["round_average"])
        best_scores = {
            "best_round": best_round,
            "best_score": valid_rounds[best_round]["round_average"],
            "best_round_scores": valid_rounds[best_round]
        }

    # Save individual instance scores
    instance_scores_path = os.path.join(task_instance_dir, 'reference_free_scores.json')
    try:
        with open(instance_scores_path, 'w') as f:
            json.dump(task_instance_scores, f, indent=4)
    except Exception as e:
        print(f"Error saving scores for {task_dir}: {e}")

    return task_dir, task_instance_scores, best_scores


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
    parser = argparse.ArgumentParser(description='Reference-free evaluation for AgenticVerifier blendergym results')
    parser.add_argument('test_id', type=str, help='Test ID (e.g., 20250815_150016)')
    parser.add_argument('--output_dir', type=str, default=None, 
                       help='Output directory for evaluation results (default: output/blendergym/{test_id}/_reference_free_evaluation)')
    parser.add_argument('--model_name', type=str, default="gpt-4o", 
                       help='GPT model to use for evaluation')
    parser.add_argument('--max_tokens', type=int, default=200,
                       help='Maximum tokens for GPT responses')
    parser.add_argument('--max_workers', type=int, default=4,
                       help='Maximum number of parallel workers')
    
    args = parser.parse_args()
    test_id = args.test_id
    
    # Set up paths
    output_base_dir = f"output/blendergym/{test_id}"
    if not os.path.exists(output_base_dir):
        raise ValueError(f"Output directory {output_base_dir} does not exist.")
    
    if args.output_dir:
        eval_output_dir = args.output_dir
    else:
        eval_output_dir = os.path.join(output_base_dir, "_reference_free_evaluation")
    
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
            'best_scores': [],
            'instance_details': {}
        }

        # Run per-instance processing in parallel threads
        max_workers = min(args.max_workers, (os.cpu_count() or 4))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(process_task_instance_reference_free, output_base_dir, task_dir, args.model_name, args.max_tokens)
                for task_dir, _ in task_instances
            ]

            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {task_type}"):
                try:
                    task_dir, task_instance_scores, best_scores = future.result()
                    scores_across_instances['instance_details'][task_dir] = task_instance_scores
                    if best_scores:
                        scores_across_instances['best_scores'].append(best_scores)
                        print(f"    {task_dir}: Best score={best_scores['best_score']:.4f} (round {best_scores['best_round']})")
                    else:
                        print(f"    {task_dir}: No valid scores")
                except Exception as e:
                    print(f"    Error processing {task_type} instance: {e}")

        # Aggregate results for this task type
        if scores_across_instances['best_scores']:
            best_scores_list = [score['best_score'] for score in scores_across_instances['best_scores']]
            scores_across_tasks[task_type] = {
                'average_best_score': sum(best_scores_list) / len(best_scores_list),
                'num_instances': len(best_scores_list),
                'best_scores': scores_across_instances['best_scores']
            }

            print(f"  Task {task_type} overall scores:")
            print(f"    Average best score: {scores_across_tasks[task_type]['average_best_score']:.4f}")
            print(f"    Number of instances: {scores_across_tasks[task_type]['num_instances']}")
        else:
            print(f"  No valid scores for task type {task_type}")
            scores_across_tasks[task_type] = {}

        intermediates[task_type] = scores_across_instances
    
    # Save overall results
    overall_scores_path = os.path.join(eval_output_dir, 'reference_free_overall_scores.json')
    with open(overall_scores_path, 'w') as f:
        json.dump(scores_across_tasks, f, indent=4)
    
    intermediate_scores_path = os.path.join(eval_output_dir, 'reference_free_intermediate_scores.json')
    with open(intermediate_scores_path, 'w') as f:
        json.dump(intermediates, f, indent=4)
    
    # Print summary
    print(f"\n=== Reference-Free Evaluation Summary ===")
    print(f"Test ID: {test_id}")
    print(f"Output directory: {eval_output_dir}")
    print(f"Results saved to: {overall_scores_path}")
    print(f"GPT Model used: {args.model_name}")
    
    for task_type, scores in scores_across_tasks.items():
        if scores:
            print(f"\n{task_type.upper()}:")
            print(f"  Average best score: {scores['average_best_score']:.4f}")
            print(f"  Instances evaluated: {scores['num_instances']}")
        else:
            print(f"\n{task_type.upper()}: No valid scores")


if __name__ == "__main__":
    main()
