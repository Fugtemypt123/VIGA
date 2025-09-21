#!/usr/bin/env python3
"""
BlenderGym Baseline Runner for AgenticVerifier
Loads completed BlenderGym tasks from output directory and runs tournament-style evaluation
using VLM to compare images and select winners.
"""
import os
import sys
import json
import time
import argparse
import base64
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
import numpy as np
import torch
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from transformers import CLIPProcessor, CLIPModel
import openai
from tqdm import tqdm

# Global CLIP model/processor to share across threads
GLOBAL_CLIP_MODEL = None
GLOBAL_CLIP_PROCESSOR = None


def convert_numpy_types(obj):
    """
    Convert numpy types to Python native types for JSON serialization.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


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


def encode_image(image_path: str) -> str:
    """
    Encode image to base64 string for OpenAI API.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Base64 encoded string
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def vlm_compare_images(image1_path: str, image2_path: str, target_path: str, 
                      api_key: str, base_url: str, model: str = "gpt-4o") -> int:
    """
    Use VLM to compare two images and determine which is closer to target.
    
    Args:
        image1_path: Path to first image
        image2_path: Path to second image  
        target_path: Path to target image
        api_key: OpenAI API key
        base_url: OpenAI base URL
        model: Vision model to use
        
    Returns:
        1 if image1 is closer to target, 2 if image2 is closer to target
    """
    try:
        # Encode images
        image1_b64 = encode_image(image1_path)
        image2_b64 = encode_image(image2_path)
        target_b64 = encode_image(target_path)
        
        # Initialize OpenAI client
        client = openai.OpenAI(api_key=api_key, base_url=base_url)
        
        # Create messages
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "You are an expert at comparing 3D rendered images. I will show you two rendered images and a target image. Please determine which of the two rendered images is closer to the target image in terms of visual similarity, lighting, materials, geometry, and overall appearance. Respond with only '1' if the first image is closer to the target, or '2' if the second image is closer to the target."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{target_b64}"
                        }
                    },
                    {
                        "type": "text", 
                        "text": "Target image:"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image1_b64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": "Image 1:"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image2_b64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": "Image 2:"
                    }
                ]
            }
        ]
        
        # Make API call
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=10,
            temperature=0.1
        )
        
        # Parse response
        result = response.choices[0].message.content.strip()
        if result == "1":
            return 1
        elif result == "2":
            return 2
        else:
            # Fallback: use CLIP similarity
            print(f"Unexpected VLM response: {result}, using CLIP fallback")
            return clip_fallback_comparison(image1_path, image2_path, target_path)
            
    except Exception as e:
        print(f"VLM comparison failed: {e}, using CLIP fallback")
        return clip_fallback_comparison(image1_path, image2_path, target_path)


def clip_fallback_comparison(image1_path: str, image2_path: str, target_path: str) -> int:
    """
    Fallback comparison using CLIP similarity when VLM fails.
    
    Args:
        image1_path: Path to first image
        image2_path: Path to second image
        target_path: Path to target image
        
    Returns:
        1 if image1 is closer to target, 2 if image2 is closer to target
    """
    try:
        image1 = Image.open(image1_path)
        image2 = Image.open(image2_path)
        target = Image.open(target_path)
        
        sim1 = clip_similarity(image1, target)
        sim2 = clip_similarity(image2, target)
        
        return 1 if sim1 > sim2 else 2
    except Exception as e:
        print(f"CLIP fallback also failed: {e}, defaulting to image1")
        return 1


def get_max_rounds_for_task(task_name: str) -> int:
    """
    Get the maximum number of rounds for a task from the reference directory.
    
    Args:
        task_name: Name of the task (e.g., 'blendshape1')
        
    Returns:
        Maximum round number found in the reference directory
    """
    reference_dir = Path(f"output/blendergym/gpt-4o/{task_name}/renders")
    if not reference_dir.exists():
        print(f"Warning: Reference directory not found: {reference_dir}")
        return 8  # Default fallback
    
    max_round = 0
    for round_dir in reference_dir.iterdir():
        if round_dir.is_dir() and round_dir.name.isdigit():
            round_num = int(round_dir.name)
            max_round = max(max_round, round_num)
    
    return max_round


def load_tasks_from_output(output_dir: str) -> List[Dict]:
    """
    Load completed tasks from output directory.
    
    Args:
        output_dir: Path to output directory containing task results
        
    Returns:
        List of task configurations
    """
    tasks = []
    output_path = Path(output_dir)
    
    if not output_path.exists():
        print(f"Error: Output directory does not exist: {output_path}")
        return tasks
    
    # Find all task directories (e.g., blendshape1, geometry5, etc.)
    for task_dir in output_path.iterdir():
        if task_dir.is_dir() and task_dir.name != "_evaluation":
            renders_dir = task_dir / "renders"
            if renders_dir.exists():
                task_name = task_dir.name
                
                # Get maximum rounds from reference directory
                max_rounds = get_max_rounds_for_task(task_name)
                print(f"Task {task_name}: using max rounds = {max_rounds}")
                
                # Check for round directories (1 to max_rounds)
                round_dirs = []
                for i in range(1, max_rounds + 1):
                    round_dir = renders_dir / str(i)
                    if round_dir.exists():
                        # Check for render files
                        render1 = round_dir / "render1.png"
                        if render1.exists():
                            round_dirs.append(i)
                
                # Find corresponding target images
                target_renders_dir = Path(f"data/blendergym/{task_name}/renders/goal")
                if target_renders_dir.exists():
                    target_render1 = target_renders_dir / "render1.png"
                    
                    if target_render1.exists():
                        task_config = {
                            "task_name": task_name,
                            "task_dir": str(task_dir),
                            "renders_dir": str(renders_dir),
                            "target_renders_dir": str(target_renders_dir),
                            "round_dirs": sorted(round_dirs),
                            "max_rounds": max_rounds
                        }
                        tasks.append(task_config)
                        
                        if len(round_dirs) >= 2:
                            print(f"Found task: {task_name} with {len(round_dirs)} rounds (max_rounds={max_rounds}) - will run tournament")
                        elif len(round_dirs) == 1:
                            print(f"Found task: {task_name} with {len(round_dirs)} round (max_rounds={max_rounds}) - will auto-win")
                        else:
                            print(f"Found task: {task_name} with {len(round_dirs)} rounds (max_rounds={max_rounds}) - will get penalty score")
                    else:
                        print(f"Warning: Target renders not found for {task_name}")
                else:
                    print(f"Warning: Target renders directory not found for {task_name}")
    
    return tasks


def run_tournament(task_config: Dict, args) -> Dict:
    """
    Run tournament-style evaluation for a single task.
    
    Args:
        task_config: Task configuration
        args: Command line arguments
        
    Returns:
        Dictionary with tournament results
    """
    task_name = task_config['task_name']
    renders_dir = Path(task_config['renders_dir'])
    target_renders_dir = Path(task_config['target_renders_dir'])
    
    print(f"\nRunning tournament for task: {task_name}")
    
    # Get all available round directories
    available_rounds = task_config['round_dirs']
    
    # Prepare images for tournament (use rounds 1 to max_rounds)
    max_rounds = task_config.get('max_rounds', 8)  # Default fallback
    images = []
    for round_num in range(1, max_rounds + 1):
        if round_num in available_rounds:  # Only use rounds that exist
            round_dir = renders_dir / str(round_num)
            render1_path = round_dir / "render1.png"
            render2_path = round_dir / "render2.png"
            
            if render1_path.exists() and render2_path.exists():
                images.append({
                    'round': round_num,
                    'render1': str(render1_path),
                    'render2': str(render2_path)
                })
    
    # Handle special cases: 0 rounds or 1 round
    if len(images) == 0:
        print(f"Warning: No rounds available for {task_name}, assigning penalty score")
        return {
            "task_name": task_name,
            "max_rounds": max_rounds,
            "total_participants": 0,
            "special_case": "no_rounds",
            "final_metrics": {
                "n_clip_render1": 1.0,  # Maximum penalty
                "n_clip_render2": 1.0,  # Maximum penalty
                "avg_n_clip": 1.0,      # Maximum penalty
                "pl_render1": 1.0,      # Maximum penalty
                "pl_render2": 1.0,      # Maximum penalty
                "avg_pl": 1.0           # Maximum penalty
            }
        }
    elif len(images) == 1:
        print(f"Only 1 round available for {task_name}, auto-winning")
        # Calculate metrics for the single round
        single_image = images[0]
        target_render1 = str(target_renders_dir / "render1.png")
        target_render2 = str(target_renders_dir / "render2.png")
        
        # Load images for metric calculation
        winner_render1 = Image.open(single_image['render1'])
        winner_render2 = Image.open(single_image['render2'])
        target_img1 = Image.open(target_render1)
        target_img2 = Image.open(target_render2)
        
        # CLIP metrics (1 - similarity = distance)
        clip1 = 1 - clip_similarity(winner_render1, target_img1)
        clip2 = 1 - clip_similarity(winner_render2, target_img2)
        avg_clip = (clip1 + clip2) / 2
        
        # Photometric loss
        pl1 = photometric_loss(winner_render1, target_img1)
        pl2 = photometric_loss(winner_render2, target_img2)
        avg_pl = (pl1 + pl2) / 2
        
        return {
            "task_name": task_name,
            "max_rounds": max_rounds,
            "total_participants": 1,
            "special_case": "auto_win",
            "final_winner": single_image,
            "final_metrics": {
                "n_clip_render1": clip1,
                "n_clip_render2": clip2,
                "avg_n_clip": avg_clip,
                "pl_render1": pl1,
                "pl_render2": pl2,
                "avg_pl": avg_pl
            }
        }
    
    # Target images
    target_render1 = str(target_renders_dir / "render1.png")
    target_render2 = str(target_renders_dir / "render2.png")
    
    # Tournament: dynamic rounds with bye logic
    current_images = images.copy()
    
    tournament_results = {
        "task_name": task_name,
        "max_rounds": max_rounds,
        "total_participants": len(images),
        "rounds": []
    }
    
    round_num = 1
    while len(current_images) > 1:
        print(f"  Tournament Round {round_num}: {len(current_images)} images remaining")
        
        round_results = {
            "round": round_num,
            "participants": len(current_images),
            "comparisons": [],
            "byes": []
        }
        
        # Pair up images for comparison
        next_round_images = []
        
        # Handle byes: if odd number of participants, the last one gets a bye
        if len(current_images) % 2 == 1:
            bye_image = current_images[-1]
            next_round_images.append(bye_image)
            round_results["byes"].append(bye_image['round'])
            print(f"    Round {bye_image['round']} gets a bye")
            current_images = current_images[:-1]  # Remove the bye image from current round
        
        # Pair up remaining images for comparison
        for i in range(0, len(current_images), 2):
            if i + 1 < len(current_images):
                img1 = current_images[i]
                img2 = current_images[i + 1]
                
                # Compare only render1 vs render1 to determine tournament winner
                winner_render1 = vlm_compare_images(
                    img1['render1'], img2['render1'], target_render1,
                    args.api_key, args.openai_base_url, args.vision_model
                )
                
                # Winner is determined solely by render1 comparison
                winner_idx = winner_render1 - 1  # Convert to 0-based index
                
                winner = current_images[i + winner_idx]
                next_round_images.append(winner)
                
                round_results["comparisons"].append({
                    "img1_round": img1['round'],
                    "img2_round": img2['round'],
                    "winner_render1": winner_render1,
                    "final_winner": img1['round'] if winner_idx == 0 else img2['round']
                })
        
        tournament_results["rounds"].append(round_results)
        current_images = next_round_images
        round_num += 1
    
    # Final winner
    if current_images:
        final_winner = current_images[0]
        tournament_results["final_winner"] = final_winner
        
        # Calculate final metrics
        winner_render1 = Image.open(final_winner['render1'])
        winner_render2 = Image.open(final_winner['render2'])
        target_img1 = Image.open(target_render1)
        target_img2 = Image.open(target_render2)
        
        # CLIP metrics (1 - similarity = distance)
        clip1 = 1 - clip_similarity(winner_render1, target_img1)
        clip2 = 1 - clip_similarity(winner_render2, target_img2)
        avg_clip = (clip1 + clip2) / 2
        
        # Photometric loss
        pl1 = photometric_loss(winner_render1, target_img1)
        pl2 = photometric_loss(winner_render2, target_img2)
        avg_pl = (pl1 + pl2) / 2
        
        tournament_results["final_metrics"] = {
            "n_clip_render1": clip1,
            "n_clip_render2": clip2,
            "avg_n_clip": avg_clip,
            "pl_render1": pl1,
            "pl_render2": pl2,
            "avg_pl": avg_pl
        }
        
        print(f"  Final winner: Round {final_winner['round']}")
        print(f"  Final metrics: n_clip={avg_clip:.4f}, pl={avg_pl:.4f}")
    
    return tournament_results


def run_tasks_parallel(tasks: List[Dict], args, max_workers: int = 4) -> Dict:
    """
    Run tournament evaluations in parallel.
    
    Args:
        tasks: List of task configurations
        args: Command line arguments
        max_workers: Maximum number of parallel workers
        
    Returns:
        Dictionary with all results
    """
    results = {
        "tasks": [],
        "summary": {
            "total_tasks": len(tasks),
            "successful_tasks": 0,
            "failed_tasks": 0,
            "execution_time": 0
        }
    }
    
    print(f"\nStarting parallel tournament evaluation with max {max_workers} workers...")
    print(f"Total tasks: {len(tasks)}")
    
    start_time = time.time()
    
    # Use ThreadPoolExecutor for parallel execution
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(run_tournament, task_config, args): task_config 
            for task_config in tasks
        }
        
        # Process completed tasks
        for future in tqdm(as_completed(future_to_task), total=len(future_to_task), desc="Running tournaments"):
            task_config = future_to_task[future]
            try:
                result = future.result()
                results["tasks"].append(result)
                
                if "error" not in result:
                    results["summary"]["successful_tasks"] += 1
                    print(f"✅ {result['task_name']} completed successfully")
                else:
                    results["summary"]["failed_tasks"] += 1
                    print(f"❌ {result['task_name']} failed: {result['error']}")
                    
            except Exception as e:
                results["summary"]["failed_tasks"] += 1
                error_result = {
                    "task_name": task_config['task_name'],
                    "error": str(e)
                }
                results["tasks"].append(error_result)
                print(f"❌ {task_config['task_name']} failed with exception: {e}")
    
    end_time = time.time()
    results["summary"]["execution_time"] = end_time - start_time
    
    return results


def main():
    parser = argparse.ArgumentParser(description="BlenderGym Baseline Tournament Runner")
    
    # Input parameters
    parser.add_argument("--test-id", required=True, help="Test ID (e.g., gpt-4o-bestofn)")
    parser.add_argument("--output-dir", default=None, help="Output directory for results (default: output/baseline/{test_id})")
    
    # VLM parameters
    parser.add_argument("--vision-model", default="gpt-4o", help="OpenAI vision model to use")
    parser.add_argument("--openai-base-url", default=os.getenv("OPENAI_BASE_URL"), help="OpenAI-compatible API base URL")
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"), help="OpenAI API key")
    
    # Execution parameters
    parser.add_argument("--max-workers", type=int, default=4, help="Maximum number of parallel workers")
    parser.add_argument("--task-filter", help="Filter tasks by name pattern (e.g., 'blendshape')")
    
    args = parser.parse_args()
    
    # Validate API key
    if not args.api_key:
        print("Error: OpenAI API key is required. Set OPENAI_API_KEY environment variable or use --api-key")
        sys.exit(1)
    
    # Set up paths
    input_dir = f"output/blendergym/{args.test_id}"
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = f"output/baseline/{args.test_id}"
    
    print(f"Loading tasks from: {input_dir}")
    tasks = load_tasks_from_output(input_dir)
    
    if not tasks:
        print("No valid tasks found!")
        sys.exit(1)
    
    # Filter tasks if specified
    if args.task_filter:
        tasks = [t for t in tasks if args.task_filter in t["task_name"]]
        print(f"Filtered to {len(tasks)} tasks matching '{args.task_filter}'")
    
    if not tasks:
        print("No tasks match the specified filters!")
        sys.exit(1)
    
    print(f"Found {len(tasks)} tasks for tournament evaluation")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save args to json
    with open(os.path.join(output_dir, "args.json"), "w") as f:
        json.dump(convert_numpy_types(args.__dict__), f, indent=2)
    
    # Ensure CLIP is loaded
    ensure_clip_loaded()
    
    # Run tournaments
    results = run_tasks_parallel(tasks, args, max_workers=args.max_workers)
    
    # Save results
    try: 
        results_path = os.path.join(output_dir, "tournament_results.json")
        with open(results_path, "w") as f:
            json.dump(convert_numpy_types(results), f, indent=2)
    except Exception as e:
        results_txt_path = os.path.join(output_dir, "tournament_results.txt")
        with open(results_txt_path, "w") as f:
            f.write(str(results))
    
    # Calculate summary statistics
    successful_tasks = results["tasks"]
    if successful_tasks:
        # Group by task type
        task_types = {}
        for task in successful_tasks:
            if "error" not in task:
                task_name = task["task_name"]
                task_type = ''.join([c for c in task_name if not c.isdigit()])  # Extract task type
                if task_type not in task_types:
                    task_types[task_type] = []
                task_types[task_type].append(task)
        
        # Calculate averages
        summary_stats = {}
        for task_type, type_tasks in task_types.items():
            n_clip_values = [t["final_metrics"]["avg_n_clip"] for t in type_tasks]
            pl_values = [t["final_metrics"]["avg_pl"] for t in type_tasks]
            
            # Count special cases
            auto_wins = sum(1 for t in type_tasks if t.get("special_case") == "auto_win")
            penalties = sum(1 for t in type_tasks if t.get("special_case") == "no_rounds")
            tournaments = len(type_tasks) - auto_wins - penalties
            
            summary_stats[task_type] = {
                "num_tasks": len(type_tasks),
                "tournaments_run": tournaments,
                "auto_wins": auto_wins,
                "penalty_scores": penalties,
                "avg_n_clip": sum(n_clip_values) / len(n_clip_values),
                "avg_pl": sum(pl_values) / len(pl_values),
                "best_n_clip": min(n_clip_values),
                "worst_n_clip": max(n_clip_values),
                "best_pl": min(pl_values),
                "worst_pl": max(pl_values)
            }
        
        # Save summary
        summary_path = os.path.join(output_dir, "summary_stats.json")
        with open(summary_path, "w") as f:
            json.dump(convert_numpy_types(summary_stats), f, indent=2)
        
        # Print summary
        print(f"\n{'='*60}")
        print("TOURNAMENT EVALUATION SUMMARY")
        print(f"{'='*60}")
        print(f"Total tasks: {results['summary']['total_tasks']}")
        print(f"Successful: {results['summary']['successful_tasks']}")
        print(f"Failed: {results['summary']['failed_tasks']}")
        print(f"Execution time: {results['summary']['execution_time']:.2f} seconds")
        print(f"Results saved to: {output_dir}")
        
        print(f"\nSummary Statistics:")
        for task_type, stats in summary_stats.items():
            print(f"\n{task_type.upper()}:")
            print(f"  Tasks evaluated: {stats['num_tasks']}")
            print(f"  Tournaments run: {stats['tournaments_run']}")
            print(f"  Auto-wins: {stats['auto_wins']}")
            print(f"  Penalty scores: {stats['penalty_scores']}")
            print(f"  Average n_clip: {stats['avg_n_clip']:.4f}")
            print(f"  Average pl: {stats['avg_pl']:.4f}")
            print(f"  Best n_clip: {stats['best_n_clip']:.4f}")
            print(f"  Worst n_clip: {stats['worst_n_clip']:.4f}")
            print(f"  Best pl: {stats['best_pl']:.4f}")
            print(f"  Worst pl: {stats['worst_pl']:.4f}")
    
    else:
        print("No successful tasks completed!")


if __name__ == "__main__":
    main()