#!/usr/bin/env python3
"""Iterative Alchemy Runner for BlenderBench.

Generates code iteratively using GPT to transform images closer to target.
Each iteration generates multiple candidates, selects the best using VLM comparison.
"""

import argparse
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

# Import shared utilities
from runners.shared import (
    execute_blender_code,
    generate_candidate_codes,
    tournament_select_best,
)


def load_blenderbench_dataset(
    base_path: str,
    task_name: str,
    task_id: Optional[str] = None
) -> List[Dict]:
    """Load BlenderBench dataset structure.

    Args:
        base_path: Path to BlenderBench dataset root.
        task_name: Task name (e.g., 'level1', 'level2', 'level3', or 'all').
        task_id: Optional task ID to filter specific tasks.

    Returns:
        List of task configurations.
    """
    tasks = []
    base_path = Path(base_path)

    if not base_path.exists():
        print(f"Error: BlenderBench dataset path does not exist: {base_path}")
        return tasks

    if task_name == 'all':
        task_list = ['level1', 'level2', 'level3']
    else:
        task_list = [task_name]

    # If task_id is not None, only run the task_id
    if task_id is not None:
        task_dirs = [(base_path / f"{task_name}/{task_id}", task_name + '-' + task_id[-1])]
    # Otherwise, run all tasks in the task_list
    else:
        task_dirs = []
        for task in task_list:
            task_path = base_path / task
            for task_dir in task_path.glob("*"):
                task_name_final = task + '-' + str(task_dir.name.split('/')[-1][-1])
                task_dirs.append((task_dir, task_name_final))

    for task_dir, task_name_final in task_dirs:
        # Check for required files
        start_code_path = task_dir / "start.py"
        start_renders_dir = task_dir / "renders" / "start"
        goal_renders_dir = task_dir / "renders" / "goal"
        blender_file = task_dir / "blender_file.blend"

        if not start_code_path.exists():
            print(f"Warning: start.py not found in {task_dir}")
            continue

        if not goal_renders_dir.exists() or not start_renders_dir.exists():
            print(f"Warning: renders directory not found: {goal_renders_dir}")
            continue

        if not blender_file.exists():
            print(f"Warning: blender_file.blend not found in {task_dir}")
            continue

        task_name_path = task_dir / "task.txt"
        if os.path.exists(task_name_path):
            task_description = open(task_name_path, 'r').read()
        else:
            task_description = ""

        task_config = {
            "task_name": task_name_final,
            "task_dir": str(task_dir),
            "init_code_path": str(start_code_path),
            "init_image_path": str(start_renders_dir),
            "target_image_path": str(goal_renders_dir),
            "blender_file": str(blender_file),
            "target_description": task_description,
        }
        tasks.append(task_config)
        print(f"Found task: {task_name_final}")

    return tasks


def run_iterative_alchemy(task_config: Dict, args: argparse.Namespace) -> Dict:
    """Run iterative alchemy process for a single task.

    Generates candidates -> tournament -> update -> repeat.
    Saves results in the format: renders/1/, renders/2/, ..., renders/10/
    and scripts/1.py, ..., scripts/10.py

    Args:
        task_config: Task configuration dictionary.
        args: Command line arguments.

    Returns:
        Dictionary with results.
    """
    task_name = "/".join(task_config['task_dir'].split('/')[-2:])
    print(f"\n{'='*60}")
    print(f"Running iterative alchemy for task: {task_name}")
    print(f"{'='*60}")

    # Setup paths
    output_dir = Path(args.output_dir) / task_name
    output_dir.mkdir(parents=True, exist_ok=True)

    script_save_dir = output_dir / "scripts"
    render_save_dir = output_dir / "renders"

    # Get initial code
    with open(task_config['init_code_path'], 'r') as f:
        current_code = f.read()

    # Get image paths
    start_image_dir = Path(task_config['init_image_path'])
    target_image_dir = Path(task_config['target_image_path'])

    # Find render1.png files
    start_images = list(start_image_dir.glob("render*.png"))
    target_images = list(target_image_dir.glob("render*.png"))

    if not start_images or not target_images:
        return {
            "task_name": task_name,
            "error": "Missing start or target images",
            "success": False
        }

    start_image_path = str(start_images[0])
    target_image_path = str(target_images[0])
    current_image_path = start_image_path

    # Copy blender file to output
    shutil.copy(task_config['blender_file'], output_dir / "blender_file.blend")
    blender_file = str(output_dir / "blender_file.blend")

    # Iterative process
    for round_num in range(1, args.max_iterations + 1):
        print(f"\n--- Round {round_num}/{args.max_iterations} ---")

        # Generate candidate codes
        print("Generating candidate codes...")
        candidate_codes = generate_candidate_codes(
            start_image_path=start_image_path,
            current_image_path=current_image_path,
            current_code=current_code,
            target_image_path=target_image_path,
            task_description=task_config.get('target_description', ''),
            model=args.model,
            num_candidates=args.num_candidates
        )

        print(f"Generated {len(candidate_codes)} candidate codes")

        # Execute all candidate codes
        candidate_results = []
        for i, code in enumerate(candidate_codes):
            print(f"  Executing candidate {i+1}/{len(candidate_codes)}...")
            success, error_msg, render_dir = execute_blender_code(
                blender_command=args.blender_command,
                blender_file=blender_file,
                blender_script=args.blender_script,
                code=code,
                round_name=f"temp_{round_num}_{i}",
                script_save_dir=None,
                render_save_dir=render_save_dir,
                gpu_devices=args.gpu_devices
            )

            if success and render_dir:
                candidate_results.append({
                    'code': code,
                    'render_dir': render_dir,
                    'index': i
                })
                print(f"    Candidate {i+1} executed successfully")
            else:
                print(f"    Candidate {i+1} failed: {error_msg}")

        if len(candidate_results) == 0:
            print(f"No candidates executed successfully for round {round_num}, using previous code")
            if round_num > 1:
                prev_render_dir = render_save_dir / str(round_num - 1)
                current_render_dir = render_save_dir / str(round_num)
                if prev_render_dir.exists():
                    shutil.copytree(prev_render_dir, current_render_dir, dirs_exist_ok=True)
                code_file = script_save_dir / f"{round_num}.py"
                with open(code_file, "w") as f:
                    f.write(current_code)
            continue

        # Run tournament to select best
        print(f"Running tournament with {len(candidate_results)} candidates...")
        winner_idx = tournament_select_best(
            candidate_results=candidate_results,
            target_image_path=target_image_path,
            model=args.model
        )

        winner = candidate_results[winner_idx]
        print(f"  Winner: Candidate {winner_idx + 1}")

        # Save winner to round directory
        winner_code = winner['code']
        winner_render_dir = Path(winner['render_dir'])
        round_render_dir = render_save_dir / str(round_num)
        round_script_file = script_save_dir / f"{round_num}.py"

        # Copy winner's render to round directory
        if winner_render_dir.exists():
            round_render_dir.mkdir(parents=True, exist_ok=True)
            for img in round_render_dir.glob("*.png"):
                img.unlink()
            for render_file in winner_render_dir.glob("*.png"):
                shutil.copy(render_file, round_render_dir / render_file.name)

        # Save winner's code
        script_save_dir.mkdir(parents=True, exist_ok=True)
        with open(round_script_file, "w") as f:
            f.write(winner_code)

        # Update current state for next iteration
        current_code = winner_code
        render_files = sorted(round_render_dir.glob("render*.png"))
        if render_files:
            current_image_path = str(render_files[0])

        print(f"  Round {round_num} completed: saved to scripts/{round_num}.py and renders/{round_num}/")

        # Clean up temporary candidate directories
        for i in range(len(candidate_codes)):
            temp_dir = render_save_dir / f"temp_{round_num}_{i}"
            if temp_dir.exists() and temp_dir.is_dir():
                shutil.rmtree(temp_dir)

    return {
        "task_name": task_name,
        "success": True,
        "rounds": args.max_iterations
    }


def main() -> None:
    """Entry point for the BlenderBench alchemy runner."""
    parser = argparse.ArgumentParser(description="Iterative Alchemy Runner for BlenderBench")

    # Input parameters
    parser.add_argument("--dataset-path", default="data/blenderbench", help="Path to BlenderBench dataset")
    parser.add_argument("--task", choices=['all', 'level1', 'level2', 'level3'], default='all', help="Task name")
    parser.add_argument("--task-id", default=None, help="Specific task ID to run")
    parser.add_argument("--output-dir", default=None, help="Output directory for results")

    # Iterative alchemy parameters
    parser.add_argument("--max-iterations", type=int, default=10, help="Maximum number of iterations (always 10 rounds)")
    parser.add_argument("--num-candidates", type=int, default=4, help="Number of candidate codes to generate per iteration")

    # Blender parameters
    parser.add_argument("--blender-command", default="utils/infinigen/blender/blender", help="Blender command path")
    parser.add_argument("--blender-script", default="data/blenderbench/generator_script.py", help="Blender execution script")
    parser.add_argument("--gpu-devices", default=None, help="GPU devices string (e.g., '0,1')")

    # VLM parameters
    parser.add_argument("--model", default="gpt-4o", help="OpenAI vision model to use")

    # Execution parameters
    parser.add_argument("--max-workers", type=int, default=4, help="Maximum number of parallel workers")
    parser.add_argument("--task-filter", help="Filter tasks by name pattern")

    args = parser.parse_args()

    # Set GPU devices if not provided
    if args.gpu_devices is None:
        args.gpu_devices = os.getenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")

    # Set output directory
    if not args.output_dir:
        args.output_dir = f"output/blenderbench/alchemy/{time.strftime('%Y%m%d_%H%M%S')}"

    # Always use 10 iterations
    args.max_iterations = 10

    # Load dataset
    print(f"Loading BlenderBench dataset from: {args.dataset_path}")
    tasks = load_blenderbench_dataset(args.dataset_path, args.task, args.task_id)

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

    print(f"Found {len(tasks)} tasks for iterative alchemy")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save args to json
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(args.__dict__, f, indent=2, default=str)

    # Save task list
    with open(os.path.join(args.output_dir, "tasks.json"), "w") as f:
        json.dump(tasks, f, indent=2)

    # Run iterative alchemy for each task
    print(f"\nStarting iterative alchemy with max {args.max_workers} workers...")
    print(f"Total tasks: {len(tasks)}")

    start_time = time.time()
    results = {
        "tasks": [],
        "summary": {
            "total_tasks": len(tasks),
            "successful_tasks": 0,
            "failed_tasks": 0,
            "execution_time": 0
        }
    }

    # Use ThreadPoolExecutor for parallel execution
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_task = {
            executor.submit(run_iterative_alchemy, task_config, args): task_config
            for task_config in tasks
        }

        for future in tqdm(as_completed(future_to_task), total=len(future_to_task), desc="Running alchemy"):
            task_config = future_to_task[future]
            try:
                result = future.result()
                results["tasks"].append(result)

                if result.get("success", False):
                    results["summary"]["successful_tasks"] += 1
                    print(f"{result['task_name']} completed successfully")
                else:
                    results["summary"]["failed_tasks"] += 1
                    print(f"{result['task_name']} failed: {result.get('error', 'Unknown error')}")

            except Exception as e:
                results["summary"]["failed_tasks"] += 1
                error_result = {
                    "task_name": task_config['task_name'],
                    "error": str(e),
                    "success": False
                }
                results["tasks"].append(error_result)
                print(f"{task_config['task_name']} failed with exception: {e}")

    end_time = time.time()
    results["summary"]["execution_time"] = end_time - start_time

    # Save results
    results_path = os.path.join(args.output_dir, "alchemy_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Print summary
    print(f"\n{'='*60}")
    print("ITERATIVE ALCHEMY SUMMARY")
    print(f"{'='*60}")
    print(f"Total tasks: {results['summary']['total_tasks']}")
    print(f"Successful: {results['summary']['successful_tasks']}")
    print(f"Failed: {results['summary']['failed_tasks']}")
    print(f"Execution time: {results['summary']['execution_time']:.2f} seconds")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
