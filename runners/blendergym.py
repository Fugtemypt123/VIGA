#!/usr/bin/env python3
"""
BlenderGym Runner for AgenticVerifier
Loads BlenderGym dataset and runs the dual-agent system for 3D scene generation.
"""
import os
import sys
import json
import time
import argparse
import subprocess
import asyncio
import signal
from pathlib import Path
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.common import get_model_info

def load_blendergym_dataset(base_path: str, task_name: str, test_id: Optional[str] = None, task_id: Optional[int] = None) -> List[Dict]:

    tasks = []
    base_path = Path(base_path)
    
    if not base_path.exists():
        print(f"Error: BlenderGym dataset path does not exist: {base_path}")
        return tasks
    
    if task_name == 'all':
        task_list = ['blendshape', 'geometry', 'lighting', 'material', 'placement']
    else:
        task_list = [task_name]
        
    current_task_dirs = []
    if test_id is not None:
        current_task_path = Path(f'output/blendergym/{test_id}')
        for task in task_list:
            for task_dir in current_task_path.glob(f"{task}*"):
                current_task_dir = task_dir / "renders/10"
                if os.path.exists(current_task_dir):
                    exist_score = False
                    with open(current_task_dir, 'r') as f:
                        scores = json.load(f)
                        for key, score in scores.items():
                            if score != {}:
                                exist_score = True
                                break
                    if exist_score:
                        current_task_dirs.append(os.path.basename(task_dir))
                
    task_dirs = []
    for task in task_list:
        for task_dir in base_path.glob(f"{task}*"):
            if os.path.basename(task_dir) not in current_task_dirs:
                task_dirs.append((task_dir, task))

    for task_dir, task_name in task_dirs:
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
            
        task_config = {
            "task_name": task_name,
            "task_dir": str(task_dir),
            "init_code_path": str(start_code_path),
            "init_image_path": str(start_renders_dir),
            "target_image_path": str(goal_renders_dir),
            "blender_file": str(blender_file),
        }
        if task_id is None or task_dir.name == f"{task}{task_id}":
            tasks.append(task_config)
        print(f"Found task: {task_name}/{task_dir.name}")
    
    return tasks

def run_blendergym_task(task_config: Dict, args) -> tuple:
    """
    Run a single BlenderGym task using main.py
    
    Args:
        task_config: Task configuration dictionary
        args: Command line arguments
        
    Returns:
        Tuple of (task_name, success: bool, error_message: str)
    """
    task_name = task_config['task_dir'].split('/')[-1]
    print(f"\n{'='*60}")
    print(f"Running task: {task_name}")
    print(f"{'='*60}")
    
    # Prepare output directories
    output_base = Path(args.output_dir + "/" + task_name)
    
    # Create directories
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Build main.py command
    cmd = [
        sys.executable, "main.py",
        "--mode", "blendergym",
        "--model", args.model,
        "--api-key", get_model_info(args.model)["api_key"],
        "--api-base-url", get_model_info(args.model)["base_url"],
        "--max-rounds", str(args.max_rounds),
        "--memory-length", str(args.memory_length),
        "--task-name", task_config["task_name"],
        "--init-code-path", str(task_config["init_code_path"]),
        "--init-image-path", str(task_config["init_image_path"]),
        "--target-image-path", str(task_config["target_image_path"]),
        "--output-dir", str(output_base),
        # Tool servers
        "--generator-tools", args.generator_tools,
        "--verifier-tools", args.verifier_tools,
        # Blender execution parameters (for generator)
        "--blender-command", args.blender_command,
        "--blender-file", str(task_config["blender_file"]),
        "--blender-script", args.blender_script,
        "--gpu-devices", args.gpu_devices,
        "--clear-memory",
        "--num-candidates", str(args.num_candidates),
    ]
    
    if args.no_tools:
        cmd.append("--no-tools")
    
    print(f"Command: {' '.join(cmd)}")
    
    try:
        # Run the command
        result = subprocess.run(cmd)  # no timeout
        print(f"Task completed successfully: {task_name}")
        return (task_name, True, "")
    except subprocess.CalledProcessError as e:
        error_msg = f"Task failed: {task_name}, Error: {e}"
        print(error_msg)
        return (task_name, False, str(e))
    except subprocess.TimeoutExpired:
        error_msg = f"Task timed out: {task_name}"
        print(error_msg)
        return (task_name, False, "Timeout")
    except Exception as e:
        error_msg = f"Task failed with exception: {task_name}, Error: {e}"
        print(error_msg)
        return (task_name, False, str(e))

def run_tasks_parallel(tasks: List[Dict], args, max_workers: int = 10) -> tuple:
    """
    Run tasks in parallel using ThreadPoolExecutor
    
    Args:
        tasks: List of task configurations
        args: Command line arguments
        max_workers: Maximum number of parallel workers
        
    Returns:
        Tuple of (successful_tasks: int, failed_tasks: int, failed_task_details: List)
    """
    successful_tasks = 0
    failed_tasks = 0
    failed_task_details = []
    
    print(f"\nStarting parallel execution with max {max_workers} workers...")
    print(f"Total tasks: {len(tasks)}")
    
    # Use ThreadPoolExecutor for parallel execution
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(run_blendergym_task, task_config, args): task_config 
            for task_config in tasks
        }
        
        # Process completed tasks
        for future in as_completed(future_to_task):
            task_config = future_to_task[future]
            try:
                task_name, success, error_msg = future.result()
                if success:
                    successful_tasks += 1
                    print(f"✅ {task_name} completed successfully")
                else:
                    failed_tasks += 1
                    failed_task_details.append({
                        "task_name": task_name,
                        "error": error_msg
                    })
                    print(f"❌ {task_name} failed: {error_msg}")
            except Exception as e:
                failed_tasks += 1
                task_name = task_config['task_dir'].split('/')[-1]
                failed_task_details.append({
                    "task_name": task_name,
                    "error": str(e)
                })
                print(f"❌ {task_name} failed with exception: {e}")
    
    return successful_tasks, failed_tasks, failed_task_details

def main():
    parser = argparse.ArgumentParser(description="BlenderGym Runner for AgenticVerifier")
    
    # Dataset parameters
    parser.add_argument("--dataset-path", default="data/blendergym", help="Path to BlenderGym dataset root directory")
    parser.add_argument("--output-dir", default=f"output/blendergym/{time.strftime('%Y%m%d_%H%M%S')}", help="Output directory for results")
    
    # Task selection
    parser.add_argument("--task", choices=['all', 'blendshape', 'geometry', 'lighting', 'material', 'placement'], default='all', help="Specific task to run")
    parser.add_argument("--task-id", default=None, help="Specific task id to run (e.g., '1')")
    parser.add_argument("--test-id", default=None, help="Test ID to check for failed cases and retest them")
    
    # Main.py parameters
    parser.add_argument("--max-rounds", type=int, default=10, help="Maximum number of interaction rounds")
    parser.add_argument("--model", default="gpt-4o", help="OpenAI vision model to use")
    parser.add_argument("--memory-length", type=int, default=24, help="Memory length")
    
    # Blender parameters
    parser.add_argument("--blender-command", default="utils/infinigen/blender/blender", help="Blender command path")
    parser.add_argument("--blender-script", default="data/blendergym/generator_script.py", help="Blender execution script")
    parser.add_argument("--save-blender-file", action="store_true", help="Save blender file")
    
    # Tool server scripts (comma-separated)
    parser.add_argument("--generator-tools", default="tools/exec_blender.py,tools/generator_base.py,tools/initialize_plan.py", help="Comma-separated list of generator tool server scripts")
    parser.add_argument("--verifier-tools", default="tools/verifier_base.py,tools/investigator.py", help="Comma-separated list of verifier tool server scripts")
    
    # Parallel execution parameters
    parser.add_argument("--max-workers", type=int, default=8, help="Maximum number of parallel workers")
    parser.add_argument("--sequential", action="store_true", help="Run tasks sequentially instead of in parallel")
    parser.add_argument("--no-tools", action="store_true", help="Use no tools mode")
    parser.add_argument("--num-candidates", type=int, default=1, help="Number of candidates for the model")
    
    available_gpu_devices = os.getenv("CUDA_VISIBLE_DEVICES")
    if available_gpu_devices is None:
        available_gpu_devices = "0,1,2,3,4,5,6,7"
    parser.add_argument("--gpu-devices", default=available_gpu_devices, help="GPU devices for Blender")
    
    args = parser.parse_args()
    
    # Handle test-id logic
    if args.test_id is not None:
        args.output_dir = f"output/blendergym/{args.test_id}"
    
    # Normal execution - load dataset
    print(f"Loading BlenderGym dataset from: {args.dataset_path}")
    tasks = load_blendergym_dataset(args.dataset_path, args.task, args.test_id, args.task_id)
    
    if not tasks:
        print("No valid tasks found in dataset!")
        sys.exit(1)
    
    print(f"Found {len(tasks)} tasks")
    
    # Filter tasks if specific task specified
    if args.task != 'all':
        tasks = [t for t in tasks if t["task_name"] == args.task]
        print(f"Filtered to {len(tasks)} tasks for task: {args.task}")
    
    if not tasks:
        print("No tasks match the specified filters!")
        sys.exit(1)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save args to json
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(args.__dict__, f, indent=4, ensure_ascii=False)
    
    # Save task list for reference
    with open(os.path.join(args.output_dir, "tasks.json"), "w") as f:
        json.dump(tasks, f, indent=2)

    # Run tasks
    start_time = time.time()
    
    if args.sequential:
        # Sequential execution
        print("\nRunning tasks sequentially...")
        successful_tasks = 0
        failed_tasks = 0
        failed_task_details = []
        
        for i, task_config in enumerate(tasks, 1):
            print(f"\nTask {i}/{len(tasks)}")
            task_name, success, error_msg = run_blendergym_task(task_config, args)
            
            if success:
                successful_tasks += 1
            else:
                failed_tasks += 1
                failed_task_details.append({
                    "task_name": task_name,
                    "error": error_msg
                })
    else:
        # Parallel execution
        successful_tasks, failed_tasks, failed_task_details = run_tasks_parallel(
            tasks, args, max_workers=args.max_workers
        )
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Summary
    print(f"\n{'='*60}")
    print("EXECUTION SUMMARY")
    print(f"{'='*60}")
    print(f"Total tasks: {len(tasks)}")
    print(f"Successful: {successful_tasks}")
    print(f"Failed: {failed_tasks}")
    print(f"Execution time: {execution_time:.2f} seconds")
    print(f"Output directory: {args.output_dir}")
    
    # Save detailed results
    results = {
        "total_tasks": len(tasks),
        "successful_tasks": successful_tasks,
        "failed_tasks": failed_tasks,
        "execution_time_seconds": execution_time,
        "failed_task_details": failed_task_details,
        "execution_mode": "sequential" if args.sequential else f"parallel_{args.max_workers}_workers"
    }
    
    with open(os.path.join(args.output_dir, "execution_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    if successful_tasks > 0:
        print(f"\nResults saved to: {args.output_dir}")
        print("Check individual task directories for renders and thought processes.")
    
    if failed_tasks > 0:
        print(f"\nFailed tasks details saved to: {os.path.join(args.output_dir, 'execution_results.json')}")

if __name__ == "__main__":
    main()
