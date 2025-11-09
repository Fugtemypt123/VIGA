#!/usr/bin/env python3
"""
AutoPresent Runner for AgenticVerifier
Loads AutoPresent dataset and runs the dual-agent system for 2D slides generation.
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

def check_failed_tasks(test_output_dir: str) -> List[Dict]:
    """
    Check for failed tasks in a test output directory.
    A task is considered failed if:
    1. The task directory doesn't exist
    2. The task directory is empty
    3. The task directory doesn't contain verifier_thoughts
    
    Args:
        test_output_dir: Path to the test output directory
        
    Returns:
        List of failed task configurations
    """
    failed_tasks = []
    test_output_path = Path(test_output_dir)
    task_list = ['art_photos', 'business', 'design', 'entrepreneur', 'environment', 'food', 'marketing', 'social_media', 'technology']
    
    if not test_output_path.exists():
        print(f"Error: Test output directory does not exist: {test_output_dir}")
        return failed_tasks
    
    print(f"Checking for failed tasks in: {test_output_dir}")
    
    # Look for task directories
    for task_type in task_list:
        for task_dir in (test_output_path / task_type).iterdir():
            if not task_dir.is_dir():
                continue
                
            task_name = task_dir.name
            verifier_thoughts_file = task_dir / "verifier_thoughts"
            
            # Check if task failed
            failed = False
            failure_reason = ""
            
            if not task_dir.exists():
                failed = True
                failure_reason = "Task directory does not exist"
            elif not any(task_dir.iterdir()):
                failed = True
                failure_reason = "Task directory is empty"
            elif not any(verifier_thoughts_file.iterdir()):
                failed = True
                failure_reason = "verifier_thoughts not found"
            
            if failed:
                print(f"❌ Found failed task: {task_name} - {failure_reason}")
                # Try to reconstruct task config from task name
                # Task name format is typically like "business/slide_1", "design/slide_2", etc.
                slides_part = task_name
                if slides_part.startswith('slide_'):
                    task_id = int(slides_part[6:])  # Remove 'slide_' prefix
                    failed_tasks.append({
                        "task_name": task_type,
                        "task_id": task_id,
                        "failure_reason": failure_reason
                    })
                else:
                    print(f"Warning: Could not parse slides part: {slides_part}")
        
    print(f"Found {len(failed_tasks)} failed tasks")
    return failed_tasks

def load_autopresent_dataset(base_path: str, task_name: str, task_id: Optional[str] = None) -> List[Dict]:
    """
    Load AutoPresent dataset structure.
    
    Args:
        base_path: Path to AutoPresent dataset root
        
    Returns:
        List of task configurations
    """
    tasks = []
    base_path = Path(base_path)
    
    if not base_path.exists():
        print(f"Error: AutoPresent dataset path does not exist: {base_path}")
        return tasks
    
    if task_name == 'all':
        task_list = ['art_photos', 'business', 'design', 'entrepreneur', 'environment', 'food', 'marketing', 'social_media', 'technology']
    else:
        task_list = [task_name]
        
    # If task_id is not None, only run the task_id
    if task_id is not None:
        task_dirs = [(base_path / task_name / f"slide_{task_id}", task_name)]
    # Otherwise, run all tasks in the task_list
    else:
        task_dirs = []
        for task in task_list:
            current_path = base_path / task
            for task_dir in current_path.glob("slide_*"):
                task_dirs.append((task_dir, task))
    
    for task_dir, task_name in task_dirs:
        # Check for required files
        start_code_path = task_dir / "start.py"
        start_image_path = task_dir / "start.jpg"
        target_description_file = task_dir / "instruction.txt"
        
        if not start_code_path.exists():
            print(f"Warning: start.py not found in {task_dir}")
            continue
            
        if not target_description_file.exists():
            print(f"Warning: target_description.txt not found in {task_dir}")
            continue
        target_description = None
        if target_description_file.exists():
            with open(target_description_file, 'r') as f:
                target_description = f.read().strip()
            
        task_config = {
            "task_name": task_name,
            "init_code_path": '',
            "init_image_path": '',
            "target_description": target_description,
            "resource_dir": str(task_dir),
        }
        tasks.append(task_config)
        print(f"Found task: {task_name}/{task_dir.name}")
    
    return tasks

def run_autopresent_task(task_config: Dict, args) -> tuple:
    """
    Run a single AutoPresent task using main.py
    
    Args:
        task_config: Task configuration dictionary
        args: Command line arguments
        
    Returns:
        Tuple of (task_name, success: bool, error_message: str)
    """
    task_name = task_config['task_name'] + "/" + task_config['resource_dir'].split('/')[-1]
    print(f"\n{'='*60}")
    print(f"Running task: {task_name}")
    print(f"{'='*60}")
    
    # Prepare output directories
    output_base = Path(args.output_dir + "/" + task_name)
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Build main.py command
    cmd = [
        sys.executable, "main.py",
        "--mode", "autopresent",
        "--model", args.model,
        "--api-key", get_model_info(args.model)["api_key"],
        "--api-base-url", get_model_info(args.model)["base_url"],
        "--max-rounds", str(args.max_rounds),
        "--memory-length", str(args.memory_length),
        "--task-name", task_config["task_name"],
        "--init-code-path", str(task_config["init_code_path"]),
        "--init-image-path", str(task_config["init_image_path"]),
        "--target-description", task_config["target_description"],
        "--output-dir", str(output_base),
        # Tool servers
        "--generator-tools", args.generator_tools,
        "--verifier-tools", args.verifier_tools,
        "--resource-dir", task_config["resource_dir"],
        "--clear-memory",
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
            executor.submit(run_autopresent_task, task_config, args): task_config 
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
                task_name = task_config['resource_dir'].split('/')[-1]
                failed_task_details.append({
                    "task_name": task_name,
                    "error": str(e)
                })
                print(f"❌ {task_name} failed with exception: {e}")
    
    return successful_tasks, failed_tasks, failed_task_details

def main():
    parser = argparse.ArgumentParser(description="AutoPresent Runner for AgenticVerifier")
    
    # Dataset parameters
    parser.add_argument("--dataset-path", default="data/autopresent/examples", help="Path to AutoPresent dataset root directory")
    parser.add_argument("--output-dir", default=f"output/autopresent/{time.strftime('%Y%m%d_%H%M%S')}", help="Output directory for results")
    
    # Task selection
    parser.add_argument("--task", default="all", choices=['all', 'art_photos', 'business', 'design', 'entrepreneur', 'environment', 'food', 'marketing', 'social_media', 'technology'], help="Specific task to run")
    parser.add_argument("--task-id", default=None, help="Specific task id to run (e.g., '1')")
    parser.add_argument("--test-id", default=None, help="Test ID to check for failed cases and retest them")
    
    # Main.py parameters
    parser.add_argument("--max-rounds", type=int, default=10, help="Maximum number of interaction rounds")
    parser.add_argument("--model", default="gpt-4o", help="OpenAI vision model to use")
    parser.add_argument("--memory-length", type=int, default=24, help="Memory length")
    
    # Tool server scripts (comma-separated)
    parser.add_argument("--generator-tools", default="tools/exec_slides.py,tools/generator_base.py", help="Comma-separated list of generator tool server scripts")
    
    # Verifier tool servers
    parser.add_argument("--verifier-tools", default="tools/verifier_base.py", help="Comma-separated list of verifier tool server scripts")
    
    # Parallel execution parameters
    parser.add_argument("--max-workers", type=int, default=8, help="Maximum number of parallel workers")
    parser.add_argument("--sequential", action="store_true", help="Run tasks sequentially instead of in parallel")
    parser.add_argument("--no-tools", action="store_true", help="Use no tools mode")
    
    args = parser.parse_args()
    
    # Handle test-id logic
    if args.test_id is not None:
        # Check for failed tasks in the specified test output directory
        test_output_dir = f"output/autopresent/{args.test_id}"
        failed_task_configs = check_failed_tasks(test_output_dir)
        
        if not failed_task_configs:
            print("No failed tasks found to retest!")
            sys.exit(0)
        
        print(f"Found {len(failed_task_configs)} failed tasks to retest")
        
        # Convert failed task configs to task configs for retesting
        tasks = []
        for failed_config in failed_task_configs:
            # Load the specific task from dataset
            retest_tasks = load_autopresent_dataset(args.dataset_path, failed_config["task_name"], failed_config["task_id"])
            if retest_tasks:
                tasks.extend(retest_tasks)
                print(f"Will retest: {failed_config['task_name']}/slide_{failed_config['task_id']}")
        
        if not tasks:
            print("No valid tasks found for retesting!")
            sys.exit(1)
    else:
        # Normal execution - load dataset
        print(f"Loading AutoPresent dataset from: {args.dataset_path}")
        tasks = load_autopresent_dataset(args.dataset_path, args.task, args.task_id)
        
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
    
    # Create output directory
    if args.test_id is not None:
        # For retesting, create a new output directory with retest suffix
        args.output_dir = f"output/autopresent/{args.test_id}"
        print(f"Retesting failed tasks. Use original output directory: {args.output_dir}")
    
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
            task_name, success, error_msg = run_autopresent_task(task_config, args)
            
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
        print("Check individual task directories for slides code and thought processes.")
    
    if failed_tasks > 0:
        print(f"\nFailed tasks details saved to: {os.path.join(args.output_dir, 'execution_results.json')}")

if __name__ == "__main__":
    main()
