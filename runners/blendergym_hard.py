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
import shutil
from pathlib import Path
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

def check_failed_tasks(test_output_dir: str) -> List[Dict]:
    """
    Check for failed tasks in a test output directory.
    A task is considered failed if:
    1. The task directory doesn't exist
    2. The task directory is empty
    3. The task directory doesn't contain generator_thoughts.json
    
    Args:
        test_output_dir: Path to the test output directory
        
    Returns:
        List of failed task configurations
    """
    failed_tasks = []
    test_output_path = Path(test_output_dir)
    
    if not test_output_path.exists():
        print(f"Error: Test output directory does not exist: {test_output_dir}")
        return failed_tasks
    
    print(f"Checking for failed tasks in: {test_output_dir}")
    
    # Look for task directories
    for task_dir in test_output_path.iterdir():
        if not task_dir.is_dir():
            continue
            
        task_name = task_dir.name
        
        # remove '_evaluation' file, it is not a task
        if '_evaluation' in task_name:
            continue
        
        scores_file = task_dir / "scores.json"
        
        # Check if task failed
        failed = False
        failure_reason = ""
        
        if not task_dir.exists():
            failed = True
            failure_reason = "Task directory does not exist"
        elif not any(task_dir.iterdir()):
            failed = True
            failure_reason = "Task directory is empty"
        elif not scores_file.exists():
            failed = True
            failure_reason = "scores.json not found"
        
        if failed:
            print(f"❌ Found failed task: {task_name} - {failure_reason}")
            # Try to reconstruct task config from task name
            # Task name format is typically like "blendshape1", "geometry2", etc.
            task_type = None
            task_id = None
            
            for task_type_name in ['blendshape', 'geometry', 'lighting', 'material', 'placement']:
                if task_name.startswith(task_type_name):
                    task_type = task_type_name
                    task_id = task_name[len(task_type_name):]
                    break
            
            if task_type and task_id:
                failed_tasks.append({
                    "task_name": task_type,
                    "task_id": task_id,
                    "failure_reason": failure_reason
                })
            else:
                print(f"Warning: Could not parse task name: {task_name}")
    
    print(f"Found {len(failed_tasks)} failed tasks")
    return failed_tasks

def load_blendergym_dataset(base_path: str, task_name: str, task_id: Optional[str] = None) -> List[Dict]:
    """
    Load BlenderGym dataset structure.
    
    Args:
        base_path: Path to BlenderGym dataset root
        
    Returns:
        List of task configurations
    """
    tasks = []
    base_path = Path(base_path)
    
    if not base_path.exists():
        print(f"Error: BlenderGym dataset path does not exist: {base_path}")
        return tasks
    
    if task_name == 'all':
        task_list = ['level1', 'level2', 'level3']
    else:
        task_list = [task_name]
        
    # If task_id is not None, only run the task_id
    if task_id is not None:
        task_dirs = [(base_path /  f"{task_name}{task_id}", task_name)]
    # Otherwise, run all tasks in the task_list
    else:
        task_dirs = []
        for task in task_list:
            task_path = base_path / task
            for task_dir in task_path.glob("*"):
                task_name = task + '-' + str(task_dir.name.split('/')[-1][-1])
                task_dirs.append((task_dir, task_name))
    
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
        tasks.append(task_config)
        print(f"Found task: {task_name}")
    
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
    level = task_name.split('-')[0]
    print(f"\n{'='*60}")
    print(f"Running task: {task_name}")
    print(f"{'='*60}")
    
    # Prepare output directories
    output_base = Path(args.output_dir + "/" + task_name)
    
    # Create directories
    output_base.mkdir(parents=True, exist_ok=True)
    
    # copy blender file to output directory
    shutil.copy(task_config["blender_file"], output_base / "blender_file.blend")
    
    # Build main.py command
    cmd = [
        sys.executable, "main.py",
        "--mode", "blendergym-hard",
        "--vision-model", args.vision_model,
        "--api-key", args.api_key,
        "--openai-base-url", args.openai_base_url if args.openai_base_url else "https://api.openai.com/v1",
        "--max-rounds", str(args.max_rounds),
        "--task-name", task_config["task_name"],
        "--init-code-path", str(task_config["init_code_path"]),
        "--init-image-path", str(task_config["init_image_path"]),
        "--target-image-path", str(task_config["target_image_path"]),
        "--output-dir", str(output_base),
        # Agent server paths
        "--generator-script", args.generator_script,
        "--verifier-script", args.verifier_script,
        # Blender execution parameters (for generator)
        "--blender-server-path", args.blender_server_path,
        "--blender-command", args.blender_command,
        "--blender-file", str(output_base / "blender_file.blend"),
        "--blender-script", f'data/blendergym_hard/{level}/pipeline_render_script.py',
        # Tool server paths (for verifier)
        "--image-server-path", args.image_server_path,
        "--scene-server-path", args.scene_server_path,
    ]
    
    if args.save_blender_file:
        cmd.append("--save-blender-file")
    
    print(f"Command: {' '.join(cmd)}")
    
    try:
        # Run the command
        result = subprocess.run(cmd, check=True, capture_output=False, timeout=3600)  # 1 hour timeout
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
    parser.add_argument("--dataset-path", default="data/blendergym_hard", help="Path to BlenderGym dataset root directory")
    parser.add_argument("--output-dir", default=f"output/blendergym_hard/{time.strftime('%Y%m%d_%H%M%S')}", help="Output directory for results")
    
    # Task selection
    parser.add_argument("--task", choices=['all', 'level1', 'level2', 'level3'], default='all', help="Specific task to run")
    parser.add_argument("--task-id", default=None, help="Specific task id to run (e.g., '1')")
    parser.add_argument("--test-id", default=None, help="Test ID to check for failed cases and retest them")
    
    # Main.py parameters
    parser.add_argument("--max-rounds", type=int, default=10, help="Maximum number of interaction rounds")
    parser.add_argument("--vision-model", default="gpt-4o", help="OpenAI vision model to use")
    parser.add_argument("--openai-base-url", default=os.getenv("OPENAI_BASE_URL"), help="OpenAI-compatible API base URL")
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"), help="OpenAI API key")
    
    # Blender parameters
    parser.add_argument("--blender-server-path", default="servers/generator/blender.py", help="Path to Blender MCP server script")
    parser.add_argument("--blender-command", default="utils/blender/infinigen/blender/blender", help="Blender command path")
    parser.add_argument("--blender-script", default="data/blendergym_hard/pipeline_render_script.py", help="Blender execution script")
    parser.add_argument("--save-blender-file", default=True, action="store_true", help="Save blender file")
    
    # Tool server paths
    parser.add_argument("--generator-script", default="agents/generator_mcp.py", help="Generator MCP script path")
    parser.add_argument("--verifier-script", default="agents/verifier_mcp.py", help="Verifier MCP script path")
    parser.add_argument("--image-server-path", default="servers/verifier/image.py", help="Path to image processing MCP server script")
    parser.add_argument("--scene-server-path", default="servers/verifier/scene.py", help="Path to scene investigation MCP server script")
    
    # Parallel execution parameters
    parser.add_argument("--max-workers", type=int, default=10, help="Maximum number of parallel workers")
    parser.add_argument("--sequential", action="store_true", help="Run tasks sequentially instead of in parallel")
    
    args = parser.parse_args()
    
    # Handle test-id logic
    if args.test_id is not None:
        # Check for failed tasks in the specified test output directory
        test_output_dir = f"output/blendergym_hard/{args.test_id}"
        failed_task_configs = check_failed_tasks(test_output_dir)
        
        if not failed_task_configs:
            print("No failed tasks found to retest!")
            sys.exit(0)
        
        print(f"Found {len(failed_task_configs)} failed tasks to retest")
        
        # Convert failed task configs to task configs for retesting
        tasks = []
        for failed_config in failed_task_configs:
            # Load the specific task from dataset
            retest_tasks = load_blendergym_dataset(args.dataset_path, failed_config["task_name"], failed_config["task_id"])
            if retest_tasks:
                tasks.extend(retest_tasks)
                print(f"Will retest: {failed_config['task_name']}{failed_config['task_id']}")
        
        if not tasks:
            print("No valid tasks found for retesting!")
            sys.exit(1)
    else:
        # Normal execution - load dataset
        print(f"Loading BlenderGym-hard dataset from: {args.dataset_path}")
        tasks = load_blendergym_dataset(args.dataset_path, args.task, args.task_id)
        
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
    
    # Create output directory (currently use the original output directory)
    if args.test_id is not None:
        # For retesting, create a new output directory with retest suffix
        args.output_dir = f"output/blendergym_hard/{args.test_id}"
        print(f"Retesting failed tasks. Use original output directory: {args.output_dir}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save args to json
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(args.__dict__, f, indent=2)
    
    # Save task list for reference
    with open(os.path.join(args.output_dir, "tasks.json"), "w") as f:
        json.dump(tasks, f, indent=2)
        
    tasks = tasks[:1]
    
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
