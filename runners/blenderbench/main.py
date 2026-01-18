#!/usr/bin/env python3
"""
BlenderStudio Runner for AgenticVerifier
Loads BlenderStudio dataset and runs the dual-agent system for 3D scene generation.
"""
import os
import sys
import time
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess

# add runners directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.common import get_image_base64, extract_code_pieces, build_client

prompt = """You are the BlenderGymGenerator—a professional Blender code agent responsible for converting an initial 3D scene into a target scene and generating it based on a provided target image. You will receive: (1) initial Python code to set up the current scene; (2) initial images displaying the current scene; and (3) target images displaying the target scene. (4) task description that describes what you need to do. Your task is to modify the code to transform the initial scene into the target scene. Please output the complete modified code."""

def load_blenderstudio_dataset(base_path: str, task_name: str, test_id: Optional[str] = None) -> List[Dict]:
    """
    Load BlenderStudio dataset structure.

    Args:
        base_path: Path to BlenderStudio dataset root.
        task_name: Name of the task type to load.
        test_id: Optional test ID for filtering completed tasks.

    Returns:
        List of task configurations.
    """
    tasks = []
    base_path = Path(base_path)
    
    if not base_path.exists():
        print(f"Error: BlenderStudio dataset path does not exist: {base_path}")
        return tasks
    
    if task_name == 'all':
        task_list = ['level1', 'level2', 'level3']
    else:
        task_list = [task_name]
        
    current_task_path = Path(f'output/blenderstudio/{test_id}')
    current_task_dirs = []
    for task in task_list:
        for task_dir in current_task_path.glob(f"{task}*"):
            current_task_dir = task_dir / "scores.json"
            if os.path.exists(current_task_dir):
                current_task_dirs.append(os.path.basename(task_dir))
                
    task_dirs = []
    for task in task_list:
        for task_dir in base_path.glob(f"{task}/*"):
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
        tasks.append(task_config)
        print(f"Found task: {task_name}/{task_dir.name}")
    
    return tasks

def run_blenderstudio_task(task_config: Dict, args: argparse.Namespace) -> Tuple[str, bool, str]:
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
    output_base = os.path.join(task_config['task_dir'], "baseline")
    os.makedirs(output_base, exist_ok=True)
    
    messages=[{"role": "system", "content": prompt}, {"role": "user", "content": [{"type": "text", "text": "Initial code: \n" + open(task_config['init_code_path']).read()}]}]
    
    messages[1]["content"].append({"type": "text", "text": "Initial images: "})
    for image_path in os.listdir(task_config['init_image_path']):
        messages[1]["content"].append({"type": "image_url", "image_url": {"url": get_image_base64(os.path.join(task_config['init_image_path'], image_path))}})
    messages[1]["content"].append({"type": "text", "text": "Target images: "})
    for image_path in os.listdir(task_config['target_image_path']):
        messages[1]["content"].append({"type": "image_url", "image_url": {"url": get_image_base64(os.path.join(task_config['target_image_path'], image_path))}})
    
    client = build_client(args.model)
    response = client.chat.completions.create(model=args.model, messages=messages)
    
    response_text = response.choices[0].message.content.strip()
    code_pieces = extract_code_pieces(response_text)
    output_name = args.model.replace('-', '_').replace('.', '_')
    
    with open(os.path.join(output_base, f"{output_name}.py"), "w") as f:
        f.write(code_pieces)
    print(f"Saving code to {os.path.join(output_base, f'{output_name}.py')}")
        
    cmd = [
        "utils/infinigen/blender/blender",
        "--background", task_config['blender_file'],
        "--python", "data/blenderstudio/generator_script.py",
        "--", os.path.join(output_base, f"{output_name}.py"), os.path.join(output_base, f"{output_name}")
    ]
    subprocess.run(cmd, check=True)
    print(f"Running blender command: {cmd}")
    return task_name, True, ""
    

def run_tasks_parallel(tasks: List[Dict], args: argparse.Namespace, max_workers: int = 10) -> Tuple[int, int, List[Dict]]:
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
            executor.submit(run_blenderstudio_task, task_config, args): task_config 
            for task_config in tasks
        }
        
        # Process completed tasks
        for future in as_completed(future_to_task):
            task_config = future_to_task[future]
            try:
                task_name, success, error_msg = future.result()
                if success:
                    successful_tasks += 1
                    print(f"{task_name} completed successfully")
                else:
                    failed_tasks += 1
                    failed_task_details.append({
                        "task_name": task_name,
                        "error": error_msg
                    })
                    print(f"{task_name} failed: {error_msg}")
            except Exception as e:
                failed_tasks += 1
                task_name = task_config['task_dir'].split('/')[-1]
                failed_task_details.append({
                    "task_name": task_name,
                    "error": str(e)
                })
                print(f"{task_name} failed with exception: {e}")
    
    return successful_tasks, failed_tasks, failed_task_details

def main() -> None:
    """Entry point for the BlenderBench baseline runner."""
    parser = argparse.ArgumentParser(description="BlenderGym Runner for AgenticVerifier")
    
    # Dataset parameters
    parser.add_argument("--dataset-path", default="data/blenderstudio", help="Path to BlenderGym dataset root directory")
    
    # Task selection
    parser.add_argument("--task", choices=['all', 'level1', 'level2', 'level3'], default='all', help="Specific task to run")
    parser.add_argument("--model", default="gpt-4o", help="OpenAI vision model to use")
    
    # Parallel execution parameters
    parser.add_argument("--max-workers", type=int, default=8, help="Maximum number of parallel workers")
    parser.add_argument("--sequential", action="store_true", help="Run tasks sequentially instead of in parallel")
    
    available_gpu_devices = os.getenv("CUDA_VISIBLE_DEVICES")
    if available_gpu_devices is None:
        available_gpu_devices = "0,1,2,3,4,5,6,7"
    parser.add_argument("--gpu-devices", default=available_gpu_devices, help="GPU devices for Blender")
    
    args = parser.parse_args()
    
    # Normal execution - load dataset
    print(f"Loading BlenderStudio dataset from: {args.dataset_path}")
    tasks = load_blenderstudio_dataset(args.dataset_path, args.task, None)
    
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
            task_name, success, error_msg = run_blenderstudio_task(task_config, args)
            
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
    
    # Save detailed results
    results = {
        "total_tasks": len(tasks),
        "successful_tasks": successful_tasks,
        "failed_tasks": failed_tasks,
        "execution_time_seconds": execution_time,
        "failed_task_details": failed_task_details,
        "execution_mode": "sequential" if args.sequential else f"parallel_{args.max_workers}_workers"
    }
    

if __name__ == "__main__":
    main()
