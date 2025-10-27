#!/usr/bin/env python3
"""
BlenderGym Hard Runner for AgenticVerifier
Loads BlenderGym Hard dataset and runs the dual-agent system for 3D scene generation.
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
from _api_keys import OPENAI_API_KEY, MESHY_API_KEY, VA_API_KEY, OPENAI_BASE_URL
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


def load_blendergym_hard_dataset(base_path: str, task_name: str, test_id: Optional[str] = None) -> List[Dict]:
    """
    Load BlenderGym Hard dataset structure.
    
    Args:
        base_path: Path to BlenderGym Hard dataset root
        task_name: Task name to load
        test_id: Optional test ID for filtering
    
    Returns:
        List of task configurations
    """
    base_path = Path(base_path)
    if not base_path.exists():
        print(f"Error: BlenderGym Hard dataset path not found: {base_path}")
        return []
    
    tasks = []
    
    # For blendergym hard, we typically have level1, level2, level3, level4
    if task_name == "all":
        # Load all available blendergym hard tasks
        task_dirs = [d for d in base_path.iterdir() if d.is_dir()]
        task_list = [d.name for d in task_dirs]
    else:
        task_list = [task_name]
    
    # Create task configurations
    for task in task_list:
        task_path = base_path / task
        if not task_path.exists():
            print(f"Warning: Task directory not found: {task_path}")
            continue
        
        print("task_path:", task_path)
            
        # Look for start code and renders
        start_code_path = task_path / "start.py"
        start_renders_dir = task_path / "renders" / "start"
        goal_renders_dir = task_path / "renders" / "goal"
        blender_file = task_path / "blender_file.blend"
        task_description_file = task_path / "task.txt"
        
        if not start_code_path.exists():
            print(f"Warning: start.py not found for task: {task}")
            continue
            
        if not goal_renders_dir.exists() or not start_renders_dir.exists():
            print(f"Warning: renders directory not found for task: {task}")
            continue
        
        if not blender_file.exists():
            print(f"Warning: blender_file.blend not found for task: {task}")
            continue
        
        # Read task description
        target_description = None
        if task_description_file.exists():
            with open(task_description_file, 'r') as f:
                target_description = f.read().strip()
        
        # Check for assets directory
        assets_path = task_path / "assets"
        os.makedirs(assets_path, exist_ok=True)
        assets_dir = str(assets_path)
        
        task_config = {
            "task_name": task,
            "task_id": task,
            "init_code_path": str(start_code_path),
            "init_image_path": str(start_renders_dir),
            "target_image_path": str(goal_renders_dir),
            "target_description": target_description,
            "blender_file": str(blender_file),
            "assets_dir": assets_dir,
            "output_dir": f"output/blendergym_hard/{test_id or time.strftime('%Y%m%d_%H%M%S')}/{task}",
        }
        
        tasks.append(task_config)
    
    return tasks


def run_blendergym_hard_task(task_config: Dict, args) -> tuple:
    """
    Run a single BlenderGym Hard task using main.py
    
    Args:
        task_config: Task configuration dictionary
        args: Command line arguments
    
    Returns:
        Tuple of (task_name, success, error_message)
    """
    task_name = task_config["task_name"]
    print(f"Running BlenderGym Hard task: {task_name}")
    
    # Create output directory
    os.makedirs(task_config["output_dir"], exist_ok=True)

    # Copy blender file to output directory
    import shutil
    output_blender_file = os.path.join(task_config["output_dir"], "blender_file.blend")
    shutil.copy(task_config["blender_file"], output_blender_file)
    
    # Build main.py command
    cmd = [
        sys.executable, "main.py",
        "--mode", "blendergym",
        "--model", args.model,
        "--api-key", args.api_key,
        "--api-base-url", args.api_base_url if args.api_base_url else "https://api.openai.com/v1",
        "--max-rounds", str(args.max_rounds),
        "--memory-length", str(args.memory_length),
        "--target-image-path", task_config["target_image_path"],
        "--output-dir", task_config["output_dir"],
        "--task-name", task_name,
        "--generator-tools", args.generator_tools,
        "--verifier-tools", args.verifier_tools,
        "--blender-command", args.blender_command,
        "--blender-file", output_blender_file,
        "--blender-script", args.blender_script,
        "--meshy_api_key", args.meshy_api_key,
        "--va_api_key", args.va_api_key,
        "--assets-dir", task_config["assets_dir"],
        "--init-code-path", task_config["init_code_path"],
        "--init-image-path", task_config["init_image_path"],
    ]
    
    if args.gpu_devices:
        cmd.extend(["--gpu-devices", args.gpu_devices])
    
    if task_config["target_description"]:
        cmd.extend(["--target-description", task_config["target_description"]])

    try:
        result = subprocess.run(cmd, check=False)  # 1 hour timeout
        
        if result.returncode == 0:
            print(f"✅ BlenderGym Hard task {task_name} completed successfully")
            return task_name, True, None
        else:
            error_msg = f"Task failed with return code {result.returncode}: {result.stderr}"
            print(f"❌ BlenderGym Hard task {task_name} failed: {error_msg}")
            return task_name, False, error_msg
            
    except subprocess.TimeoutExpired:
        error_msg = f"Task timed out after 1 hour"
        print(f"⏰ BlenderGym Hard task {task_name} timed out")
        return task_name, False, error_msg
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        print(f"💥 BlenderGym Hard task {task_name} failed with exception: {error_msg}")
        return task_name, False, error_msg


def run_blendergym_hard_tasks_parallel(tasks: List[Dict], args, max_workers: int = 4):
    """Run BlenderGym Hard tasks in parallel."""
    print(f"Running {len(tasks)} BlenderGym Hard tasks with {max_workers} workers")
    
    successful_tasks = 0
    failed_tasks = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(run_blendergym_hard_task, task_config, args): task_config 
            for task_config in tasks
        }
        
        # Process completed tasks
        for future in as_completed(future_to_task):
            task_config = future_to_task[future]
            try:
                task_name, success, error_msg = future.result()
                if success:
                    successful_tasks += 1
                else:
                    failed_tasks += 1
                    print(f"Failed task: {task_name} - {error_msg}")
            except Exception as e:
                failed_tasks += 1
                print(f"Task {task_config['task_name']} generated an exception: {e}")
    
    print(f"\nBlenderGym Hard task execution completed:")
    print(f"  Successful: {successful_tasks}")
    print(f"  Failed: {failed_tasks}")
    print(f"  Total: {len(tasks)}")


def main():
    parser = argparse.ArgumentParser(description="BlenderGym Hard Runner for AgenticVerifier")
    time_str = time.strftime('%Y%m%d_%H%M%S')
    
    # Dataset parameters
    parser.add_argument("--dataset-path", default="data/blendergym_hard", help="Path to BlenderGym Hard dataset root directory")
    parser.add_argument("--output-dir", default=f"output/blendergym_hard/{time_str}", help="Output directory for results")
    
    # Task selection
    parser.add_argument("--task", default="all", help="Specific task to run (default: all)")
    parser.add_argument("--test-id", help="Test ID for output directory naming")
    
    # Main.py parameters
    parser.add_argument("--max-rounds", type=int, default=100, help="Maximum number of interaction rounds")
    parser.add_argument("--model", default="gpt-5", help="OpenAI vision model to use")
    parser.add_argument("--api-base-url", default=OPENAI_BASE_URL, help="OpenAI-compatible API base URL")
    parser.add_argument("--api-key", default=OPENAI_API_KEY, help="OpenAI API key")
    parser.add_argument("--memory-length", type=int, default=12, help="Memory length")
    
    # Blender parameters
    parser.add_argument("--blender-command", default="utils/blender/infinigen/blender/blender", help="Blender command path")
    parser.add_argument("--blender-file", default="data/blendergym_hard/empty_scene.blend", help="Empty blender file for BlenderGym Hard")
    parser.add_argument("--blender-script", default="data/blendergym/pipeline_render_script_hard.py", help="Blender execution script")
    parser.add_argument("--blender-save", default=f"data/blendergym_hard/empty_scene.blend", help="Save blender file")
    
    # Tool server scripts (comma-separated)
    parser.add_argument("--generator-tools", default="tools/exec_blender.py,tools/meshy.py,tools/generator_base.py", help="Comma-separated list of generator tool server scripts")
    parser.add_argument("--verifier-tools", default="tools/verifier_base.py", help="Comma-separated list of verifier tool server scripts")
    
    # API keys
    parser.add_argument("--meshy_api_key", default=MESHY_API_KEY, help="Meshy API key")
    parser.add_argument("--va_api_key", default=VA_API_KEY, help="VA API key")
    
    # Execution parameters
    parser.add_argument("--max-workers", type=int, default=1, help="Maximum number of parallel workers")
    parser.add_argument("--gpu-devices", default=os.getenv("CUDA_VISIBLE_DEVICES"), help="GPU devices for Blender")
    
    args = parser.parse_args()
    
    # Validate API key
    if not args.api_key:
        print("Error: OpenAI API key is required. Set OPENAI_API_KEY environment variable or use --api-key")
        sys.exit(1)
    
    # Load dataset
    print(f"Loading BlenderGym Hard dataset from: {args.dataset_path}")
    tasks = load_blendergym_hard_dataset(args.dataset_path, args.task, args.test_id)
    
    if not tasks:
        print("No valid BlenderGym Hard tasks found in dataset!")
        sys.exit(1)
    
    print(f"Found {len(tasks)} BlenderGym Hard tasks")
    for task in tasks:
        print(f"  - {task['task_name']}: {task['init_code_path']}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run tasks
    if args.max_workers == 1:
        # Sequential execution
        successful_tasks = 0
        failed_tasks = 0
        
        for i, task_config in enumerate(tasks, 1):
            print(f"\nTask {i}/{len(tasks)}")
            task_name, success, error_msg = run_blendergym_hard_task(task_config, args)
            
            if success:
                successful_tasks += 1
            else:
                failed_tasks += 1
                print(f"Failed: {error_msg}")
        
        print(f"\nBlenderGym Hard task execution completed:")
        print(f"  Successful: {successful_tasks}")
        print(f"  Failed: {failed_tasks}")
        print(f"  Total: {len(tasks)}")
    else:
        # Parallel execution
        run_blendergym_hard_tasks_parallel(tasks, args, args.max_workers)


if __name__ == "__main__":
    main()
