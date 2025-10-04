#!/usr/bin/env python3
"""
Dynamic Scene Runner for AgenticVerifier
Loads dynamic scene dataset and runs the dual-agent system for 3D dynamic scene generation from scratch with animation/rigging.
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
from api_keys import OPENAI_API_KEY
import threading
import torch


def load_dynamic_scene_dataset(base_path: str, task_name: str, test_id: Optional[str] = None) -> List[Dict]:
    """
    Load dynamic scene dataset structure.
    
    Args:
        base_path: Path to dynamic scene dataset root
        task_name: Task name to load
        test_id: Optional test ID for filtering
    
    Returns:
        List of task configurations
    """
    base_path = Path(base_path)
    if not base_path.exists():
        print(f"Error: Dynamic scene dataset path not found: {base_path}")
        return []
    
    tasks = []
    
    # For dynamic scenes, we typically have target animations/videos and descriptions
    if task_name == "all":
        # Load all available dynamic scene tasks
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
            
        # Look for target animation/video
        target_image_path = None
        if (task_path / "target.mp4").exists():
            target_image_path = str(task_path / "target.mp4")
        elif (task_path / "target.gif").exists():
            target_image_path = str(task_path / "target.gif")
        elif (task_path / "target.png").exists():
            target_image_path = str(task_path / "target.png")
        elif (task_path / "target.jpg").exists():
            target_image_path = str(task_path / "target.jpg")
        elif (task_path / "target").exists() and (task_path / "target").is_dir():
            target_image_path = str(task_path / "target")
        
        if not target_image_path:
            print(f"Warning: No target animation/video found for task: {task}")
            continue
        
        # Look for description file
        description_path = task_path / "description.txt"
        target_description = None
        if description_path.exists():
            with open(description_path, 'r') as f:
                target_description = f.read().strip()
        
        # Look for animation parameters
        animation_config_path = task_path / "animation_config.json"
        animation_config = {}
        if animation_config_path.exists():
            with open(animation_config_path, 'r') as f:
                animation_config = json.load(f)
        
        task_config = {
            "task_name": task,
            "task_id": task,
            "target_image_path": target_image_path,
            "target_description": target_description,
            "animation_config": animation_config,
            "output_dir": f"output/dynamic_scene/{test_id or time.strftime('%Y%m%d_%H%M%S')}/{task}",
            "init_code_path": None,  # Dynamic scenes start from scratch
            "init_image_path": None,  # No initial scene
        }
        
        tasks.append(task_config)
    
    return tasks


def run_dynamic_scene_task(task_config: Dict, args) -> tuple:
    """
    Run a single dynamic scene task using main.py
    
    Args:
        task_config: Task configuration dictionary
        args: Command line arguments
    
    Returns:
        Tuple of (task_name, success, error_message)
    """
    task_name = task_config["task_name"]
    task_id = task_config["task_id"]
    
    print(f"Running dynamic scene task: {task_name}")
    
    # Create output directory
    os.makedirs(task_config["output_dir"], exist_ok=True)
    
    # Build main.py command
    cmd = [
        sys.executable, "main.py",
        "--mode", "dynamic_scene",
        "--vision-model", args.vision_model,
        "--api-key", args.api_key,
        "--openai-base-url", args.openai_base_url if args.openai_base_url else "https://api.openai.com/v1",
        "--max-rounds", str(args.max_rounds),
        "--target-image-path", task_config["target_image_path"],
        "--output-dir", task_config["output_dir"],
        "--task-name", task_name,
        "--generator-script", args.generator_script,
        "--verifier-script", args.verifier_script,
        "--blender-server-path", args.blender_server_path,
        "--blender-command", args.blender_command,
        "--blender-file", args.blender_file,
        "--blender-script", args.blender_script,
        "--scene-server-path", args.scene_server_path,
        "--meshy_api_key", args.meshy_api_key,
        "--va_api_key", args.va_api_key,
    ]
    
    if args.save_blender_file:
        cmd.append("--save-blender-file")
    
    if args.gpu_devices:
        cmd.extend(["--gpu-devices", args.gpu_devices])
    
    if task_config["target_description"]:
        cmd.extend(["--target-description", task_config["target_description"]])
    
    # Add animation-specific parameters
    if task_config["animation_config"]:
        animation_config_path = os.path.join(task_config["output_dir"], "animation_config.json")
        with open(animation_config_path, 'w') as f:
            json.dump(task_config["animation_config"], f)
        cmd.extend(["--animation-config", animation_config_path])
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)  # 2 hour timeout for dynamic scenes
        
        if result.returncode == 0:
            print(f"✅ Dynamic scene task {task_name} completed successfully")
            return task_name, True, None
        else:
            error_msg = f"Task failed with return code {result.returncode}: {result.stderr}"
            print(f"❌ Dynamic scene task {task_name} failed: {error_msg}")
            return task_name, False, error_msg
            
    except subprocess.TimeoutExpired:
        error_msg = f"Task timed out after 2 hours"
        print(f"⏰ Dynamic scene task {task_name} timed out")
        return task_name, False, error_msg
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        print(f"💥 Dynamic scene task {task_name} failed with exception: {error_msg}")
        return task_name, False, error_msg


def run_dynamic_scene_tasks_parallel(tasks: List[Dict], args, max_workers: int = 2):
    """Run dynamic scene tasks in parallel (fewer workers due to higher resource requirements)."""
    print(f"Running {len(tasks)} dynamic scene tasks with {max_workers} workers")
    
    successful_tasks = 0
    failed_tasks = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(run_dynamic_scene_task, task_config, args): task_config 
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
    
    print(f"\nDynamic scene task execution completed:")
    print(f"  Successful: {successful_tasks}")
    print(f"  Failed: {failed_tasks}")
    print(f"  Total: {len(tasks)}")


def main():
    parser = argparse.ArgumentParser(description="Dynamic Scene Runner for AgenticVerifier")
    
    # Dataset parameters
    parser.add_argument("--dataset-path", default="data/dynamic_scene", help="Path to dynamic scene dataset root directory")
    parser.add_argument("--output-dir", default=f"output/dynamic_scene/{time.strftime('%Y%m%d_%H%M%S')}", help="Output directory for results")
    
    # Task selection
    parser.add_argument("--task", default="all", help="Specific task to run (default: all)")
    parser.add_argument("--test-id", help="Test ID for output directory naming")
    
    # Main.py parameters
    parser.add_argument("--max-rounds", type=int, default=15, help="Maximum number of interaction rounds (higher for dynamic scenes)")
    parser.add_argument("--vision-model", default="gpt-4o", help="OpenAI vision model to use")
    parser.add_argument("--openai-base-url", default=os.getenv("OPENAI_BASE_URL"), help="OpenAI-compatible API base URL")
    parser.add_argument("--api-key", default=OPENAI_API_KEY, help="OpenAI API key")
    
    # Blender parameters
    parser.add_argument("--blender-server-path", default="servers/generator/blender.py", help="Path to Blender MCP server script")
    parser.add_argument("--blender-command", default="utils/blender/infinigen/blender/blender", help="Blender command path")
    parser.add_argument("--blender-file", default="data/dynamic_scene/empty_scene.blend", help="Empty blender file for dynamic scenes")
    parser.add_argument("--blender-script", default="data/dynamic_scene/pipeline_render_script.py", help="Blender execution script")
    parser.add_argument("--save-blender-file", action="store_true", help="Save blender file")
    
    # Tool server paths
    parser.add_argument("--generator-script", default="agents/generator.py", help="Generator MCP script path")
    parser.add_argument("--verifier-script", default="agents/verifier.py", help="Verifier MCP script path")
    parser.add_argument("--scene-server-path", default="servers/verifier/scene.py", help="Scene server path")
    
    # API keys
    parser.add_argument("--meshy_api_key", default=os.getenv("MESHY_API_KEY"), help="Meshy API key")
    parser.add_argument("--va_api_key", default=os.getenv("VA_API_KEY"), help="VA API key")
    
    # Execution parameters
    parser.add_argument("--max-workers", type=int, default=2, help="Maximum number of parallel workers (lower for dynamic scenes)")
    parser.add_argument("--gpu-devices", default=os.getenv("CUDA_VISIBLE_DEVICES"), help="GPU devices for Blender")
    
    args = parser.parse_args()
    
    # Validate API key
    if not args.api_key:
        print("Error: OpenAI API key is required. Set OPENAI_API_KEY environment variable or use --api-key")
        sys.exit(1)
    
    # Load dataset
    print(f"Loading dynamic scene dataset from: {args.dataset_path}")
    tasks = load_dynamic_scene_dataset(args.dataset_path, args.task, args.test_id)
    
    if not tasks:
        print("No valid dynamic scene tasks found in dataset!")
        sys.exit(1)
    
    print(f"Found {len(tasks)} dynamic scene tasks")
    for task in tasks:
        print(f"  - {task['task_name']}: {task['target_image_path']}")
        if task['animation_config']:
            print(f"    Animation config: {task['animation_config']}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run tasks
    if args.max_workers == 1:
        # Sequential execution
        successful_tasks = 0
        failed_tasks = 0
        
        for i, task_config in enumerate(tasks, 1):
            print(f"\nTask {i}/{len(tasks)}")
            task_name, success, error_msg = run_dynamic_scene_task(task_config, args)
            
            if success:
                successful_tasks += 1
            else:
                failed_tasks += 1
                print(f"Failed: {error_msg}")
        
        print(f"\nDynamic scene task execution completed:")
        print(f"  Successful: {successful_tasks}")
        print(f"  Failed: {failed_tasks}")
        print(f"  Total: {len(tasks)}")
    else:
        # Parallel execution
        run_dynamic_scene_tasks_parallel(tasks, args, args.max_workers)


if __name__ == "__main__":
    main()
