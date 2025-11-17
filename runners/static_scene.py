#!/usr/bin/env python3
"""
Static Scene Runner for AgenticVerifier
Loads static scene dataset and runs the dual-agent system for 3D static scene generation from scratch.
"""
import os
import sys
import time
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import shutil

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.common import get_model_info, get_meshy_info

def load_static_scene_dataset(base_path: str, task_name: str, setting: str, test_id: Optional[str] = None) -> List[Dict]:
    """
    Load static scene dataset structure.
    
    Args:
        base_path: Path to static scene dataset root
        task_name: Task name to load
        test_id: Optional test ID for filtering

    Returns:
        List of task configurations
    """
    base_path = Path(base_path)
    if not base_path.exists():
        print(f"Error: Static scene dataset path not found: {base_path}")
        return []
    
    tasks = []
    
    # For static scenes, we typically have target images and descriptions
    if task_name == "all":
        # Load all available static scene tasks
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
            
        # Look for target images
        target_image_path = None
        if (task_path / "target.png").exists():
            target_image_path = str(task_path / "target.png")
        elif (task_path / "target.jpg").exists():
            target_image_path = str(task_path / "target.jpg")
        elif (task_path / "target").exists() and (task_path / "target").is_dir():
            target_image_path = str(task_path / "target")
        
        if not target_image_path:
            print(f"Warning: No target image found for task: {task}")
            continue
        
        init_image_path = task_path / f"{setting}_init" / "render1.png"
        
        if not init_image_path.exists():
            init_image_path = ''
        
        # Look for description file
        description_path = task_path / "description.txt"
        target_description = None
        if description_path.exists():
            with open(description_path, 'r') as f:
                target_description = f.read().strip()
        
        # Check for assets directory
        assets_path = task_path / "assets"
        os.makedirs(assets_path, exist_ok=True)
        assets_dir = str(assets_path)
        
        task_config = {
            "task_name": task,
            "task_id": task,
            "target_image_path": target_image_path,
            "assets_dir": assets_dir,  # Add assets directory path
            "output_dir": f"output/static_scene/{test_id or time.strftime('%Y%m%d_%H%M%S')}/{task}",
            "init_code_path": "",  # Static scenes start from scratch
            "init_image_path": str(init_image_path),
        }
        
        tasks.append(task_config)
    
    return tasks


def run_static_scene_task(task_config: Dict, args) -> tuple:
    """
    Run a single static scene task using main.py
    
    Args:
        task_config: Task configuration dictionary
        args: Command line arguments
    
    Returns:
        Tuple of (task_name, success, error_message)
    """
    task_name = task_config["task_name"]
    print(f"Running static scene task: {task_name}")
    
    # Create output directory
    os.makedirs(task_config["output_dir"], exist_ok=True)

    # Create an empty blender file inside output_dir for build-from-scratch flows
    created_blender_file = os.path.join(task_config["output_dir"], "blender_file.blend")
    # copy the blender file to the output directory
    if os.path.exists(args.blender_file):
        shutil.copy(args.blender_file, created_blender_file)
    else:
        # Create a new blender file
        create_empty_blend_cmd = (
            f"{args.blender_command} --background --factory-startup "
            f"--python-expr \"import bpy; bpy.ops.wm.read_factory_settings(use_empty=True); bpy.ops.wm.save_mainfile(filepath='" + created_blender_file + "')\""
        )
        subprocess.run(create_empty_blend_cmd, shell=True, check=True)
    
    # Build main.py command
    cmd = [
        sys.executable, "main.py",
        "--mode", "static_scene",
        "--model", args.model,
        "--api-key", get_model_info(args.model)["api_key"],
        "--api-base-url", get_model_info(args.model)["base_url"],
        "--max-rounds", str(args.max_rounds),
        "--memory-length", str(args.memory_length),
        "--target-image-path", task_config["target_image_path"] if not args.text_only else "",
        "--output-dir", task_config["output_dir"],
        "--task-name", task_name,
        "--generator-tools", args.generator_tools,
        "--verifier-tools", args.verifier_tools,
        "--blender-command", args.blender_command,
        "--blender-file", created_blender_file,
        "--blender-script", args.blender_script,
        "--meshy_api_key", get_meshy_info()["meshy_api_key"],
        "--va_api_key", get_meshy_info()["va_api_key"],
        "--blender-save", created_blender_file,
        "--assets-dir", task_config["assets_dir"],
        "--init-code-path", task_config["init_code_path"],
        "--init-image-path", task_config["init_image_path"],
        "--clear-memory",
    ]
    
    if args.explicit_comp:
        cmd.extend(["--explicit-comp"])
    if args.gpu_devices:
        cmd.extend(["--gpu-devices", args.gpu_devices])
    if args.with_prior:
        cmd.extend(["--with-prior"])
    if "target_description" in task_config:
        cmd.extend(["--target-description", task_config["target_description"]])

    try:
        result = subprocess.run(cmd)  # no timeout
        
        if result.returncode == 0:
            print(f"✅ Static scene task {task_name} completed successfully")
            return task_name, True, None
        else:
            error_msg = f"Task failed with return code {result.returncode}: {result.stderr}"
            print(f"❌ Static scene task {task_name} failed: {error_msg}")
            return task_name, False, error_msg
            
    except subprocess.TimeoutExpired:
        error_msg = f"Task timed out after 1 hour"
        print(f"⏰ Static scene task {task_name} timed out")
        return task_name, False, error_msg
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        print(f"💥 Static scene task {task_name} failed with exception: {error_msg}")
        return task_name, False, error_msg


def run_static_scene_tasks_parallel(tasks: List[Dict], args, max_workers: int = 4):
    """Run static scene tasks in parallel."""
    print(f"Running {len(tasks)} static scene tasks with {max_workers} workers")
    
    successful_tasks = 0
    failed_tasks = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(run_static_scene_task, task_config, args): task_config 
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
    
    print(f"\nStatic scene task execution completed:")
    print(f"  Successful: {successful_tasks}")
    print(f"  Failed: {failed_tasks}")
    print(f"  Total: {len(tasks)}")


def main():
    parser = argparse.ArgumentParser(description="Static Scene Runner for AgenticVerifier")
    time_str = time.strftime('%Y%m%d_%H%M%S')
    
    # Dataset parameters
    parser.add_argument("--dataset-path", default="data/static_scene", help="Path to static scene dataset root directory")
    parser.add_argument("--output-dir", default=f"output/static_scene/{time_str}", help="Output directory for results")
    
    # Task selection
    parser.add_argument("--task", default="all", help="Specific task to run (default: all)")
    parser.add_argument("--test-id", help="Test ID for output directory naming")
    
    # Main.py parameters
    parser.add_argument("--max-rounds", type=int, default=100, help="Maximum number of interaction rounds")
    parser.add_argument("--model", default="gpt-5", help="OpenAI vision model to use")
    parser.add_argument("--memory-length", type=int, default=12, help="Memory length")
    
    # Blender parameters
    parser.add_argument("--blender-command", default="utils/infinigen/blender/blender", help="Blender command path")
    parser.add_argument("--blender-file", default="data/static_scene/empty_scene.blend", help="Empty blender file for static scenes")
    parser.add_argument("--blender-script", default="data/static_scene/generator_init_script.py", help="Blender execution script")
    parser.add_argument("--blender-save", default=f"data/static_scene/empty_scene.blend", help="Save blender file")
    
    # Tool server scripts (comma-separated)
    parser.add_argument("--generator-tools", default="tools/exec_blender.py,tools/generator_base.py,tools/meshy.py, tools/initialize_plan.py", help="Comma-separated list of generator tool server scripts")
    parser.add_argument("--verifier-tools", default="tools/investigator.py,tools/verifier_base.py", help="Comma-separated list of verifier tool server scripts")
    
    # Execution parameters
    parser.add_argument("--max-workers", type=int, default=1, help="Maximum number of parallel workers")
    parser.add_argument("--gpu-devices", default=os.getenv("CUDA_VISIBLE_DEVICES"), help="GPU devices for Blender")
    
    # Additional parameters
    parser.add_argument("--explicit-comp", action="store_true", help="Enable explicit completion")
    parser.add_argument("--text-only", action="store_true", help="Only use text as reference")
    parser.add_argument("--setting", choices=["minimal", "reasonable"], default="minimal", help="Setting for the static scene task")
    parser.add_argument("--with-prior", action="store_true", help="Enable with prior")
    
    args = parser.parse_args()
    
    # Load dataset
    print(f"Loading static scene dataset from: {args.dataset_path}")
    tasks = load_static_scene_dataset(args.dataset_path, args.task, args.setting, args.test_id)
    
    if not tasks:
        print("No valid static scene tasks found in dataset!")
        sys.exit(1)
    
    print(f"Found {len(tasks)} static scene tasks")
    for task in tasks:
        print(f"  - {task['task_name']}: {task['target_image_path']}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(args.output_dir + "/args.json", "w") as f:
        json.dump(args.__dict__, f, indent=4, ensure_ascii=False)
    
    # Run tasks
    start_time = time.time()
    
    # Run tasks
    if args.max_workers == 1:
        # Sequential execution
        successful_tasks = 0
        failed_tasks = 0
        
        for i, task_config in enumerate(tasks, 1):
            print(f"\nTask {i}/{len(tasks)}")
            task_name, success, error_msg = run_static_scene_task(task_config, args)
            
            if success:
                successful_tasks += 1
            else:
                failed_tasks += 1
                print(f"Failed: {error_msg}")
        
        print(f"\nStatic scene task execution completed:")
        print(f"  Successful: {successful_tasks}")
        print(f"  Failed: {failed_tasks}")
        print(f"  Total: {len(tasks)}")
    else:
        # Parallel execution
        run_static_scene_tasks_parallel(tasks, args, args.max_workers)
        
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.2f} seconds")


if __name__ == "__main__":
    main()
