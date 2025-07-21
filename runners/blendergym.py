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
from pathlib import Path
from typing import List, Dict, Optional
from prompts.blender import blender_generator_hints, blender_verifier_hints

api_key = os.getenv("OPENAI_API_KEY")

def load_blendergym_dataset(base_path: str, task_name: str, task_id: str) -> List[Dict]:
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
        task_list = ['blendshape', 'geometry', 'lighting', 'material', 'placement']
    else:
        task_list = [task_name]
        
    # If task_id is not None, only run the task_id
    if task_id is not None:
        task_dirs = [(base_path / task_name / f"{task_name}{task_id}", task_name)]
    # Otherwise, run all tasks in the task_list
    else:
        task_dirs = []
        for task in task_list:
            for task_dir in base_path.glob(f"{task}*"):
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
            "task_dir": task_dir,
            "init_code_path": start_code_path,
            "init_image_path": start_renders_dir,
            "target_image_path": goal_renders_dir,
            "blender_file": blender_file,
        }
        tasks.append(task_config)
        print(f"Found task: {task_name}/{task_dir.name}")
    
    return tasks

def run_blendergym_task(task_config: Dict, args) -> bool:
    """
    Run a single BlenderGym task using main.py
    
    Args:
        task_config: Task configuration dictionary
        args: Command line arguments
        
    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"Running task: {task_config['task_name']}/{task_config['task_dir'].name}")
    print(f"{'='*60}")
    
    # Prepare output directories
    output_base = Path(args.output_dir) / task_config['task_dir'].name
    # render_save = output_base / "renders"
    # generator_thought_save = output_base / "generator_thought.json"
    # verifier_thought_save = output_base / "verifier_thought.json"
    
    # Create directories
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Build main.py command
    cmd = [
        sys.executable, "main.py",
        "--mode", "blendergym",
        "--vision-model", args.vision_model,
        "--api-key", api_key,
        "--max-rounds", str(args.max_rounds),
        "--task-name", task_config["task_name"],
        "--init-code-path", str(task_config["init_code_path"]),
        "--init-image-path", str(task_config["init_image_path"]),
        "--target-image-path", str(task_config["target_image_path"]),
        "--blender-file", str(task_config["blender_file"]),
        "--blender-server-path", args.blender_server_path,
        "--blender-command", args.blender_command,
        "--blender-script", args.blender_script,
        "--image-server-path", args.image_server_path,
        "--scene-server-path", args.scene_server_path,
        "--output-dir", str(output_base),
    ]
    
    if args.blender_file_save:
        cmd.append("--save-blender-file")
    
    print(f"Command: {' '.join(cmd)}")
    
    try:
        # Run the command
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"Task completed successfully: {task_config['task_name']}/{task_config['task_dir'].name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Task failed: {task_config['task_name']}/{task_config['task_dir'].name}")
        print(f"Error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="BlenderGym Runner for AgenticVerifier")
    
    # Dataset parameters
    parser.add_argument("--dataset-path", default="data/blendergym", help="Path to BlenderGym dataset root directory")
    parser.add_argument("--output-dir", default=f"output/blendergym/{time.strftime('%Y%m%d_%H%M%S')}", help="Output directory for results")
    
    # Task selection
    parser.add_argument("--task", choices=['all', 'blendshape', 'geometry', 'lighting', 'material', 'placement'], default='all', help="Specific task to run")
    parser.add_argument("--task-id", default=None, help="Specific task id to run (e.g., '1')")
    
    # Main.py parameters
    parser.add_argument("--max-rounds", type=int, default=10, help="Maximum number of interaction rounds")
    parser.add_argument("--vision-model", default="gpt-4o", help="OpenAI vision model to use")
    
    # Blender parameters
    parser.add_argument("--blender-server-path", default="servers/generator/blender.py", help="Path to Blender MCP server script")
    parser.add_argument("--blender-command", default="utils/blender/infinigen/blender/blender", help="Blender command path")
    parser.add_argument("--blender-script", default="data/blendergym/pipeline_render_script.py", help="Blender execution script")
    parser.add_argument("--save-blender-file", action="store_true", help="Blender save path")
    
    # Tool server paths
    parser.add_argument("--image-server-path", default="servers/verifier/image.py", help="Path to image processing MCP server script")
    parser.add_argument("--scene-server-path", default="servers/verifier/scene.py", help="Path to scene investigation MCP server script")
    
    args = parser.parse_args()
    
    # Load dataset
    print(f"Loading BlenderGym dataset from: {args.dataset_path}")
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
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save task list for reference
    with open(os.path.join(args.output_dir, "tasks.json"), "w") as f:
        json.dump(tasks, f, indent=2)
    
    # Run tasks
    successful_tasks = 0
    failed_tasks = 0
    
    for i, task_config in enumerate(tasks, 1):
        print(f"\nTask {i}/{len(tasks)}")
        
        success = run_blendergym_task(task_config, args)
        
        if success:
            successful_tasks += 1
        else:
            failed_tasks += 1
    
    # Summary
    print(f"\n{'='*60}")
    print("EXECUTION SUMMARY")
    print(f"{'='*60}")
    print(f"Total tasks: {len(tasks)}")
    print(f"Successful: {successful_tasks}")
    print(f"Failed: {failed_tasks}")
    print(f"Output directory: {args.output_dir}")
    
    if successful_tasks > 0:
        print(f"\nResults saved to: {args.output_dir}")
        print("Check individual task directories for renders and thought processes.")

if __name__ == "__main__":
    main()
