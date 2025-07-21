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
from pathlib import Path
from typing import List, Dict, Optional

api_key = os.getenv("OPENAI_API_KEY")

def load_autopresent_dataset(base_path: str, task_name: str, task_id: str) -> List[Dict]:
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
        task_list = ['presentation', 'slide', 'design']
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
        target_description_file = task_dir / "target_description.txt"
        
        if not start_code_path.exists():
            print(f"Warning: start.py not found in {task_dir}")
            continue
            
        if not target_description_file.exists():
            print(f"Warning: target_description.txt not found in {task_dir}")
            continue
            
        # Read target description
        with open(target_description_file, 'r') as f:
                target_description = f.read().strip()
            
        task_config = {
            "task_name": task_name,
            "task_dir": task_dir,
            "init_code": start_code_path,
            "target_description": target_description,
        }
        tasks.append(task_config)
        print(f"Found task: {task_name}/{task_dir.name}")
    
    return tasks

def run_autopresent_task(task_config: Dict, args) -> bool:
    """
    Run a single AutoPresent task using main.py
    
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
    code_save = output_base / "slides_code"
    thoughtprocess_save = output_base / "generator_thought.json"
    verifier_thoughtprocess_save = output_base / "verifier_thought.json"
    
    # Create directories
    code_save.mkdir(parents=True, exist_ok=True)
    
    # Build main.py command
    cmd = [
        sys.executable, "main.py",
        "--mode", "2d",
        "--init-code", str(task_config["init_code"]),
        "--target-description", task_config["target_description"],
        "--max-rounds", str(args.max_rounds),
        "--code-save", str(code_save),
        "--thoughtprocess-save", str(thoughtprocess_save),
        "--verifier-thoughtprocess-save", str(verifier_thoughtprocess_save),
        "--vision-model", args.vision_model,
        "--api-key", api_key,
        "--generator-hints", args.generator_hints,
        "--verifier-hints", args.verifier_hints,
        "--slides-server-path", args.slides_server_path,
        "--blender-server-path", "none",  # Not used for AutoPresent
        "--image-server-path", args.image_server_path,
        "--scene-server-path", args.scene_server_path,
    ]
    
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
    parser = argparse.ArgumentParser(description="AutoPresent Runner for AgenticVerifier")
    
    # Dataset parameters
    parser.add_argument("--dataset-path", default="data/autopresent", 
                       help="Path to AutoPresent dataset root directory")
    parser.add_argument("--output-dir", default=f"output/autopresent/{time.strftime('%Y%m%d_%H%M%S')}",
                       help="Output directory for results")
    
    # Task selection
    parser.add_argument("--task", choices=['all', 'presentation', 'slide', 'design'], default='all', help="Specific task to run")
    parser.add_argument("--task-id", default=None, help="Specific task id to run (e.g., '1')")
    
    # Main.py parameters
    parser.add_argument("--max-rounds", type=int, default=10,
                       help="Maximum number of interaction rounds")
    parser.add_argument("--vision-model", default="gpt-4o",
                       help="OpenAI vision model to use")
    parser.add_argument("--generator-hints", default="Generate slides based on the target description",
                       help="Hints for generator agent")
    parser.add_argument("--verifier-hints", default="Verify that the generated slides match the target description",
                       help="Hints for verifier agent")
    
    # Tool server paths
    parser.add_argument("--slides-server-path", default="servers/generator/slides.py",
                       help="Path to Slides MCP server script")
    parser.add_argument("--image-server-path", default="servers/verifier/image.py",
                       help="Path to image processing MCP server script")
    parser.add_argument("--scene-server-path", default="servers/verifier/scene.py",
                       help="Path to scene investigation MCP server script")
    
    args = parser.parse_args()
    
    # Load dataset
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
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save task list for reference
    with open(os.path.join(args.output_dir, "tasks.json"), "w") as f:
        json.dump(tasks, f, indent=2)
    
    # Run tasks
    successful_tasks = 0
    failed_tasks = 0
    
    for i, task_config in enumerate(tasks, 1):
        print(f"\nTask {i}/{len(tasks)}")
        
        success = run_autopresent_task(task_config, args)
        
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
        print("Check individual task directories for slides code and thought processes.")

if __name__ == "__main__":
    main()
