#!/usr/bin/env python3
"""
Iterative Alchemy Runner for AgenticVerifier
Generates code iteratively using GPT to transform images closer to target.
Each iteration generates multiple candidates, selects the best using VLM comparison.
"""
import os
import sys
import json
import time
import argparse
import base64
import subprocess
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
import openai
from tqdm import tqdm
import tempfile

# Import API keys from runners directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.common import get_image_base64, extract_code_pieces, build_client

def encode_image(image_path: str) -> str:
    """
    Encode image to base64 string for OpenAI API.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Base64 encoded string
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def vlm_compare_images(image1_path: str, image2_path: str, target_path: str, model: str = "gpt-4o") -> int:
    """
    Use VLM to compare two images and determine which is closer to target.
    
    Args:
        image1_path: Path to first image
        image2_path: Path to second image  
        target_path: Path to target image
        model: Vision model to use
        
    Returns:
        1 if image1 is closer to target, 2 if image2 is closer to target
    """
    try:
        # Encode images
        image1_b64 = encode_image(image1_path)
        image2_b64 = encode_image(image2_path)
        target_b64 = encode_image(target_path)
        
        # Initialize OpenAI client
        client = build_client(model)
        
        # Create messages
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "You are an expert at comparing 3D rendered images. I will show you two rendered images and a target image. Please determine which of the two rendered images is closer to the target image in terms of visual similarity, lighting, materials, geometry, and overall appearance. Respond with only '1' if the first image is closer to the target, or '2' if the second image is closer to the target."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{target_b64}"
                        }
                    },
                    {
                        "type": "text", 
                        "text": "Target image:"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image1_b64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": "Image 1:"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image2_b64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": "Image 2:"
                    }
                ]
            }
        ]
        
        # Make API call
        response = client.chat.completions.create(model=model, messages=messages)
        
        # Parse response
        result = response.choices[0].message.content.strip()
        if result == "1":
            return 1
        elif result == "2":
            return 2
        else:
            # Default to image1 if response is unclear
            print(f"Unexpected VLM response: {result}, defaulting to image1")
            return 1
            
    except Exception as e:
        print(f"VLM comparison failed: {e}, defaulting to image1")
        return 1


def load_blendergym_dataset(base_path: str, task_name: str, test_id: Optional[str] = None,
                            task_id: Optional[str] = None) -> List[Dict]:
    """
    Load BlenderGym dataset structure (mirrors runners/blendergym.py).
    
    Args:
        base_path: Path to BlenderGym dataset root
        task_name: Task category (e.g., 'blendshape', 'geometry', etc., or 'all')
        test_id: Optional test ID for re-running failed cases
        task_id: Optional specific task index within the category
        
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
        
    current_task_dirs: List[str] = []
    if test_id is not None:
        current_task_path = Path(f"output/blendergym/{test_id}")
        for task in task_list:
            for task_dir in current_task_path.glob(f"{task}*"):
                scores_path = task_dir / "scores.json"
                if scores_path.exists():
                    exist_score = False
                    with open(scores_path, 'r') as f:
                        scores = json.load(f)
                        if any(score != {} for score in scores.values()):
                            exist_score = True
                    if exist_score:
                        current_task_dirs.append(os.path.basename(task_dir))
    
    task_dirs: List[Tuple[Path, str]] = []
    for task in task_list:
        for task_dir in base_path.glob(f"{task}*"):
            if os.path.basename(task_dir) in current_task_dirs:
                continue
            task_dirs.append((task_dir, task))
    
    for task_dir, category in task_dirs:
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
        
        if task_id is not None and task_dir.name != f"{category}{task_id}":
            continue
        
        task_config = {
            "task_name": f"{category}/{task_dir.name}",
            "task_dir": str(task_dir),
            "init_code_path": str(start_code_path),
            "init_image_path": str(start_renders_dir),
            "target_image_path": str(goal_renders_dir),
            "blender_file": str(blender_file),
            "target_description": "",
        }
        tasks.append(task_config)
        print(f"Found task: {category}/{task_dir.name}")
    
    return tasks


def execute_blender_code(blender_command: str, blender_file: str, blender_script: str, 
                         code: str, round_name: str, script_save_dir: Path, 
                         render_save_dir: Path, gpu_devices: Optional[str] = None) -> Tuple[bool, str, str]:
    """
    Execute Blender Python code and render images.
    
    Args:
        blender_command: Path to Blender executable
        blender_file: Path to Blender file
        blender_script: Path to Blender execution script
        code: Python code to execute
        round_name: Name for the round (e.g., "1", "temp_1_0")
        script_save_dir: Directory to save the code file (for temp rounds, use None to skip saving)
        render_save_dir: Directory to save rendered images
        gpu_devices: GPU devices string (e.g., "0,1")
        
    Returns:
        Tuple of (success: bool, error_message: str, render_dir_path: str)
        render_dir_path is the path to the render directory if successful
    """
    # Save code to temporary file for execution
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp_code:
        tmp_code.write(code)
        tmp_code_path = tmp_code.name
    
    try:
        # Create render directory
        render_dir = render_save_dir / round_name
        render_dir.mkdir(parents=True, exist_ok=True)
        # Clear existing files in render directory
        for img in render_dir.glob("*.png"):
            img.unlink()
        
        # Execute Blender
        cmd = [
            blender_command,
            "--background", blender_file,
            "--python", blender_script,
            "--", tmp_code_path, str(render_dir)
        ]
        
        if gpu_devices:
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu_devices
        else:
            env = None
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
            env=env
        )
        
        if result.returncode != 0:
            os.unlink(tmp_code_path)
            return False, result.stderr + result.stdout, ""
        
        # Check if render files exist
        render_files = list(render_dir.glob("*.png"))
        if len(render_files) == 0:
            os.unlink(tmp_code_path)
            return False, "No render output generated", ""
        
        os.unlink(tmp_code_path)
        return True, "", str(render_dir)
        
    except subprocess.TimeoutExpired:
        os.unlink(tmp_code_path)
        return False, "Execution timeout", ""
    except Exception as e:
        if os.path.exists(tmp_code_path):
            os.unlink(tmp_code_path)
        return False, str(e), ""


def generate_candidate_codes(start_image_path: str, current_image_path: str, current_code: str,
                             target_image_path: str, task_description: str,
                             model: str = "gpt-4o",
                             num_candidates: int = 4) -> List[str]:
    """
    Use GPT to generate multiple candidate codes to transform current image to target.
    
    Args:
        start_image_path: Path to starting image
        current_image_path: Path to current image
        current_code: Current Blender Python code
        target_image_path: Path to target image
        task_description: Task description text
        model: Model name
        num_candidates: Number of candidate codes to generate (3-4)
        
    Returns:
        List of candidate code strings
    """
    try:
        # Encode images
        start_b64 = encode_image(start_image_path)
        current_b64 = encode_image(current_image_path)
        target_b64 = encode_image(target_image_path)
        client = build_client(model)
        # Create messages
        messages = [
            {
                "role": "system",
                "content": "You are an expert at writing Blender Python code to transform 3D scenes. Given a starting image, current image, current code, and target image, generate multiple candidate code solutions."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""Task description: {task_description}

You are given:
1. Starting image (initial state)
2. Current image (current state after applying current code)
3. Current Blender Python code
4. Target image (desired final state)

Please generate {num_candidates} different candidate Blender Python code solutions that can transform the current image closer to the target image. Each candidate should be a complete, runnable Blender Python script.

Current code:
```python
{current_code}
```

Please output {num_candidates} complete code solutions, separated by "===CANDIDATE_1===", "===CANDIDATE_2===", etc. Each code block should be complete and executable."""
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{start_b64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": "Starting image:"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{current_b64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": "Current image:"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{target_b64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": "Target image:"
                    }
                ]
            }
        ]
        
        # Make API call
        response = client.chat.completions.create(model=model, messages=messages)
        
        # Parse response to extract candidate codes
        content = response.choices[0].message.content
        candidates = []
        
        # Split by candidate markers
        parts = content.split("===CANDIDATE_")
        for i, part in enumerate(parts[1:], 1):  # Skip first empty part
            # Extract code between markers
            if "===" in part:
                code = part.split("===", 1)[1]
                # Remove markdown code blocks if present
                code = code.replace("```python", "").replace("```", "").strip()
                candidates.append(code)
            else:
                # Last candidate or no marker
                code = part.strip()
                code = code.replace("```python", "").replace("```", "").strip()
                if code:
                    candidates.append(code)
        
        # If no markers found, try to extract code blocks
        if len(candidates) == 0:
            import re
            code_blocks = re.findall(r'```(?:python)?\n(.*?)\n```', content, re.DOTALL)
            candidates = code_blocks[:num_candidates]
        
        # Ensure we have the right number of candidates
        while len(candidates) < num_candidates and len(candidates) > 0:
            candidates.append(candidates[-1])  # Duplicate last candidate if needed
        
        return candidates[:num_candidates]
        
    except Exception as e:
        print(f"Error generating candidate codes: {e}")
        # Return empty codes as fallback
        return [current_code] * num_candidates


def tournament_select_best(candidate_results: List[Dict], target_image_path: str, model: str = "gpt-4o") -> int:
    """
    Run tournament to select the best candidate using VLM comparison.
    
    Args:
        candidate_results: List of dicts with keys 'render_dir' (path to render directory)
        target_image_path: Path to target image
        model: Vision model name
        
    Returns:
        Index of the winning candidate
    """
    if len(candidate_results) == 0:
        return 0
    
    if len(candidate_results) == 1:
        return 0
    
    # Tournament: keep pairing and comparing until one winner
    current_candidates = list(range(len(candidate_results)))
    
    while len(current_candidates) > 1:
        next_round = []
        
        # Pair up candidates
        for i in range(0, len(current_candidates), 2):
            if i + 1 < len(current_candidates):
                idx1 = current_candidates[i]
                idx2 = current_candidates[i + 1]
                
                render_dir1 = candidate_results[idx1]['render_dir']
                render_dir2 = candidate_results[idx2]['render_dir']
                
                # Find render1.png in each directory
                render_dir1_path = Path(render_dir1)
                render_dir2_path = Path(render_dir2)
                
                render1_files = sorted(render_dir1_path.glob("render*.png"))
                render2_files = sorted(render_dir2_path.glob("render*.png"))
                
                if not render1_files or not render2_files:
                    # If no renders, default to first candidate
                    next_round.append(idx1)
                    continue
                
                img1_path = str(render1_files[0])
                img2_path = str(render2_files[0])
                
                # Compare which is closer to target
                winner = vlm_compare_images(img1_path, img2_path, target_image_path, model)
                
                # Winner is 1 or 2, convert to index
                winner_idx = idx1 if winner == 1 else idx2
                next_round.append(winner_idx)
            else:
                # Odd number, last one gets bye
                next_round.append(current_candidates[i])
        
        current_candidates = next_round
    
    return current_candidates[0]


def run_iterative_alchemy(task_config: Dict, args) -> Dict:
    """
    Run iterative alchemy process: generate candidates -> tournament -> update -> repeat.
    Saves results in the format: renders/1/, renders/2/, ..., renders/10/ and scripts/1.py, ..., scripts/10.py
    
    Args:
        task_config: Task configuration dictionary
        args: Command line arguments
        
    Returns:
        Dictionary with results
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
    
    start_image_path = str(start_images[0])  # Use first render
    target_image_path = str(target_images[0])  # Use first target
    
    # Initialize current image path (starts from initial image)
    current_image_path = start_image_path
    
    # Copy blender file to output
    shutil.copy(task_config['blender_file'], output_dir / "blender_file.blend")
    blender_file = str(output_dir / "blender_file.blend")
    
    # Iterative process - always generate 10 rounds
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
        
        # Execute all candidate codes (save to temporary directories)
        candidate_results = []
        for i, code in enumerate(candidate_codes):
            print(f"  Executing candidate {i+1}/{len(candidate_codes)}...")
            # Use temporary directory for candidate evaluation
            success, error_msg, render_dir = execute_blender_code(
                blender_command=args.blender_command,
                blender_file=blender_file,
                blender_script=args.blender_script,
                code=code,
                round_name=f"temp_{round_num}_{i}",
                script_save_dir=None,  # Don't save temp scripts
                render_save_dir=render_save_dir,
                gpu_devices=args.gpu_devices
            )
            
            if success and render_dir:
                candidate_results.append({
                    'code': code,
                    'render_dir': render_dir,
                    'index': i
                })
                print(f"    ✓ Candidate {i+1} executed successfully")
            else:
                print(f"    ✗ Candidate {i+1} failed: {error_msg}")
        
        if len(candidate_results) == 0:
            print(f"No candidates executed successfully for round {round_num}, using previous code")
            # If all candidates failed, keep previous code and copy previous render
            if round_num > 1:
                prev_render_dir = render_save_dir / str(round_num - 1)
                current_render_dir = render_save_dir / str(round_num)
                if prev_render_dir.exists():
                    shutil.copytree(prev_render_dir, current_render_dir, dirs_exist_ok=True)
                # Save current code as this round's script
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
            # Clear existing files
            for img in round_render_dir.glob("*.png"):
                img.unlink()
            # Copy all render files
            for render_file in winner_render_dir.glob("*.png"):
                shutil.copy(render_file, round_render_dir / render_file.name)
        
        # Save winner's code
        script_save_dir.mkdir(parents=True, exist_ok=True)
        with open(round_script_file, "w") as f:
            f.write(winner_code)
        
        # Update current state for next iteration
        current_code = winner_code
        # Update current_image_path to the winner's render
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


def main():
    parser = argparse.ArgumentParser(description="Iterative Alchemy Runner for BlenderGym")
    
    # Input parameters
    parser.add_argument("--dataset-path", default="data/blendergym", help="Path to BlenderGym dataset")
    parser.add_argument("--task", choices=['all', 'blendshape', 'geometry', 'lighting', 'material', 'placement'], default='all', help="Task name")
    parser.add_argument("--task-id", default=None, help="Specific task ID to run")
    parser.add_argument("--test-id", default=None, help="Test ID for reruns (matches BlenderGym runner behavior)")
    parser.add_argument("--output-dir", default=None, help="Output directory for results")
    
    # Iterative alchemy parameters
    parser.add_argument("--max-iterations", type=int, default=10, help="Maximum number of iterations (always 10 rounds)")
    parser.add_argument("--num-candidates", type=int, default=4, help="Number of candidate codes to generate per iteration")
    
    # Blender parameters
    parser.add_argument("--blender-command", default="utils/infinigen/blender/blender", help="Blender command path")
    parser.add_argument("--blender-script", default="data/blenderstudio/generator_script.py", help="Blender execution script")
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
        args.output_dir = f"output/blendergym/alchemy/{time.strftime('%Y%m%d_%H%M%S')}"
    
    # Always use 10 iterations
    args.max_iterations = 10
    
    # Load dataset
    print(f"Loading BlenderGym dataset from: {args.dataset_path}")
    tasks = load_blendergym_dataset(args.dataset_path, args.task, args.test_id, args.task_id)
    
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
        # Submit all tasks
        future_to_task = {
            executor.submit(run_iterative_alchemy, task_config, args): task_config 
            for task_config in tasks
        }
        
        # Process completed tasks
        for future in tqdm(as_completed(future_to_task), total=len(future_to_task), desc="Running alchemy"):
            task_config = future_to_task[future]
            try:
                result = future.result()
                results["tasks"].append(result)
                
                if result.get("success", False):
                    results["summary"]["successful_tasks"] += 1
                    print(f"✅ {result['task_name']} completed successfully")
                else:
                    results["summary"]["failed_tasks"] += 1
                    print(f"❌ {result['task_name']} failed: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                results["summary"]["failed_tasks"] += 1
                error_result = {
                    "task_name": task_config['task_name'],
                    "error": str(e),
                    "success": False
                }
                results["tasks"].append(error_result)
                print(f"❌ {task_config['task_name']} failed with exception: {e}")
    
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