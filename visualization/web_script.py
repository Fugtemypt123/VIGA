#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
web_script.py
Create a web interface to display trajectory steps with thought, code, and rendered images.
Features:
- Character-by-character typing for thought (like ChatGPT)
- Scrollable code area with syntax highlighting
- Rendered image display (with potential for 3D Blender file interaction)
- Continue button to proceed to next step
- Real-time appearance even though loading existing trajectory

Note on 3D Blender File Interaction:
To enable interactive 3D viewing of Blender files (move mouse to change camera):
1. Convert .blend files to glTF/glb format (requires Blender Python API)
2. Serve the converted files via the /api/blend endpoint
3. Load them in the frontend using three.js GLTFLoader
4. Use OrbitControls for camera interaction

For now, the rendered images are displayed. The structure is in place for future 3D support.
"""

import os
import json
import argparse
import subprocess
import threading
import time
from pathlib import Path
from typing import List, Dict, Optional
from flask import Flask, render_template, jsonify, send_from_directory, request
from werkzeug.utils import secure_filename

# Set template folder explicitly
template_dir = os.path.join(os.path.dirname(__file__), 'templates')
app = Flask(__name__, template_folder=template_dir)

# Optional CORS support
try:
    from flask_cors import CORS
    CORS(app)
except ImportError:
    # CORS not available, but not critical for local development
    pass

# Global state
STEPS_DATA: List[Dict] = []  # Current active trajectory
BASE_PATH: str = ""
RENDERS_DIR: Path = None
IMAGE_PATH: str = ""
VIDEO_PATH: str = ""

# Scene to trajectory path mapping
SCENE_TRAJECTORY_MAP = {
    "restroom5": "output/static_scene/demo/20251017_133317/restroom5",
    "house11": "output/static_scene/demo/20251030_121641/house11",
    "bathroom20": "output/static_scene/demo/20251030_121643/bathroom20",
    "glass24": "output/static_scene/demo/20251030_121642/glass24",
    "blueroom26": "output/static_scene/demo/20251205_133154/blueroom",
    "bedroom32": "output/static_scene/demo/20251214_043022/bedroom32"
}

# Preloaded trajectories for all scenes
PRELOADED_TRAJECTORIES: Dict[str, Dict] = {}

# User upload and monitoring state
USER_UPLOAD_DIR = "/mnt/data/users/shaofengyin/AgenticVerifier/data/static_scene/user"
OUTPUT_BASE_DIR = "/mnt/data/users/shaofengyin/AgenticVerifier/output/static_scene"
ACTIVE_MONITORING: Dict[str, Dict] = {}  # {task_id: {output_dir, last_step_count, monitoring_thread}}
KNOWN_OUTPUT_DIRS: set = set()  # Track known output directories


def parse_trajectory(traj_path: str, animation: bool = False, fix_camera: bool = False, return_data: bool = False):
    """Parse trajectory file and extract steps, same logic as video_script.py
    
    Args:
        traj_path: Path to trajectory file
        animation: Whether this is an animation trajectory
        fix_camera: Whether to use fixed camera renders
        return_data: If True, return parsed data instead of modifying global STEPS_DATA
    
    Returns:
        If return_data is True, returns dict with steps_data, renders_dir, image_path, video_path
        Otherwise, modifies global STEPS_DATA and returns None
    """
    global STEPS_DATA, RENDERS_DIR, IMAGE_PATH, VIDEO_PATH
    
    traj = json.loads(Path(traj_path).read_text(encoding="utf-8"))
    renders_dir = Path(traj_path).parent / "renders"
    
    if return_data:
        steps_data = []
        image_path = ""
        video_path = ""
    else:
        STEPS_DATA = []
        RENDERS_DIR = renders_dir
        IMAGE_PATH = ""
        VIDEO_PATH = ""
        steps_data = STEPS_DATA
    
    if fix_camera:
        image_path_val = os.path.dirname(traj_path) + '/video/renders'
        if return_data:
            image_path = image_path_val
        else:
            IMAGE_PATH = image_path_val
    if animation:
        video_path_val = os.path.dirname(traj_path) + '/video/renders'
        if return_data:
            video_path = video_path_val
        else:
            VIDEO_PATH = video_path_val
    
    code_count = 0
    count = 0
    
    for i, complete_step in enumerate(traj, start=1):
        if 'tool_calls' not in complete_step:
            continue
        tool_call = complete_step['tool_calls'][0]
        if tool_call['function']['name'] == "execute_and_evaluate" or tool_call['function']['name'] == "get_scene_info":
            code_count += 1
        if tool_call['function']['name'] != "execute_and_evaluate":
            continue
        
        step = json.loads(tool_call['function']['arguments'])
        thought = step.get("thought", "").strip()
        if "code" in step:
            code = step.get("code", "").strip()
        else:
            code = step.get("full_code", "").strip()
        
        if i+1 >= len(traj):
            continue
        
        step_data = {
            "step_index": len(steps_data),
            "code_count": code_count,
            "thought": thought,
            "code": code,
            "image_path": None,
            "blend_path": None,
            "video_path": None,
            "is_animation": animation,
            "is_fix_camera": fix_camera
        }
        
        if animation:
            # Check for video file
            video_dir = os.path.join(video_path_val if return_data else VIDEO_PATH, f'{code_count}')
            video_file = os.path.join(video_dir, 'Camera_Main.mp4')
            if os.path.exists(video_file):
                step_data["video_path"] = video_file
                step_data["image_path"] = None  # Use video instead
        elif fix_camera:
            # Use fixed camera renders
            right_img_path = os.path.join(image_path_val if return_data else IMAGE_PATH, f'{code_count}.png')
            if os.path.exists(right_img_path):
                step_data["image_path"] = right_img_path
        else:
            # Use trajectory-based image paths
            user_message = traj[i+1]
            if user_message['role'] != 'user':
                continue
            if len(user_message['content']) < 3:
                continue
            if 'Image loaded from local path: ' not in user_message['content'][2]['text']:
                continue
            img_path = user_message['content'][2]['text'].split("Image loaded from local path: ")[1]
            image_name = img_path.split("/renders/")[-1]
            right_img_path = os.path.join(str(renders_dir), image_name)
            right_img_path = os.path.abspath(right_img_path)  # 转换为绝对路径并规范化
            if os.path.exists(right_img_path):
                step_data["image_path"] = right_img_path
                count += 1
        
        # Check for .blend file (saved in render directories)
        if fix_camera or animation:
            # For fix_camera/animation, blend file might be in video/renders/{code_count}/state.blend
            blend_path = os.path.join(os.path.dirname(traj_path), 'video', 'renders', f'{code_count}', 'state.blend')
            if not os.path.exists(blend_path):
                # Try alternative location
                blend_path = os.path.join(os.path.dirname(traj_path), 'renders', f'{code_count}', 'state.blend')
        else:
            # For normal trajectory, blend file is in renders/{code_count}/state.blend
            blend_path = os.path.join(str(renders_dir), f'{code_count}', 'state.blend')
            if not os.path.exists(blend_path):
                # Try parent renders directory
                blend_path = os.path.join(renders_dir.parent, 'renders', f'{code_count}', 'state.blend')
        
        if os.path.exists(blend_path):
            step_data["blend_path"] = blend_path
        
        # Only add step if it has image or video
        if step_data["image_path"] or step_data["video_path"]:
            steps_data.append(step_data)
    
    if return_data:
        return {
            "steps_data": steps_data,
            "renders_dir": renders_dir,
            "image_path": image_path if fix_camera else "",
            "video_path": video_path if animation else ""
        }
    else:
        print(f"Parsed {len(STEPS_DATA)} steps from trajectory")


@app.route('/')
def index():
    """Main page"""
    return render_template('trajectory_viewer.html')


@app.route('/api/preview-images')
def get_preview_images():
    """Get preview images for entry page (6 target images)"""
    preview_images = []
    
    # List of target images to display (scene name, file extension)
    target_images = [
        ("restroom5", "png"),
        ("house11", "png"),
        ("bathroom20", "png"),
        ("glass24", "png"),
        ("blueroom26", "jpeg"),
        ("bedroom32", "png")
    ]
    
    # Add all target images
    for idx, (scene_name, ext) in enumerate(target_images):
        target_path = os.path.abspath(f'data/static_scene/{scene_name}/target.{ext}')
        if os.path.exists(target_path):
            preview_images.append({
                "step_index": -1 - idx,  # Special negative index for target images
                "image_url": f"/api/target-image/{scene_name}",
                "is_target": True,
                "scene_name": scene_name
            })
    
    return jsonify({"images": preview_images})


@app.route('/api/target-image/<scene_name>')
def get_target_image(scene_name):
    """Serve target image for a specific scene"""
    # Special handling for user uploads
    if scene_name == "user":
        target_path = os.path.join(USER_UPLOAD_DIR, "target.png")
        if not os.path.exists(target_path):
            target_path = os.path.join(USER_UPLOAD_DIR, "target.jpeg")
    else:
        # Try different file extensions
        possible_extensions = ['png', 'jpeg', 'jpg']
        target_path = None
        
        for ext in possible_extensions:
            path = os.path.abspath(f'data/static_scene/{scene_name}/target.{ext}')
            if os.path.exists(path):
                target_path = path
                break
    
    if not target_path or not os.path.exists(target_path):
        print(f"[DEBUG] Target image not found for scene: {scene_name}")
        return jsonify({"error": f"Target image not found for scene: {scene_name}"}), 404
    
    # 确保路径是字符串格式
    target_path = str(target_path)
    image_dir = os.path.dirname(target_path)
    image_file = os.path.basename(target_path)
    return send_from_directory(image_dir, image_file)


def find_trajectory_file(base_path):
    """Find generator_memory.json in base_path or its subdirectories"""
    if not os.path.isdir(base_path):
        return None
    
    # Check if generator_memory.json is directly in base_path
    traj_path = os.path.join(base_path, 'generator_memory.json')
    if os.path.exists(traj_path):
        return traj_path
    
    # Look in subdirectories
    for task in os.listdir(base_path):
        task_path = os.path.join(base_path, task)
        if os.path.isdir(task_path):
            traj_path = os.path.join(task_path, 'generator_memory.json')
            if os.path.exists(traj_path):
                return traj_path
    
    return None


def preload_all_trajectories():
    """Preload all trajectories for all scenes"""
    global PRELOADED_TRAJECTORIES
    
    print("Preloading all trajectories...")
    for scene_name, trajectory_path in SCENE_TRAJECTORY_MAP.items():
        if trajectory_path is None:
            continue
        
        base_path = trajectory_path
        traj_path = find_trajectory_file(base_path)
        
        if traj_path and os.path.exists(traj_path):
            try:
                print(f"  Loading {scene_name} from {traj_path}")
                parsed_data = parse_trajectory(traj_path, animation=False, fix_camera=False, return_data=True)
                PRELOADED_TRAJECTORIES[scene_name] = {
                    "steps_data": parsed_data["steps_data"],
                    "base_path": base_path,
                    "renders_dir": parsed_data["renders_dir"],
                    "image_path": parsed_data["image_path"],
                    "video_path": parsed_data["video_path"]
                }
                print(f"  Loaded {len(parsed_data['steps_data'])} steps for {scene_name}")
            except Exception as e:
                print(f"  Error loading {scene_name}: {e}")
        else:
            print(f"  Warning: Could not find trajectory for {scene_name} at {base_path}")
    
    print(f"Preloaded {len(PRELOADED_TRAJECTORIES)} trajectories")


@app.route('/api/load-trajectory/<scene_name>')
def load_trajectory(scene_name):
    """Load trajectory for a specific scene from preloaded data"""
    global STEPS_DATA, BASE_PATH, RENDERS_DIR, IMAGE_PATH, VIDEO_PATH
    
    # Check if trajectory is preloaded
    if scene_name in PRELOADED_TRAJECTORIES:
        preloaded = PRELOADED_TRAJECTORIES[scene_name]
        STEPS_DATA = preloaded["steps_data"]
        BASE_PATH = preloaded["base_path"]
        RENDERS_DIR = preloaded["renders_dir"]
        IMAGE_PATH = preloaded["image_path"]
        VIDEO_PATH = preloaded["video_path"]
        
        return jsonify({
            "success": True,
            "total_steps": len(STEPS_DATA),
            "scene_name": scene_name
        })
    else:
        return jsonify({"error": f"Trajectory not found for scene: {scene_name}"}), 404


@app.route('/api/steps')
def get_steps():
    """Get all steps metadata"""
    return jsonify({
        "total_steps": len(STEPS_DATA),
        "steps": [{
            "step_index": s["step_index"],
            "code_count": s["code_count"]
        } for s in STEPS_DATA]
    })


@app.route('/api/step/<int:step_index>')
def get_step(step_index):
    """Get step data"""
    if step_index < 0 or step_index >= len(STEPS_DATA):
        return jsonify({"error": "Step not found"}), 404
    
    step = STEPS_DATA[step_index]
    response = {
        "step_index": step["step_index"],
        "code_count": step["code_count"],
        "thought": step["thought"],
        "code": step["code"],
        "has_image": step["image_path"] is not None,
        "has_video": step["video_path"] is not None,
        "has_blend": step["blend_path"] is not None
    }
    
    # Add relative paths for serving
    if step["image_path"]:
        # Make path relative to base
        rel_path = os.path.relpath(step["image_path"], BASE_PATH)
        response["image_url"] = f"/api/image/{step_index}"
    
    if step["video_path"]:
        rel_path = os.path.relpath(step["video_path"], BASE_PATH)
        response["video_url"] = f"/api/video/{step_index}"
    
    if step["blend_path"]:
        response["blend_url"] = f"/api/blend/{step_index}"
    
    return jsonify(response)


@app.route('/api/image/<int:step_index>')
def get_image(step_index):
    """Serve rendered image"""
    if step_index < 0 or step_index >= len(STEPS_DATA):
        return jsonify({"error": "Step not found"}), 404
    
    step = STEPS_DATA[step_index]
    image_path = step.get("image_path")
    
    # 确保路径是字符串格式
    if image_path:
        image_path = str(image_path)
    
    if not image_path or not os.path.exists(image_path):
        # 添加调试信息
        print(f"[DEBUG] Image not found for step {step_index}: {image_path}")
        return jsonify({"error": f"Image not found: {image_path}"}), 404
    
    image_dir = os.path.dirname(image_path)
    image_file = os.path.basename(image_path)
    return send_from_directory(image_dir, image_file)


@app.route('/api/video/<int:step_index>')
def get_video(step_index):
    """Serve video file"""
    if step_index < 0 or step_index >= len(STEPS_DATA):
        return jsonify({"error": "Step not found"}), 404
    
    step = STEPS_DATA[step_index]
    if not step["video_path"] or not os.path.exists(step["video_path"]):
        return jsonify({"error": "Video not found"}), 404
    
    video_dir = os.path.dirname(step["video_path"])
    video_file = os.path.basename(step["video_path"])
    return send_from_directory(video_dir, video_file)


@app.route('/api/blend/<int:step_index>')
def get_blend(step_index):
    """Serve Blender file (for potential 3D interaction)"""
    if step_index < 0 or step_index >= len(STEPS_DATA):
        print(f"[DEBUG] /api/blend: invalid step_index={step_index}, total_steps={len(STEPS_DATA)}")
        return jsonify({"error": "Step not found"}), 404
    
    step = STEPS_DATA[step_index]
    blend_path = step.get("blend_path")

    # 如果预解析阶段没找到 blend 文件，或者路径无效，则根据当前渲染图片目录重新推断
    if not blend_path or not os.path.exists(blend_path):
        image_path = step.get("image_path")
        if image_path:
            # 确保为字符串
            image_path = str(image_path)
            candidate_blend = os.path.join(os.path.dirname(image_path), "state.blend")
            print(f"[DEBUG] /api/blend: step={step_index}, image_path={image_path}, candidate_blend={candidate_blend}")
            if os.path.exists(candidate_blend):
                blend_path = candidate_blend
                # 缓存回步骤数据，后续请求可直接使用
                step["blend_path"] = blend_path

    if not blend_path or not os.path.exists(blend_path):
        print(f"[DEBUG] /api/blend: blend not found for step={step_index}, blend_path={blend_path}")
        return jsonify({"error": "Blend file not found"}), 404
    
    # 确保路径为绝对字符串路径
    blend_path = os.path.abspath(str(blend_path))
    blend_dir = os.path.dirname(blend_path)
    blend_file = os.path.basename(blend_path)
    print(f"[DEBUG] /api/blend: sending blend for step={step_index}, path={blend_path}")
    return send_from_directory(blend_dir, blend_file, as_attachment=True)


def find_new_output_directories():
    """Find new output directories in output/static_scene"""
    global KNOWN_OUTPUT_DIRS
    
    if not os.path.exists(OUTPUT_BASE_DIR):
        return []
    
    current_dirs = set()
    new_dirs = []
    
    # Scan for directories (format: output/static_scene/demo/TIMESTAMP/TASKNAME or output/static_scene/useful/TIMESTAMP/TASKNAME)
    for subdir in ['demo', 'useful']:
        subdir_path = os.path.join(OUTPUT_BASE_DIR, subdir)
        if not os.path.exists(subdir_path):
            continue
        
        for timestamp_dir in os.listdir(subdir_path):
            timestamp_path = os.path.join(subdir_path, timestamp_dir)
            if not os.path.isdir(timestamp_path):
                continue
            
            for task_dir in os.listdir(timestamp_path):
                task_path = os.path.join(timestamp_path, task_dir)
                if not os.path.isdir(task_path):
                    continue
                
                full_path = os.path.abspath(task_path)
                current_dirs.add(full_path)
                
                if full_path not in KNOWN_OUTPUT_DIRS:
                    # Check if generator_memory.json exists
                    traj_file = os.path.join(task_path, 'generator_memory.json')
                    if os.path.exists(traj_file):
                        new_dirs.append(full_path)
    
    # Update known directories
    KNOWN_OUTPUT_DIRS.update(current_dirs)
    
    return new_dirs


def monitor_user_task(task_id, output_dir):
    """Monitor a user task's output directory for new steps"""
    global ACTIVE_MONITORING
    
    print(f"[MONITOR] Starting monitoring for task {task_id}, output_dir: {output_dir}")
    last_step_count = 0
    
    while task_id in ACTIVE_MONITORING:
        try:
            traj_file = os.path.join(output_dir, 'generator_memory.json')
            if os.path.exists(traj_file):
                # Parse trajectory to get current step count
                try:
                    parsed_data = parse_trajectory(traj_file, animation=False, fix_camera=False, return_data=True)
                    current_step_count = len(parsed_data["steps_data"])
                    
                    if current_step_count > last_step_count:
                        print(f"[MONITOR] Task {task_id}: Found {current_step_count} steps (was {last_step_count})")
                        last_step_count = current_step_count
                        # Update the active monitoring state
                        ACTIVE_MONITORING[task_id]["last_step_count"] = current_step_count
                except Exception as e:
                    print(f"[MONITOR] Error parsing trajectory for {task_id}: {e}")
            
            time.sleep(2)  # Check every 2 seconds
        except Exception as e:
            print(f"[MONITOR] Error monitoring {task_id}: {e}")
            time.sleep(2)


@app.route('/api/upload-image', methods=['POST'])
def upload_image():
    """Handle image upload and save to user directory"""
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    # Create user directory if it doesn't exist
    os.makedirs(USER_UPLOAD_DIR, exist_ok=True)
    
    # Delete all existing image files in the user directory
    image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp']
    if os.path.exists(USER_UPLOAD_DIR):
        for filename in os.listdir(USER_UPLOAD_DIR):
            file_path = os.path.join(USER_UPLOAD_DIR, filename)
            if os.path.isfile(file_path):
                _, ext = os.path.splitext(filename.lower())
                if ext in image_extensions:
                    try:
                        os.remove(file_path)
                        print(f"[UPLOAD] Deleted existing image: {file_path}")
                    except Exception as e:
                        print(f"[UPLOAD] Warning: Failed to delete {file_path}: {e}")
    
    # Get file extension
    filename = secure_filename(file.filename)
    _, ext = os.path.splitext(filename)
    
    # Save as target.png (or preserve extension if it's jpeg/jpg)
    if ext.lower() in ['.jpg', '.jpeg']:
        target_filename = 'target.jpeg'
    else:
        target_filename = 'target.png'
    
    target_path = os.path.join(USER_UPLOAD_DIR, target_filename)
    file.save(target_path)
    
    print(f"[UPLOAD] Saved image to {target_path}")
    return jsonify({
        "success": True,
        "path": target_path,
        "message": "Image uploaded successfully"
    })


@app.route('/api/start-task', methods=['POST'])
def start_task():
    """Start a new task using sbatch"""
    try:
        # Create sbatch script
        script_content = f"""#!/bin/bash
#SBATCH --job-name=user_task
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=128
#SBATCH --mem=512G
#SBATCH --time=24:00:00

source ~/.bashrc
cd /mnt/data/users/shaofengyin/AgenticVerifier
/mnt/home/shaofengyin19260817/anaconda3/envs/agent/bin/python runners/static_scene.py --task=user --model=gpt-5 --generator-tools=tools/exec_blender.py,tools/generator_base.py,tools/initialize_plan.py
"""
        
        # Save script to temporary location
        script_path = "/tmp/user_task_sbatch.sh"
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Submit job
        result = subprocess.run(
            ['sbatch', script_path],
            capture_output=True,
            text=True,
            cwd='/mnt/data/users/shaofengyin/AgenticVerifier'
        )
        
        if result.returncode != 0:
            return jsonify({"error": f"Failed to submit job: {result.stderr}"}), 500
        
        # Extract job ID from output (format: "Submitted batch job 12345")
        job_id = result.stdout.strip().split()[-1]
        
        # Generate task ID
        task_id = f"user_{int(time.time())}"
        
        # Initialize monitoring (will find output directory later)
        ACTIVE_MONITORING[task_id] = {
            "job_id": job_id,
            "output_dir": None,
            "last_step_count": 0,
            "monitoring_thread": None,
            "status": "starting"
        }
        
        # Start monitoring thread to find output directory
        def find_and_monitor():
            global ACTIVE_MONITORING
            max_wait = 600  # Wait up to 10 minutes
            waited = 0
            start_time = time.time()
            
            while task_id in ACTIVE_MONITORING and waited < max_wait:
                # Look for new directories
                new_dirs = find_new_output_directories()
                for new_dir in new_dirs:
                    # Check if this directory was created after task started (within 10 minutes)
                    # and contains 'user' in the task name or is in useful/ directory
                    dir_mtime = os.path.getmtime(new_dir)
                    time_since_creation = time.time() - dir_mtime
                    
                    # Check if directory was created recently (within 10 minutes) and contains user task
                    if (time_since_creation < 600 and 
                        ('user' in new_dir.lower() or 'useful' in new_dir.lower())):
                        # Verify it's actually a user task by checking if it's in useful/ or has user in name
                        task_name = os.path.basename(new_dir)
                        if 'user' in task_name.lower() or 'useful' in new_dir.lower():
                            ACTIVE_MONITORING[task_id]["output_dir"] = new_dir
                            ACTIVE_MONITORING[task_id]["status"] = "running"
                            
                            # Start monitoring thread
                            monitor_thread = threading.Thread(
                                target=monitor_user_task,
                                args=(task_id, new_dir),
                                daemon=True
                            )
                            monitor_thread.start()
                            ACTIVE_MONITORING[task_id]["monitoring_thread"] = monitor_thread
                            print(f"[MONITOR] Found output directory for {task_id}: {new_dir}")
                            return
                
                time.sleep(5)
                waited += 5
            
            if task_id in ACTIVE_MONITORING:
                ACTIVE_MONITORING[task_id]["status"] = "failed"
                print(f"[MONITOR] Failed to find output directory for {task_id} after {waited} seconds")
        
        find_thread = threading.Thread(target=find_and_monitor, daemon=True)
        find_thread.start()
        
        return jsonify({
            "success": True,
            "task_id": task_id,
            "job_id": job_id,
            "message": "Task started successfully"
        })
    
    except Exception as e:
        print(f"[ERROR] Failed to start task: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/check-new-steps/<task_id>')
def check_new_steps(task_id):
    """Check if there are new steps for a user task"""
    if task_id not in ACTIVE_MONITORING:
        return jsonify({"error": "Task not found"}), 404
    
    task_info = ACTIVE_MONITORING[task_id]
    output_dir = task_info.get("output_dir")
    
    if not output_dir or not os.path.exists(output_dir):
        return jsonify({
            "has_new": False,
            "status": task_info.get("status", "unknown"),
            "message": "Output directory not found yet"
        })
    
    traj_file = os.path.join(output_dir, 'generator_memory.json')
    if not os.path.exists(traj_file):
        return jsonify({
            "has_new": False,
            "status": "preparing",
            "message": "Preparing the assets..."
        })
    
    try:
        parsed_data = parse_trajectory(traj_file, animation=False, fix_camera=False, return_data=True)
        current_step_count = len(parsed_data["steps_data"])
        last_step_count = task_info.get("last_step_count", 0)
        
        has_new = current_step_count > last_step_count
        
        return jsonify({
            "has_new": has_new,
            "current_steps": current_step_count,
            "last_steps": last_step_count,
            "status": "running",
            "output_dir": output_dir
        })
    except Exception as e:
        return jsonify({
            "has_new": False,
            "status": "error",
            "error": str(e)
        })


@app.route('/api/load-user-trajectory/<task_id>')
def load_user_trajectory(task_id):
    """Load trajectory for a user task"""
    global STEPS_DATA, BASE_PATH, RENDERS_DIR, IMAGE_PATH, VIDEO_PATH
    
    if task_id not in ACTIVE_MONITORING:
        return jsonify({"error": "Task not found"}), 404
    
    task_info = ACTIVE_MONITORING[task_id]
    output_dir = task_info.get("output_dir")
    
    if not output_dir or not os.path.exists(output_dir):
        return jsonify({"error": "Output directory not found"}), 404
    
    traj_file = os.path.join(output_dir, 'generator_memory.json')
    if not os.path.exists(traj_file):
        return jsonify({"error": "Trajectory file not found"}), 404
    
    try:
        parsed_data = parse_trajectory(traj_file, animation=False, fix_camera=False, return_data=True)
        
        STEPS_DATA = parsed_data["steps_data"]
        BASE_PATH = output_dir
        RENDERS_DIR = parsed_data["renders_dir"]
        IMAGE_PATH = parsed_data["image_path"]
        VIDEO_PATH = parsed_data["video_path"]
        
        # Update monitoring state
        task_info["last_step_count"] = len(STEPS_DATA)
        
        return jsonify({
            "success": True,
            "total_steps": len(STEPS_DATA),
            "task_id": task_id
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def main():
    global BASE_PATH
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--host", type=str, default="0.0.0.0")
    args = ap.parse_args()
    
    # Initialize known output directories
    if os.path.exists(OUTPUT_BASE_DIR):
        for subdir in ['demo', 'useful']:
            subdir_path = os.path.join(OUTPUT_BASE_DIR, subdir)
            if os.path.exists(subdir_path):
                for timestamp_dir in os.listdir(subdir_path):
                    timestamp_path = os.path.join(subdir_path, timestamp_dir)
                    if os.path.isdir(timestamp_path):
                        for task_dir in os.listdir(timestamp_path):
                            task_path = os.path.join(timestamp_path, task_dir)
                            if os.path.isdir(task_path):
                                KNOWN_OUTPUT_DIRS.add(os.path.abspath(task_path))
    
    print(f"Initialized {len(KNOWN_OUTPUT_DIRS)} known output directories")
    
    # Preload all trajectories first
    preload_all_trajectories()
    
    # Load the first available preloaded trajectory as initial active trajectory
    if PRELOADED_TRAJECTORIES:
        first_scene = next(iter(PRELOADED_TRAJECTORIES.keys()))
        preloaded = PRELOADED_TRAJECTORIES[first_scene]
        STEPS_DATA = preloaded["steps_data"]
        BASE_PATH = preloaded["base_path"]
        RENDERS_DIR = preloaded["renders_dir"]
        IMAGE_PATH = preloaded["image_path"]
        VIDEO_PATH = preloaded["video_path"]
        print(f"Loaded default trajectory ({first_scene}) with {len(STEPS_DATA)} steps")
    else:
        print("Error: No trajectories could be preloaded from SCENE_TRAJECTORY_MAP.")
        return
    
    print(f"Starting web server on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=True)


if __name__ == "__main__":
    main()

# python visualization/web_script.py --name 20251030_033307