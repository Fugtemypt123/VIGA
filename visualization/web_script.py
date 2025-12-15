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
from pathlib import Path
from typing import List, Dict, Optional
from flask import Flask, render_template, jsonify, send_from_directory, request

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
STEPS_DATA: List[Dict] = []
BASE_PATH: str = ""
RENDERS_DIR: Path = None
IMAGE_PATH: str = ""
VIDEO_PATH: str = ""


def parse_trajectory(traj_path: str, animation: bool = False, fix_camera: bool = False):
    """Parse trajectory file and extract steps, same logic as video_script.py"""
    global STEPS_DATA, RENDERS_DIR, IMAGE_PATH, VIDEO_PATH
    
    traj = json.loads(Path(traj_path).read_text(encoding="utf-8"))
    RENDERS_DIR = Path(traj_path).parent / "renders"
    
    if fix_camera:
        IMAGE_PATH = os.path.dirname(traj_path) + '/video/renders'
    if animation:
        VIDEO_PATH = os.path.dirname(traj_path) + '/video/renders'
    
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
            "step_index": len(STEPS_DATA),
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
            video_dir = os.path.join(VIDEO_PATH, f'{code_count}')
            video_file = os.path.join(video_dir, 'Camera_Main.mp4')
            if os.path.exists(video_file):
                step_data["video_path"] = video_file
                step_data["image_path"] = None  # Use video instead
        elif fix_camera:
            # Use fixed camera renders
            right_img_path = os.path.join(IMAGE_PATH, f'{code_count}.png')
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
            image_path = user_message['content'][2]['text'].split("Image loaded from local path: ")[1]
            image_name = image_path.split("/renders/")[-1]
            right_img_path = os.path.join(str(RENDERS_DIR), image_name)
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
            blend_path = os.path.join(RENDERS_DIR, f'{code_count}', 'state.blend')
            if not os.path.exists(blend_path):
                # Try parent renders directory
                blend_path = os.path.join(RENDERS_DIR.parent, 'renders', f'{code_count}', 'state.blend')
        
        if os.path.exists(blend_path):
            step_data["blend_path"] = blend_path
        
        # Only add step if it has image or video
        if step_data["image_path"] or step_data["video_path"]:
            STEPS_DATA.append(step_data)
    
    print(f"Parsed {len(STEPS_DATA)} steps from trajectory")


@app.route('/')
def index():
    """Main page"""
    return render_template('trajectory_viewer.html')


@app.route('/api/preview-images')
def get_preview_images():
    """Get preview images for entry page (first 10 images)"""
    preview_images = []
    for i, step in enumerate(STEPS_DATA[:10]):  # Only first 10 steps
        if step.get("image_path"):
            preview_images.append({
                "step_index": step["step_index"],
                "image_url": f"/api/image/{step['step_index']}"
            })
    return jsonify({"images": preview_images})


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
        return jsonify({"error": "Step not found"}), 404
    
    step = STEPS_DATA[step_index]
    if not step["blend_path"] or not os.path.exists(step["blend_path"]):
        return jsonify({"error": "Blend file not found"}), 404
    
    blend_dir = os.path.dirname(step["blend_path"])
    blend_file = os.path.basename(step["blend_path"])
    return send_from_directory(blend_dir, blend_file, as_attachment=True)


def main():
    global BASE_PATH
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", type=str, default="20251028_133713")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--host", type=str, default="0.0.0.0")
    ap.add_argument("--fix_camera", action="store_true", help="固定相机位置和方向")
    ap.add_argument("--animation", action="store_true", help="从video_path加载MP4文件并拼接（与fix_camera相同逻辑）")
    args = ap.parse_args()
    
    # Determine base path (same logic as video_script.py)
    if args.animation:
        if os.path.exists(f'output/dynamic_scene/demo/{args.name}'):
            base_path = f'output/dynamic_scene/demo/{args.name}'
        else:
            base_path = f'output/dynamic_scene/{args.name}'
    else:
        if os.path.exists(f'output/static_scene/demo/{args.name}'):
            base_path = f'output/static_scene/demo/{args.name}'
        else:
            base_path = f'output/static_scene/{args.name}'
    
    BASE_PATH = base_path
    
    # Find trajectory file
    traj_path = ''
    for task in os.listdir(base_path):
        task_path = os.path.join(base_path, task)
        if os.path.isdir(task_path) and os.path.exists(os.path.join(task_path, 'generator_memory.json')):
            traj_path = os.path.join(task_path, 'generator_memory.json')
            break
    
    if not traj_path or not os.path.exists(traj_path):
        print(f"Error: Could not find generator_memory.json in {base_path}")
        return
    
    print(f"Loading trajectory from: {traj_path}")
    parse_trajectory(traj_path, animation=args.animation, fix_camera=args.fix_camera)
    
    if len(STEPS_DATA) == 0:
        print("Warning: No steps found in trajectory")
        return
    
    print(f"Starting web server on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=True)


if __name__ == "__main__":
    main()

# python visualization/web_script.py --name 20251030_033307