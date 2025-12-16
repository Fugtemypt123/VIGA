# run_blendergym_steps.py
# 枚举 scripts 目录下的脚本，使用 executor 执行并保存 .blend 文件，然后渲染图片
import sys
import argparse
import os
from pathlib import Path
import subprocess
import logging
from typing import Optional, Dict, Tuple
import base64
from PIL import Image
import io
import math
import shutil

# 导入 Executor 类（从 tools/exec_blender.py）
class Executor:
    def __init__(self,
                 blender_command: str,
                 blender_file: str,
                 blender_script: str,
                 script_save: str,
                 render_save: str,
                 blender_save: Optional[str] = None,
                 gpu_devices: Optional[str] = None):
        self.blender_command = blender_command
        self.blender_file = blender_file
        self.blender_script = blender_script
        self.script_path = Path(script_save)
        self.render_path = Path(render_save)
        self.blender_save = blender_save
        self.gpu_devices = gpu_devices  # e.g.: "0,1" or "0"
        self.count = 0

        self.script_path.mkdir(parents=True, exist_ok=True)
        self.render_path.mkdir(parents=True, exist_ok=True)

    def _execute_blender(self, script_path: str, render_path: str = '') -> Tuple[bool, str, str]:
        cmd = [
            self.blender_command,
            "--background", self.blender_file,
            "--python", self.blender_script,
            "--", script_path, render_path
        ]
        if self.blender_save:
            cmd.append(self.blender_save)
        cmd_str = " ".join(cmd)
        
        # Set environment variables to control GPU devices
        env = os.environ.copy()
        if self.gpu_devices:
            env['CUDA_VISIBLE_DEVICES'] = self.gpu_devices
            logging.info(f"Setting CUDA_VISIBLE_DEVICES to: {self.gpu_devices}")
            
        # Ban blender audio error
        env['AL_LIB_LOGLEVEL'] = '0'
        
        try:
            proc = subprocess.run(cmd_str, shell=True, check=True, capture_output=True, text=True, env=env)
            out = proc.stdout
            err = proc.stderr
            if os.path.isdir(render_path):
                imgs = sorted([str(p) for p in Path(render_path).glob("*") if p.suffix in ['.png','.jpg']])
                return True, imgs, out
            return True, out, err
        except subprocess.CalledProcessError as e:
            logging.error(f"Blender failed: {e}")
            return False, e.stderr, e.stdout

    def _encode_image(self, img_path: str) -> str:
        img = Image.open(img_path)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    def _parse_code(self, full_code: str) -> str:
        if full_code.startswith("```python") and full_code.endswith("```"):
            return full_code[len("```python"):-len("```")]
        return full_code

    def execute(self, code: str) -> Dict:
        self.count += 1
        code_file = self.script_path / f"{self.count}.py"
        render_file = self.render_path / f"{self.count}"
        code = self._parse_code(code)
        
        # File operations
        with open(code_file, "w") as f:
            f.write(code)
        os.makedirs(render_file, exist_ok=True)
        for img in os.listdir(render_file):
            os.remove(os.path.join(render_file, img))
            
        # Execute Blender
        success, stdout, stderr = self._execute_blender(str(code_file), str(render_file))
        # Check if render_file is empty or not exist
        if not success:
            os.rmdir(render_file)
            return {"status": "error", "output": {"text": ['Error: ' + (stderr or stdout)]}}
        elif len(os.listdir(render_file)) == 0:
            os.rmdir(render_file)
            return {"status": "success", "output": {"text": ['The code executed successfully, but no image was generated. Please check and make sure that:\n(1) you have added the camera in the code (just modify the camera pose and other information, do not render the image in the code).\n(2) You may need to handle errors in the code. The following is the return message for reference. Please check if there are any errors and fix them: ' + (stderr or stdout)]}}
        else:
            return {"status": "success", "output": {"image": stdout, "text": [f"Render from camera {x}" for x in range(len(stdout))], 'require_verifier': True}}


logging.basicConfig(level=logging.INFO)

def calculate_camera_rotation_from_look_at(cam_loc: tuple, look_at: tuple) -> tuple:
    """
    从相机位置和目标点计算旋转欧拉角（度）
    返回 (rx, ry, rz) 以度为单位的欧拉角
    """
    import numpy as np
    
    cam_pos = np.array(cam_loc)
    target_pos = np.array(look_at)
    
    # 计算方向向量（从相机指向目标）
    direction = target_pos - cam_pos
    distance = np.linalg.norm(direction)
    
    if distance < 1e-6:
        # 如果相机和目标重合，返回默认旋转
        return (0.0, 0.0, 0.0)
    
    direction = direction / distance  # 归一化
    
    # 计算旋转角度
    # Blender 相机默认 -Z 轴为前向，Y 轴为上
    # 我们需要旋转使得 -Z 指向目标点
    
    # 计算 pitch (绕 X 轴旋转，上下)
    pitch = math.degrees(math.asin(-direction[2]))
    
    # 计算 yaw (绕 Z 轴旋转，左右)
    yaw = math.degrees(math.atan2(direction[1], direction[0]))
    
    # 在 Blender 的 Euler XYZ 模式下，顺序是 (X, Y, Z)
    # 我们需要调整以适应 Blender 的坐标系
    # X 旋转对应 pitch
    # Z 旋转对应 yaw  
    # Y 旋转设为 0（不绕 Y 轴旋转）
    
    # 在 Blender 中，相机看向 -Z 方向，所以需要调整
    # 使用 to_track_quat 的等效计算
    
    # 简化计算：使用 atan2 和 asin
    rx = pitch  # 上下角度
    ry = 0.0    # 不绕 Y 轴旋转
    rz = yaw    # 左右角度
    
    return (rx, ry, rz)

def create_render_script(render_dir: Path, cam_name: str, cam_loc: tuple, 
                        cam_rot_eul: tuple, lens: float, engine: str, 
                        res: tuple, file_format: str, samples: int, 
                        device: str, color_mgt_view: str, color_mgt_look: str):
    """创建在 Blender 中运行的渲染脚本"""
    script_content = f'''import bpy
import sys
from math import radians

# 渲染输出路径
render_output = sys.argv[sys.argv.index("--") + 2] if len(sys.argv) > sys.argv.index("--") + 2 else None
if not render_output:
    print("Error: No render output path specified")
    sys.exit(1)
    
# 设置固定相机
cam_name = "{cam_name}"
cam_loc = {cam_loc}
cam_rot_eul = {cam_rot_eul}
lens = {lens}

cam_obj = bpy.data.objects.get(cam_name)
if cam_obj is None or cam_obj.type != 'CAMERA':
    cam_data = bpy.data.cameras.new(cam_name)
    cam_obj = bpy.data.objects.new(cam_name, cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)

cam_obj.location = cam_loc
cam_obj.rotation_mode = 'XYZ'
cam_obj.rotation_euler = cam_rot_eul
cam_obj.data.lens = lens
bpy.context.scene.camera = cam_obj

# 渲染设置
scene = bpy.context.scene
W, H = {res}
scene.render.resolution_x = W
scene.render.resolution_y = H
scene.render.image_settings.file_format = "{file_format}"
scene.view_settings.view_transform = "{color_mgt_view}"
scene.view_settings.look = "{color_mgt_look}"

if "{engine}" == "CYCLES":
    scene.render.engine = "CYCLES"
    cycles = scene.cycles
    cycles.device = "GPU" if "{device}" == "GPU" else "CPU"
    try:
        prefs = bpy.context.preferences
        prefs.addons["cycles"].preferences.compute_device_type = "CUDA"
    except Exception:
        pass
    cycles.samples = {samples}
    cycles.use_adaptive_sampling = True
else:
    scene.render.engine = "BLENDER_EEVEE"
    scene.eevee.taa_render_samples = max(1, {samples})

# 渲染
bpy.context.scene.render.filepath = render_output
bpy.ops.render.render(write_still=True)
print("Render completed to", bpy.context.scene.render.filepath)
'''
    return script_content

def create_empty_blend_file(blender_command: str, output_path: str, gpu_devices: Optional[str] = None) -> bool:
    """创建一个新的空 blender 文件"""
    create_script = f'''import bpy
bpy.ops.wm.save_as_mainfile(filepath=r"{output_path}")
'''
    script_path = Path(output_path).parent / "_temp_create.py"
    with open(script_path, "w") as f:
        f.write(create_script)
    
    cmd = [
        blender_command,
        "--background",
        "--python", str(script_path)
    ]
    cmd_str = " ".join(cmd)
    
    env = os.environ.copy()
    if gpu_devices:
        env['CUDA_VISIBLE_DEVICES'] = gpu_devices
    env['AL_LIB_LOGLEVEL'] = '0'
    
    try:
        proc = subprocess.run(cmd_str, shell=True, check=True, capture_output=True, text=True, env=env)
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to create empty blend file: {e.stderr}")
        return False
    finally:
        if script_path.exists():
            script_path.unlink()

def render_blend_file(blender_command: str, blend_file: str, render_output: str, 
                     render_script_content: str, gpu_devices: Optional[str] = None):
    """渲染单个 .blend 文件"""
    render_script_path = Path(blend_file).parent / "_temp_render.py"
    with open(render_script_path, "w") as f:
        f.write(render_script_content)
    
    cmd = [
        blender_command,
        "--background", blend_file,
        "--python", str(render_script_path),
        "--", blend_file, render_output
    ]
    cmd_str = " ".join(cmd)
    
    env = os.environ.copy()
    if gpu_devices:
        env['CUDA_VISIBLE_DEVICES'] = gpu_devices
    env['AL_LIB_LOGLEVEL'] = '0'
    
    try:
        proc = subprocess.run(cmd_str, shell=True, check=True, capture_output=True, text=True, env=env)
        return True, proc.stdout
    except subprocess.CalledProcessError as e:
        logging.error(f"Render failed for {blend_file}: {e.stderr}")
        return False, e.stderr
    finally:
        if render_script_path.exists():
            render_script_path.unlink()

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, type=str, help="Output name (e.g., 20251028_133713)")
    ap.add_argument("--blender_command", type=str, default="utils/infinigen/blender/blender", help="Blender command")
    # blender_file 不再需要，会在输出目录下自动创建
    ap.add_argument("--blender_script", type=str, default="data/static_scene/verifier_script.py", help="Blender execution script")
    ap.add_argument("--blender_save", type=str, default=None, help="Directory to save intermediate .blend files")
    ap.add_argument("--output_dir", type=str, default=None, help="Directory to save rendered images")
    ap.add_argument("--gpu_devices", type=str, default=None, help="GPU devices (e.g., '0,1')")
    return ap.parse_args()

def main():
    args = parse_args()
    
    # 构建路径
    base_path = Path(f"output/static_scene/demo/{args.name}/")
    for dir in os.listdir(base_path):
        if os.path.isdir(dir):
            base_path = base_path / dir
            os.makedirs(base_path, exist_ok=True)
        break
    
    scripts_dir = os.path.dirname(base_path) + "/scripts"    
    if not os.path.exists(scripts_dir):
        logging.error(f"Scripts directory not found: {scripts_dir}")
        return
    
    # 设置输出目录
    if args.blender_save is None:
        args.blender_save = str(base_path)
    if args.output_dir is None:
        args.output_dir = str(base_path / "renders")
    render_dir = Path(args.output_dir)
    
    blender_save_dir = Path(args.blender_save)
    blender_save_dir.mkdir(parents=True, exist_ok=True)
    render_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建初始的空 blender 文件
    initial_blend_file = blender_save_dir / "video.blend"
    if initial_blend_file.exists():
        os.remove(initial_blend_file)
    create_empty_blend_cmd = (
        f"{args.blender_command} --background --factory-startup "
        f"--python-expr \"import bpy; bpy.ops.wm.read_factory_settings(use_empty=True); bpy.ops.wm.save_mainfile(filepath='" + str(initial_blend_file) + "')\""
    )
    subprocess.run(create_empty_blend_cmd, shell=True, check=True)
    
    # 枚举脚本文件
    script_files = sorted(os.listdir(scripts_dir), key=lambda x: int(x.split('.')[0]))
    logging.info(f"Found {len(script_files)} script files")
    
    temp_executor = Executor(
        blender_command=args.blender_command,
        blender_file=str(initial_blend_file),
        blender_script=args.blender_script,
        script_save=str(os.path.join(os.path.dirname(render_dir), "scripts")),
        render_save=str(render_dir),
        blender_save=str(initial_blend_file),  # 保存为当前步骤的 .blend 文件
        gpu_devices=args.gpu_devices
    )
    
    for script_file in script_files:
        step_num = script_file.split('.')[0]
        logging.info(f"[Step {step_num}] Processing {script_file}...")
        
        # 读取脚本内容
        with open(os.path.join(scripts_dir, script_file), "r") as f:
            code = f.read()
        
        # 执行脚本并保存 .blend 文件
        result = temp_executor.execute(code)
        if result["status"] != "success":
            logging.error(f"[Step {step_num}] Execution failed: {result['output'].get('text', ['Unknown error'])}")
            # 即使失败也继续，但跳过渲染
            continue
    
        # 将initial_blend_file拷贝到对应的id目录下
        step_dir = render_dir / step_num
        step_dir.mkdir(parents=True, exist_ok=True)
        target_blend_file = step_dir / "state.blend"
        shutil.copy2(str(initial_blend_file), str(target_blend_file))
        logging.info(f"[Step {step_num}] Copied blend file to {target_blend_file}")
    
    logging.info("[OK] All steps processed and rendered.")

if __name__ == "__main__":
    main()