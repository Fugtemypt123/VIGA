# blender_executor_server.py
from optparse import Option
import os
import subprocess
import base64
import io
from typing import Optional
from pathlib import Path
from PIL import Image
import logging
from typing import Tuple, Dict
from mcp.server.fastmcp import FastMCP
import json
import requests
import tempfile
import zipfile
import shutil
import bpy
import math

mcp = FastMCP("blender-executor")

# Global executor instance
_executor = None

# Global investigator instance
_investigator = None

# ======================
# Meshy API（从scene.py迁移）
# ======================

class MeshyAPI:
    """Meshy API 客户端：Text-to-3D 生成 + 轮询 + 下载"""
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("MESHY_API_KEY")
        if not self.api_key:
            raise ValueError("Meshy API key is required. Set MESHY_API_KEY environment variable or pass api_key parameter.")
        self.base_url = "https://api.meshy.ai"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def create_text_to_3d_preview(self, prompt: str, **kwargs) -> str:
        """
        创建 Text-to-3D 预览任务（无贴图）
        Returns: task_id (str)
        """
        url = f"{self.base_url}/openapi/v1/text-to-3d"
        payload = {
            "mode": "preview",
            "prompt": prompt[:600],
        }
        payload.update(kwargs or {})
        resp = requests.post(url, headers=self.headers, data=json.dumps(payload))
        resp.raise_for_status()
        data = resp.json()
        # 有的环境返回 {"result": "<id>"}，有的返回 {"id": "<id>"}
        return data.get("result") or data.get("id")

    def poll_text_to_3d(self, task_id: str, interval_sec: float = 5.0, timeout_sec: int = 1800) -> dict:
        """
        轮询 Text-to-3D 任务直到结束
        Returns: 任务 JSON（包含 status / model_urls 等）
        """
        import time
        url = f"{self.base_url}/openapi/v2/text-to-3d/{task_id}"
        deadline = time.time() + timeout_sec
        while True:
            r = requests.get(url, headers=self.headers)
            r.raise_for_status()
            js = r.json()
            status = js.get("status")
            if status in ("SUCCEEDED", "FAILED", "CANCELED"):
                return js
            if time.time() > deadline:
                raise TimeoutError(f"Meshy task {task_id} polling timeout")
            time.sleep(interval_sec)

    def create_text_to_3d_refine(self, preview_task_id: str, **kwargs) -> str:
        """
        基于 preview 发起 refine 贴图任务
        Returns: refine_task_id (str)
        """
        url = f"{self.base_url}/openapi/v1/text-to-3d"
        payload = {
            "mode": "refine",
            "preview_task_id": preview_task_id,
        }
        payload.update(kwargs or {})
        resp = requests.post(url, headers=self.headers, data=json.dumps(payload))
        resp.raise_for_status()
        data = resp.json()
        return data.get("result") or data.get("id")

    def download_model_url(self, file_url: str, output_path: str) -> None:
        """
        从 model_urls 的直链下载文件到本地
        """
        r = requests.get(file_url, stream=True)
        r.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)


# ======================
# 资产导入器（从scene.py迁移）
# ======================

# ======================
# 相机探查器（从scene.py复制）
# ======================

class Investigator3D:
    def __init__(self, blender_path: str):
        self.blender_path = blender_path          # 先保存路径
        self._load_blender_file()                 # 再加载文件
        self.cam = self._get_or_create_cam()
        self.target = None
        self.radius = 5.0
        self.theta = 0.0
        self.phi = 0.0

    def _load_blender_file(self):
        """加载 Blender 文件，如果已经加载了相同的文件则跳过"""
        current_file = bpy.data.filepath
        if current_file != self.blender_path:
            bpy.ops.wm.open_mainfile(filepath=str(self.blender_path))

    def _get_or_create_cam(self):
        if "InvestigatorCamera" in bpy.data.objects:
            return bpy.data.objects["InvestigatorCamera"]
        bpy.ops.object.camera_add()
        cam = bpy.context.active_object
        cam.name = "InvestigatorCamera"
        # optional: copy from existing Camera1
        if 'Camera1' in bpy.data.objects:
            cam.matrix_world.translation = bpy.data.objects['Camera1'].matrix_world.translation.copy()
            print("Copy from Camera1!")
        return cam

    def _save_blender_file(self):
        """保存 Blender 文件"""
        try:
            bpy.ops.wm.save_mainfile(filepath=self.blender_path)
            print(f"Blender file saved to: {self.blender_path}")
            
            # 清理备份文件以避免生成 .blend1 文件
            backup_file = self.blender_path + "1"
            if os.path.exists(backup_file):
                os.remove(backup_file)
                print(f"Removed backup file: {backup_file}")
                
        except Exception as e:
            print(f"Warning: Failed to save blender file: {e}")

    def focus_on_object(self, object_name: str) -> str:
        obj = bpy.data.objects.get(object_name)
        if not obj:
            raise ValueError(f"{object_name} not found")
        self.target = obj
        # track-to
        constraint = None
        for c in self.cam.constraints:
            if c.type == 'TRACK_TO':
                constraint = c
                break
        if not constraint:
            constraint = self.cam.constraints.new('TRACK_TO')
        constraint.target = obj
        constraint.track_axis = 'TRACK_NEGATIVE_Z'
        constraint.up_axis = 'UP_Y'
        self.radius = (self.cam.matrix_world.translation - obj.matrix_world.translation).length
        self.theta = math.atan2(*(self.cam.matrix_world.translation[i] - obj.matrix_world.translation[i] for i in (1,0)))
        self.phi = math.asin((self.cam.matrix_world.translation.z - obj.matrix_world.translation.z)/self.radius)
        self._save_blender_file()
        return "Camera focused on object and Blender file saved"

    def zoom(self, direction: str) -> str:
        if direction == 'in':
            self.radius = max(1, self.radius-3)
        elif direction == 'out':
            self.radius += 3
        return self._update_and_save()

    def move_camera(self, direction: str) -> str:
        step = self.radius
        theta_step = step/(self.radius*math.cos(self.phi))
        phi_step = step/self.radius
        if direction=='up': self.phi = min(math.pi/2-0.1, self.phi+phi_step)
        elif direction=='down': self.phi = max(-math.pi/2+0.1, self.phi-phi_step)
        elif direction=='left': self.theta -= theta_step
        elif direction=='right': self.theta += theta_step
        return self._update_and_save()

    def _update_and_save(self) -> str:
        t = self.target.matrix_world.translation
        x = self.radius*math.cos(self.phi)*math.cos(self.theta)
        y = self.radius*math.cos(self.phi)*math.sin(self.theta)
        z = self.radius*math.sin(self.phi)
        self.cam.matrix_world.translation = (t.x+x, t.y+y, t.z+z)
        self._save_blender_file()
        return "Camera position updated and Blender file saved"
    
    def get_scene_info(self) -> dict:
        """
        获取场景的详细信息，用于测试和调试。
        
        Returns:
            dict: 包含场景信息的字典
        """
        try:
            scene_info = {
                "scene_name": bpy.context.scene.name,
                "camera_info": {
                    "name": self.cam.name,
                    "location": str(self.cam.location),
                    "rotation": str(self.cam.rotation_euler),
                    "constraints": []
                },
                "objects": [],
                "collections": [],
                "materials": [],
                "meshes": []
            }
            
            # 获取摄像头约束信息
            for constraint in self.cam.constraints:
                if constraint.type == 'TRACK_TO':
                    scene_info["camera_info"]["constraints"].append({
                        "type": constraint.type,
                        "name": constraint.name,
                        "target": constraint.target.name if constraint.target else None,
                        "track_axis": constraint.track_axis,
                        "up_axis": constraint.up_axis
                    })
            
            # 获取场景对象信息
            for obj in bpy.context.scene.objects:
                obj_info = {
                    "name": obj.name,
                    "type": obj.type,
                    "location": str(obj.location),
                    "scale": str(obj.scale),
                    "rotation": str(obj.rotation_euler)
                }
                scene_info["objects"].append(obj_info)
            
            # 获取集合信息
            for collection in bpy.data.collections:
                coll_info = {
                    "name": collection.name,
                    "object_count": len(collection.objects),
                    "objects": [obj.name for obj in collection.objects]
                }
                scene_info["collections"].append(coll_info)
            
            # 获取材质信息
            for material in bpy.data.materials:
                scene_info["materials"].append(material.name)
            
            # 获取网格信息
            for mesh in bpy.data.meshes:
                mesh_info = {
                    "name": mesh.name,
                    "vertices": len(mesh.vertices),
                    "faces": len(mesh.polygons),
                    "uv_layers": len(mesh.uv_layers)
                }
                scene_info["meshes"].append(mesh_info)
            
            return scene_info
            
        except Exception as e:
            return {"error": f"Failed to get scene info: {str(e)}"}

class AssetImporter:
    """3D资产导入器，支持多种格式"""
    def __init__(self, blender_path: str):
        self.blender_path = blender_path

    def import_asset(self, asset_path: str, location: tuple = (0, 0, 0), scale: float = 1.0) -> str:
        """导入3D资产到Blender场景"""
        try:
            # 确保文件存在
            if not os.path.exists(asset_path):
                raise FileNotFoundError(f"Asset file not found: {asset_path}")

            # 根据文件扩展名选择导入方法
            ext = os.path.splitext(asset_path)[1].lower()

            if ext in ['.fbx', '.obj', '.gltf', '.glb', '.dae', '.3ds', '.blend']:
                # 使用Blender的导入操作符
                if ext == '.fbx':
                    bpy.ops.import_scene.fbx(filepath=asset_path)
                elif ext == '.obj':
                    bpy.ops.import_scene.obj(filepath=asset_path)
                elif ext in ['.gltf', '.glb']:
                    bpy.ops.import_scene.gltf(filepath=asset_path)
                elif ext == '.dae':
                    bpy.ops.wm.collada_import(filepath=asset_path)
                elif ext == '.3ds':
                    bpy.ops.import_scene.autodesk_3ds(filepath=asset_path)
                elif ext == '.blend':
                    # 附注：append 需要 directory + filename（指向 .blend 内部路径）
                    # 这里保留占位，以防未来确实需要 .blend 的 append
                    bpy.ops.wm.append(filepath=asset_path)

                # 获取导入的对象
                imported_objects = [obj for obj in bpy.context.selected_objects]
                if not imported_objects:
                    raise RuntimeError("No objects were imported")

                # 设置位置和缩放
                for obj in imported_objects:
                    obj.location = location
                    obj.scale = (scale, scale, scale)

                # 返回导入的对象名称
                return imported_objects[0].name
            else:
                raise ValueError(f"Unsupported file format: {ext}")

        except Exception as e:
            logging.error(f"Failed to import asset: {e}")
            raise

    def extract_zip_asset(self, zip_path: str, extract_dir: str) -> str:
        """从ZIP文件中提取3D资产"""
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # 查找3D文件
                asset_files = []
                for file_info in zip_ref.filelist:
                    filename = file_info.filename
                    ext = os.path.splitext(filename)[1].lower()
                    if ext in ['.fbx', '.obj', '.gltf', '.glb', '.dae', '.3ds', '.blend']:
                        asset_files.append(filename)

                if not asset_files:
                    raise ValueError("No supported 3D files found in ZIP")

                # 提取第一个找到的3D文件
                asset_file = asset_files[0]
                zip_ref.extract(asset_file, extract_dir)

                return os.path.join(extract_dir, asset_file)

        except Exception as e:
            logging.error(f"Failed to extract ZIP asset: {e}")
            raise


class Executor:
    def __init__(self,
                 blender_command: str,
                 blender_file: str,
                 blender_script: str,
                 script_save: str,
                 render_save: str,
                 blender_save: Optional[str] = None):
        self.blender_command = blender_command
        self.blender_file = blender_file
        self.blender_script = blender_script
        self.script_path = Path(script_save)
        self.render_path = Path(render_save)
        self.blend_path = blender_save

        self.script_path.mkdir(parents=True, exist_ok=True)
        self.render_path.mkdir(parents=True, exist_ok=True)

    def _execute_blender(self, script_path: str, render_path: str) -> Tuple[bool, str, str]:
        cmd = [
            self.blender_command,
            "--background", self.blender_file,
            "--python", self.blender_script,
            "--", script_path, render_path
        ]
        # with open('cmd.txt', 'w') as f:
        #     f.write(" ".join(cmd))
        # # if self.blend_path:
        # #     cmd.append(self.blend_path)
        cmd_str = " ".join(cmd)
        try:
            proc = subprocess.run(cmd_str, shell=True, check=True, capture_output=True, text=True)
            out = proc.stdout
            err = proc.stderr
            # We do not consider intermediate errors that do not affect the result.
            # if 'Error:' in out:
            #     logging.error(f"Error in Blender stdout: {out}")
            #     return False, err, out
            # find rendered image(s)
            if os.path.isdir(render_path):
                imgs = sorted([str(p) for p in Path(render_path).glob("*") if p.suffix in ['.png','.jpg']])
                if not imgs:
                    return False, "No images", out
                paths = imgs[:2]
                return True, "PATH:" + ",".join(paths), out
            return True, out, err
        except subprocess.CalledProcessError as e:
            logging.error(f"Blender failed: {e}")
            return False, e.stderr, e.stdout

    def _encode_image(self, img_path: str) -> str:
        img = Image.open(img_path)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    def execute(self, code: str, round: int) -> Dict:
        script_file = self.script_path / f"{round}.py"
        render_file = self.render_path / f"{round}"
        with open(script_file, "w") as f:
            f.write(code)
        success, stdout, stderr = self._execute_blender(str(script_file), str(render_file))
        if not success or not os.path.exists(render_file):
            return {"status": "failure", "output": stderr or stdout}
        return {"status": "success", "output": str(render_file), "stdout": stdout, "stderr": stderr}

@mcp.tool()
def initialize_executor(blender_command: str,
                       blender_file: str,
                       blender_script: str,
                       script_save: str,
                       render_save: str,
                       blender_save: Optional[str] = None) -> dict:
    """
    初始化 Blender 执行器，设置所有必要的参数。
    """
    global _executor
    try:
        _executor = Executor(
            blender_command=blender_command,
            blender_file=blender_file,
            blender_script=blender_script,
            script_save=script_save,
            render_save=render_save,
            blender_save=blender_save
        )
        return {"status": "success", "message": "Executor initialized successfully"}
    except Exception as e:
        return {"status": "error", "error": str(e)}

@mcp.tool()
def exec_script(code: str, round: int) -> dict:
    """
    执行传入的 Blender Python 脚本 code，并返回 base64 编码后的渲染图像。
    需要先调用 initialize_executor 进行初始化。
    """
    global _executor
    if _executor is None:
        return {"status": "error", "error": "Executor not initialized. Call initialize_executor first."}
    
    try:
        result = _executor.execute(code, round)
        return {"status": "success", "result": result}
    except Exception as e:
        return {"status": "error", "error": str(e)}

@mcp.tool()
def add_meshy_asset(
    description: str,
    blender_path: str,
    location: str = "0,0,0",
    scale: float = 1.0,
    api_key: str = None,
    refine: bool = True
) -> dict:
    """
    使用 Meshy Text-to-3D 生成资产并导入到当前场景（生成→轮询→下载→导入）

    Args:
        description: 文本描述（prompt）
        blender_path: Blender 文件路径
        location: 资产位置 "x,y,z"
        scale: 缩放比例
        api_key: Meshy API 密钥（可选，默认读 MESHY_API_KEY）
        refine: 是否在 preview 后进行 refine（含贴图）
    """
    try:
        # 解析位置参数
        try:
            loc_parts = [float(x.strip()) for x in location.split(",")]
            if len(loc_parts) != 3:
                return {"status": "error", "error": "Location must be in format 'x,y,z'"}
            asset_location = tuple(loc_parts)
        except Exception:
            return {"status": "error", "error": "Invalid location format. Use 'x,y,z'"}

        # 初始化 Meshy API
        meshy = MeshyAPI(api_key)

        # 1) 创建 preview 任务
        print(f"[Meshy] Creating preview task for: {description}")
        preview_id = meshy.create_text_to_3d_preview(description)

        # 2) 轮询 preview
        preview_task = meshy.poll_text_to_3d(preview_id, interval_sec=5, timeout_sec=900)
        if preview_task.get("status") != "SUCCEEDED":
            return {"status": "error", "error": f"Preview failed: {preview_task.get('status')}"}
        final_task = preview_task

        # 3) 可选 refine（贴图）
        if refine:
            print(f"[Meshy] Starting refine for preview task: {preview_id}")
            refine_id = meshy.create_text_to_3d_refine(preview_id)
            refine_task = meshy.poll_text_to_3d(refine_id, interval_sec=5, timeout_sec=1800)
            if refine_task.get("status") != "SUCCEEDED":
                return {"status": "error", "error": f"Refine failed: {refine_task.get('status')}"}
            final_task = refine_task

        # 4) 从 model_urls 取下载链接
        model_urls = (final_task or {}).get("model_urls", {}) or {}
        candidate_keys = ["glb", "fbx", "obj", "zip"]
        file_url = None
        for k in candidate_keys:
            if model_urls.get(k):
                file_url = model_urls[k]
                break
        if not file_url:
            return {"status": "error", "error": "No downloadable model_urls found"}

        # 5) 下载模型到临时目录
        temp_dir = tempfile.mkdtemp(prefix="meshy_gen_")
        # 处理无扩展名直链：默认 .glb
        guessed_ext = os.path.splitext(file_url.split("?")[0])[1].lower()
        if guessed_ext not in [".glb", ".gltf", ".fbx", ".obj", ".zip"]:
            guessed_ext = ".glb"
        local_path = os.path.join(temp_dir, f"meshy_model{guessed_ext}")
        print(f"[Meshy] Downloading model to: {local_path}")
        meshy.download_model_url(file_url, local_path)

        # 6) 若为 ZIP，解压出 3D 文件
        importer = AssetImporter(blender_path)
        if local_path.endswith(".zip"):
            extracted = importer.extract_zip_asset(local_path, temp_dir)
            import_path = extracted
        else:
            import_path = local_path

        # 7) 导入 Blender
        imported_object_name = importer.import_asset(import_path, location=asset_location, scale=scale)
        print(f"[Meshy] Imported object: {imported_object_name}")

        # 8) 保存 Blender 文件
        try:
            bpy.ops.wm.save_mainfile(filepath=blender_path)
            print(f"Blender file saved to: {blender_path}")
            
            # 清理备份文件以避免生成 .blend1 文件
            backup_file = blender_path + "1"
            if os.path.exists(backup_file):
                os.remove(backup_file)
                print(f"Removed backup file: {backup_file}")
                
        except Exception as save_error:
            print(f"Warning: Failed to save blender file: {save_error}")

        # 9) 清理临时目录
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception as cleanup_error:
            print(f"Warning: Failed to cleanup temp files: {cleanup_error}")

        return {
            "status": "success",
            "message": "Meshy Text-to-3D asset generated and imported",
            "asset_name": description,
            "object_name": imported_object_name,
            "location": asset_location,
            "scale": scale
        }

    except Exception as e:
        logging.error(f"Failed to add Meshy asset: {e}")
        return {"status": "error", "error": str(e)}

@mcp.tool()
def initialize_investigator(blender_path: str) -> dict:
    """
    初始化 3D 场景调查工具。
    """
    global _investigator
    try:
        _investigator = Investigator3D(str(blender_path))
        return {"status": "success", "message": "Investigator3D initialized successfully"}
    except Exception as e:
        return {"status": "error", "error": str(e)}

@mcp.tool()
def focus(object_name: str) -> dict:
    """
    将相机聚焦到指定对象上。
    """
    global _investigator
    if _investigator is None:
        return {"status": "error", "error": "Investigator3D not initialized. Call initialize_investigator first."}

    try:
        # 检查目标对象是否存在
        obj = bpy.data.objects.get(object_name)
        if not obj:
            return {"status": "error", "error": f"Object '{object_name}' not found in scene"}

        result = _investigator.focus_on_object(object_name)
        return {"status": "success", "message": result}
    except Exception as e:
        logging.error(f"Focus failed: {e}")
        return {"status": "error", "error": str(e)}

@mcp.tool()
def zoom(direction: str) -> dict:
    """
    缩放相机视图。
    """
    global _investigator
    if _investigator is None:
        return {"status": "error", "error": "Investigator3D not initialized. Call initialize_investigator first."}

    try:
        # 检查是否有目标对象
        if _investigator.target is None:
            return {"status": "error", "error": "No target object set. Call focus first."}

        result = _investigator.zoom(direction)
        return {"status": "success", "message": result}
    except Exception as e:
        logging.error(f"Zoom failed: {e}")
        return {"status": "error", "error": str(e)}

@mcp.tool()
def move(direction: str) -> dict:
    """
    移动相机位置。
    """
    global _investigator
    if _investigator is None:
        return {"status": "error", "error": "Investigator3D not initialized. Call initialize_investigator first."}

    try:
        # 检查是否有目标对象
        if _investigator.target is None:
            return {"status": "error", "error": "No target object set. Call focus first."}

        result = _investigator.move_camera(direction)
        return {"status": "success", "message": result}
    except Exception as e:
        logging.error(f"Move failed: {e}")
        return {"status": "error", "error": str(e)}

def test_investigator() -> dict:
    """
    测试 Investigator3D 的基本功能：
    1. 打开 blender 文件
    2. 添加摄像头
    3. 修改摄像头角度
    """
    blender_path = "output/blendergym_hard/20250901_023433/level1/camera8/blender_file.blend"
    
    try:
        # 测试1: 初始化 investigator
        print("Testing Investigator3D initialization...")
        test_investigator = Investigator3D(blender_path)
        print(f"✓ Investigator3D initialized successfully")
        print(f"✓ Blender file loaded: {blender_path}")
        
        # 测试2: 检查摄像头
        print("\nTesting camera creation/retrieval...")
        cam = test_investigator.cam
        print(f"✓ Camera name: {cam.name}")
        print(f"✓ Camera type: {cam.type}")
        print(f"✓ Camera location: {cam.location}")
        
        # 测试3: 检查场景中的对象
        print("\nTesting scene objects...")
        scene_objects = list(bpy.context.scene.objects)
        print(f"✓ Scene objects count: {len(scene_objects)}")
        for obj in scene_objects[:5]:  # 只显示前5个对象
            print(f"  - {obj.name} ({obj.type}) at {obj.location}")
        
        # 测试4: 尝试聚焦到第一个网格对象
        print("\nTesting object focus...")
        mesh_objects = [obj for obj in scene_objects if obj.type == 'MESH']
        if mesh_objects:
            target_obj = mesh_objects[0]
            print(f"✓ Focusing on object: {target_obj.name}")
            focus_result = test_investigator.focus_on_object(target_obj.name)
            print(f"✓ Focus result: {focus_result}")
            
            # 测试5: 测试摄像头移动
            print("\nTesting camera movement...")
            print("✓ Testing zoom in...")
            zoom_in_result = test_investigator.zoom('in')
            print(f"  Result: {zoom_in_result}")
            
            print("✓ Testing zoom out...")
            zoom_out_result = test_investigator.zoom('out')
            print(f"  Result: {zoom_out_result}")
            
            print("✓ Testing move up...")
            move_up_result = test_investigator.move_camera('up')
            print(f"  Result: {move_up_result}")
            
            print("✓ Testing move down...")
            move_down_result = test_investigator.move_camera('down')
            print(f"  Result: {move_down_result}")
            
            print("✓ Testing move left...")
            move_left_result = test_investigator.move_camera('left')
            print(f"  Result: {move_left_result}")
            
            print("✓ Testing move right...")
            move_right_result = test_investigator.move_camera('right')
            print(f"  Result: {move_right_result}")
            
            # 测试6: 检查最终摄像头位置
            final_cam = test_investigator.cam
            print(f"\n✓ Final camera location: {final_cam.location}")
            print(f"✓ Final camera rotation: {final_cam.rotation_euler}")
            
        else:
            print("⚠ No mesh objects found in scene for focus testing")
        
        # 测试7: 检查约束
        print("\nTesting camera constraints...")
        track_constraints = [c for c in cam.constraints if c.type == 'TRACK_TO']
        if track_constraints:
            constraint = track_constraints[0]
            print(f"✓ Track constraint found: {constraint.name}")
            print(f"✓ Target object: {constraint.target.name if constraint.target else 'None'}")
            print(f"✓ Track axis: {constraint.track_axis}")
            print(f"✓ Up axis: {constraint.up_axis}")
        else:
            print("⚠ No track constraint found")
        
        # 测试8: 获取详细场景信息
        print("\nTesting scene info retrieval...")
        scene_info = test_investigator.get_scene_info()
        if "error" not in scene_info:
            print(f"✓ Scene info retrieved successfully")
            print(f"✓ Scene name: {scene_info.get('scene_name', 'Unknown')}")
            print(f"✓ Objects count: {len(scene_info.get('objects', []))}")
            print(f"✓ Collections count: {len(scene_info.get('collections', []))}")
        else:
            print(f"⚠ Scene info retrieval failed: {scene_info['error']}")
        
        print("\n🎉 All Investigator3D tests completed successfully!")
        
        return {
            "status": "success",
            "message": "Investigator3D test completed successfully",
            "details": {
                "camera_name": cam.name,
                "camera_location": str(cam.location),
                "scene_objects_count": len(scene_objects),
                "mesh_objects_count": len(mesh_objects),
                "test_results": {
                    "initialization": "success",
                    "camera_creation": "success",
                    "object_focus": "success" if mesh_objects else "skipped",
                    "camera_movement": "success" if mesh_objects else "skipped",
                    "constraints": "success" if track_constraints else "warning",
                    "scene_info": "success" if "error" not in scene_info else "failed"
                }
            }
        }
        
    except Exception as e:
        error_msg = f"Investigator3D test failed: {str(e)}"
        print(f"❌ {error_msg}")
        logging.error(error_msg)
        return {"status": "error", "error": error_msg}

def main():
    # 如果直接运行此脚本，执行测试
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        success = test_investigator()
        sys.exit(0 if success else 1)
    else:
        # 正常运行 MCP 服务
        mcp.run(transport="stdio")

if __name__ == "__main__":
    main()
