# blender_server.py
import bpy
import json
import math
import tempfile
import os
import sys
from pathlib import Path
import logging
from mcp.server.fastmcp import FastMCP

# 创建全局 MCP 实例
mcp = FastMCP("scene-server")

# 全局工具实例
_investigator = None

# 内置工具

class GetSceneInfo:
    def __init__(self, blender_path: str):
        bpy.ops.wm.open_mainfile(filepath=str(blender_path))

    def get_info(self) -> dict:
        try:
            scene_info = {"objects": [], "materials": [], "lights": [], "cameras": [], "render_settings": {}}
            for obj in bpy.data.objects:
                obj_info = {"name": obj.name, "type": obj.type,
                            "location": list(obj.matrix_world.translation),
                            "rotation": list(obj.rotation_euler),
                            "scale": list(obj.scale),
                            "visible": not (obj.hide_viewport or obj.hide_render),
                            "active": obj == bpy.context.active_object}
                if obj.type == 'MESH':
                    obj_info["vertices"] = len(obj.data.vertices)
                    obj_info["faces"] = len(obj.data.polygons)
                    obj_info["materials"] = [mat.name for mat in obj.material_slots if mat.material]
                scene_info["objects"].append(obj_info)

            for mat in bpy.data.materials:
                scene_info["materials"].append({
                    "name": mat.name,
                    "use_nodes": mat.use_nodes,
                    "diffuse_color": list(mat.diffuse_color),
                    "metallic": getattr(mat, 'metallic', None),
                    "roughness": getattr(mat, 'roughness', None)
                })

            for light in [o for o in bpy.data.objects if o.type == 'LIGHT']:
                scene_info["lights"].append({
                    "name": light.name,
                    "type": light.data.type,
                    "energy": light.data.energy,
                    "color": list(light.data.color),
                    "location": list(light.matrix_world.translation),
                    "rotation": list(light.rotation_euler)
                })

            for cam in [o for o in bpy.data.objects if o.type == 'CAMERA']:
                scene = bpy.context.scene
                scene_info["cameras"].append({
                    "name": cam.name,
                    "lens": cam.data.lens,
                    "location": list(cam.matrix_world.translation),
                    "rotation": list(cam.rotation_euler),
                    "is_active": cam == scene.camera,
                    "dof_distance": cam.data.dof_distance if cam.data.dof.use_dof else None,
                    "dof_aperture_fstop": cam.data.dof.aperture_fstop if cam.data.dof.use_dof else None
                })

            rnd = bpy.context.scene.render
            scene_info["render_settings"] = {
                "resolution_x": rnd.resolution_x,
                "resolution_y": rnd.resolution_y,
                "resolution_percentage": rnd.resolution_percentage,
                "engine": rnd.engine,
                "samples": bpy.context.scene.cycles.samples if rnd.engine == 'CYCLES' else None
            }

            return scene_info
        except Exception as e:
            logging.error(f"scene info error: {e}")
            return {}

class Investigator3D:
    def __init__(self, thoughtprocess_save: str, blender_path: str, round_num: int):
        self.blender_path = blender_path
        self.thoughtprocess_save = thoughtprocess_save
        self.round_num = round_num
        self._load_blender_file()
        self.base = Path(thoughtprocess_save) / f"investigator_{round_num}"
        self.base.mkdir(parents=True, exist_ok=True)
        self.cam = self._get_or_create_cam()
        self.target = None
        self.radius = 5.0
        self.theta = 0.0
        self.phi = 0.0
        self.count = 0

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

    def _render(self):
        bpy.context.scene.camera = self.cam
        bpy.context.scene.render.engine = 'CYCLES'
        bpy.context.scene.render.filepath = str(self.base / f"{self.count:03d}.png")
        bpy.ops.render.render(write_still=True)
        out = bpy.context.scene.render.filepath
        self.count += 1
        return out

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
        return self._render()

    def zoom(self, direction: str) -> str:
        if direction == 'in':
            self.radius = max(1, self.radius-3)
        elif direction == 'out':
            self.radius += 3
        return self._update_and_render()

    def move_camera(self, direction: str) -> str:
        step = self.radius
        theta_step = step/(self.radius*math.cos(self.phi))
        phi_step = step/self.radius
        if direction=='up': self.phi = min(math.pi/2-0.1, self.phi+phi_step)
        elif direction=='down': self.phi = max(-math.pi/2+0.1, self.phi-phi_step)
        elif direction=='left': self.theta -= theta_step
        elif direction=='right': self.theta += theta_step
        return self._update_and_render()

    def _update_and_render(self) -> str:
        t = self.target.matrix_world.translation
        x = self.radius*math.cos(self.phi)*math.cos(self.theta)
        y = self.radius*math.cos(self.phi)*math.sin(self.theta)
        z = self.radius*math.sin(self.phi)
        self.cam.matrix_world.translation = (t.x+x, t.y+y, t.z+z)
        return self._render()

@mcp.tool()
def get_scene_info(blender_path: str) -> dict:
    """
    获取 Blender 场景信息，包括对象、材质、灯光、相机和渲染设置。
    """
    try:
        info = GetSceneInfo(blender_path).get_info()
        return {"status": "success", "info": info}
    except Exception as e:
        logging.error(f"Failed to get scene info: {e}")
        return {"status": "error", "error": str(e)}

@mcp.tool()
def initialize_investigator(thoughtprocess_save: str, blender_path: str, round_num: int) -> dict:
    """
    初始化 3D 场景调查工具。
    """
    global _investigator
    try:
        _investigator = Investigator3D(thoughtprocess_save, str(blender_path), round_num)
        return {"status": "success", "message": "Investigator3D initialized successfully"}
    except Exception as e:
        return {"status": "error", "error": str(e)}

@mcp.tool()
def focus(blender_path: str, save_dir: str, round_num: int, object_name: str) -> dict:
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
        
        img = _investigator.focus_on_object(object_name)
        return {"status": "success", "image": str(img)}
    except Exception as e:
        logging.error(f"Focus failed: {e}")
        return {"status": "error", "error": str(e)}

@mcp.tool()
def zoom(save_dir: str, direction: str) -> dict:
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
        
        img = _investigator.zoom(direction)
        return {"status": "success", "image": str(img)}
    except Exception as e:
        logging.error(f"Zoom failed: {e}")
        return {"status": "error", "error": str(e)}

@mcp.tool()
def move(save_dir: str, direction: str) -> dict:
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
        
        img = _investigator.move_camera(direction)
        return {"status": "success", "image": str(img)}
    except Exception as e:
        logging.error(f"Move failed: {e}")
        return {"status": "error", "error": str(e)}

def main():
    # 检查是否直接运行此脚本（用于测试）
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        print("Running scene.py tools test...")
        test_tools()
    else:
        # 正常运行 MCP 服务器
        mcp.run(transport="stdio")

def test_tools():
    """测试所有工具函数"""
    print("=" * 50)
    print("Testing Scene Tools")
    print("=" * 50)
    
    # 设置测试路径
    blender_file = "data/blendergym/_hard1/blender_file.blend"
    test_save_dir = "output/scene_test"
    test_round = 1
    
    # 检查 blender 文件是否存在
    if not os.path.exists(blender_file):
        print(f"⚠ Blender file not found: {blender_file}")
        print("Skipping all tests.")
        return
    
    print(f"✓ Using blender file: {blender_file}")
    
    # 测试 1: 获取场景信息
    print("\n1. Testing get_scene_info...")
    try:
        result = get_scene_info(blender_file)
        print(f"Result: {result}")
        if result.get("status") == "success":
            print("✓ get_scene_info passed")
            info = result.get("info", {})
            print(f"  - Objects: {len(info.get('objects', []))}")
            print(f"  - Materials: {len(info.get('materials', []))}")
            print(f"  - Lights: {len(info.get('lights', []))}")
            print(f"  - Cameras: {len(info.get('cameras', []))}")
            
            # 获取第一个对象名称用于后续测试
            objects = info.get("objects", [])
            if not objects:
                print("⚠ No objects found in scene, skipping camera tests")
                return
            first_object = objects[0]["name"]
            print(f"  - Will focus on: {first_object}")
        else:
            print("✗ get_scene_info failed")
            return
    except Exception as e:
        print(f"✗ get_scene_info failed with exception: {e}")
        return
    
    # 测试 2: 初始化调查工具
    print("\n2. Testing initialize_investigator...")
    try:
        result = initialize_investigator(test_save_dir, blender_file, test_round)
        print(f"Result: {result}")
        if result.get("status") == "success":
            print("✓ initialize_investigator passed")
        else:
            print("✗ initialize_investigator failed")
            return
    except Exception as e:
        print(f"✗ initialize_investigator failed with exception: {e}")
        return
    
    # 测试 3: 聚焦对象
    print("\n3. Testing focus...")
    try:
        result = focus(blender_file, test_save_dir, test_round, first_object)
        print(f"Result: {result}")
        if result.get("status") == "success":
            print("✓ focus passed")
            print(f"  - Focused on: {first_object}")
            print(f"  - Image saved: {result.get('image', 'N/A')}")
        else:
            print("✗ focus failed")
            return
    except Exception as e:
        print(f"✗ focus failed with exception: {e}")
        return
    
    # 测试 4: 缩放功能
    print("\n4. Testing zoom...")
    try:
        result = zoom(test_save_dir, "in")
        print(f"Result: {result}")
        if result.get("status") == "success":
            print("✓ zoom passed")
            print(f"  - Image saved: {result.get('image', 'N/A')}")
        else:
            print("✗ zoom failed")
    except Exception as e:
        print(f"✗ zoom failed with exception: {e}")
    
    # 测试 5: 移动功能
    print("\n5. Testing move...")
    try:
        result = move(test_save_dir, "up")
        print(f"Result: {result}")
        if result.get("status") == "success":
            print("✓ move passed")
            print(f"  - Image saved: {result.get('image', 'N/A')}")
        else:
            print("✗ move failed")
    except Exception as e:
        print(f"✗ move failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print("Test completed!")
    print("=" * 50)
    print(f"\nTest files saved to: {test_save_dir}")
    print("\nTo run the MCP server normally, use:")
    print("python scene.py")
    print("\nTo run tests, use:")
    print("python scene.py --test")

if __name__ == "__main__":
    main()
