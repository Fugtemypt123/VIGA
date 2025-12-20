import bpy
import sys
import os
import json
from mathutils import Vector, Euler  # type: ignore

# ----------------- 参数解析 -----------------
def parse_args():
    argv = sys.argv
    if "--" not in argv:
        print("[ERROR] Usage:")
        print("  blender -b -P import_glbs_to_blend.py -- transforms.json output.blend")
        sys.exit(1)
    idx = argv.index("--")
    if len(argv) < idx + 3:
        print("[ERROR] Need transforms JSON file and output path.")
        sys.exit(1)
    transforms_json_path = os.path.abspath(argv[idx + 1])
    blend_path = os.path.abspath(argv[idx + 2])
    return transforms_json_path, blend_path

# ----------------- 清空场景 -----------------
def clear_scene():
    """Delete all existing mesh objects in the scene."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)

# ----------------- 设置相机 -----------------
def setup_camera():
    """Setup camera at world origin with correct orientation and FOV."""
    if "Camera" not in bpy.data.objects:
        bpy.ops.object.camera_add()
        camera = bpy.context.active_object
        camera.name = "Camera"
    else:
        camera = bpy.data.objects["Camera"]
    
    camera.location = Vector((0.0, 0.0, 0.0))
    camera.rotation_euler = Euler((0.0, 0.0, 0.0), 'XYZ')
    camera.data.lens_unit = 'FOV'
    camera.data.angle = 60.0 * (3.14159265359 / 180.0)
    camera.data.sensor_fit = 'VERTICAL'
    
    print(f"[INFO] Camera setup complete: location={camera.location}, FOV=60°")

# ----------------- 设置环境光 -----------------
def setup_lighting():
    """Setup environment lighting for the scene."""
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    
    world.use_nodes = True
    bg_node = world.node_tree.nodes.get("Background")
    if bg_node is None:
        bg_node = world.node_tree.nodes.new(type='ShaderNodeBackground')
    
    bg_node.inputs['Strength'].default_value = 1.0
    bg_node.inputs['Color'].default_value = (1.0, 1.0, 1.0, 1.0)
    
    output_node = world.node_tree.nodes.get("World Output")
    if output_node:
        world.node_tree.links.new(bg_node.outputs['Background'], output_node.inputs['Surface'])
    
    if "Sun" not in bpy.data.objects:
        bpy.ops.object.light_add(type='SUN')
        sun = bpy.context.active_object
        sun.name = "Sun"
        sun.location = Vector((5.0, 5.0, 10.0))
        sun.data.energy = 3.0
        sun.data.angle = 0.261799
    
    print(f"[INFO] Lighting setup complete")

# ----------------- 导入 glb -----------------
def import_glb(glb_path, name_prefix=""):
    """
    Import GLB file directly without any transformation.
    """
    print(f"[INFO] Importing GLB: {glb_path}")
    if not os.path.exists(glb_path):
        print(f"[WARN] GLB file not found: {glb_path}, skipping")
        return None
    
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.import_scene.gltf(filepath=glb_path)
    
    imported_objects = bpy.context.selected_objects
    if not imported_objects:
        print(f"[WARN] No objects imported from {glb_path}")
        return None
    
    # 找到根对象
    root = None
    for obj in imported_objects:
        if obj.parent not in imported_objects:
            root = obj
            break
    if not root:
        root = imported_objects[0]
    
    if name_prefix:
        root.name = name_prefix
    
    # 对所有导入的 MESH 对象执行 origin_set 操作
    mesh_count = 0
    for obj in imported_objects:
        if obj.type == 'MESH':
            # 必须设为 Active 才能执行 Ops
            bpy.context.view_layer.objects.active = obj
            # 执行原点重设
            bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='MEDIAN')
            mesh_count += 1
            print(f"[INFO] Set origin for mesh: {obj.name}, location: {obj.location}")
    
    print(f"[INFO] Imported {len(imported_objects)} objects from {glb_path} (processed {mesh_count} meshes)")
    return root

# ----------------- 保存为 .blend 文件 -----------------
def save_blend(path):
    print(f"[INFO] Saving Blender file to: {path}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=path)
    print(f"[INFO] Saved: {path}")

# ----------------- 主函数 -----------------
def main():
    transforms_json_path, blend_path = parse_args()
    print(f"[INFO] Loading transforms from: {transforms_json_path}")
    print(f"[INFO] Output: {blend_path}")
    
    with open(transforms_json_path, 'r') as f:
        objects_data = json.load(f)
    
    print(f"[INFO] Importing {len(objects_data)} GLB files")
    
    clear_scene()
    setup_camera()
    setup_lighting()
    
    success_count = 0
    for idx, obj_data in enumerate(objects_data):
        glb_path = obj_data.get("glb") or obj_data.get("glb_path")
        if not glb_path:
            print(f"[WARN] No 'glb' or 'glb_path' key for object {idx}, skipping")
            continue
        
        glb_filename = os.path.basename(glb_path)
        object_name = os.path.splitext(glb_filename)[0]
        
        root = import_glb(glb_path=glb_path, name_prefix=object_name)
        if root:
            success_count += 1
    
    print(f"[INFO] Successfully imported {success_count}/{len(objects_data)} GLB files")
    
    save_blend(blend_path)
    print("[INFO] Done.")

if __name__ == "__main__":
    main()
