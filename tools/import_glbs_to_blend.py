import bpy
import sys
import os
import json
# mathutils 是 Blender 内置模块，只在 Blender 环境中可用
import mathutils  # type: ignore
from mathutils import Vector, Matrix, Euler, Quaternion  # type: ignore

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
    # 获取或创建相机
    if "Camera" not in bpy.data.objects:
        bpy.ops.object.camera_add()
        camera = bpy.context.active_object
        camera.name = "Camera"
    else:
        camera = bpy.data.objects["Camera"]
    
    # 设置相机位置和旋转
    camera.location = Vector((0.0, 0.0, 0.0))
    camera.rotation_euler = Euler((0.0, 0.0, 0.0), 'XYZ')
    
    # 设置相机参数
    camera.data.lens_unit = 'FOV'
    camera.data.angle = 60.0 * (3.14159265359 / 180.0)  # 60 degrees in radians
    camera.data.sensor_fit = 'VERTICAL'
    
    print(f"[INFO] Camera setup complete: location={camera.location}, FOV=60°")

# ----------------- 设置环境光 -----------------
def setup_lighting():
    """Setup environment lighting for the scene."""
    # 设置 World 环境光
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    
    # 使用节点系统设置环境光
    world.use_nodes = True
    bg_node = world.node_tree.nodes.get("Background")
    if bg_node is None:
        bg_node = world.node_tree.nodes.new(type='ShaderNodeBackground')
    
    # 设置环境光强度和颜色（白色环境光，强度适中）
    bg_node.inputs['Strength'].default_value = 1.0  # 环境光强度
    bg_node.inputs['Color'].default_value = (1.0, 1.0, 1.0, 1.0)  # 白色
    
    # 确保环境光节点连接到输出
    output_node = world.node_tree.nodes.get("World Output")
    if output_node:
        world.node_tree.links.new(bg_node.outputs['Background'], output_node.inputs['Surface'])
    
    # 添加一个 Sun 光源作为主光源
    if "Sun" not in bpy.data.objects:
        bpy.ops.object.light_add(type='SUN')
        sun = bpy.context.active_object
        sun.name = "Sun"
        sun.location = Vector((5.0, 5.0, 10.0))
        sun.data.energy = 3.0  # 光源强度
        sun.data.angle = 0.261799  # 约 15 度的角度
    else:
        sun = bpy.data.objects["Sun"]
    
    print(f"[INFO] Lighting setup complete: World background strength=1.0, Sun energy=3.0")

# ----------------- 计算坐标系统修正矩阵 -----------------
def get_coordinate_fix_matrix():
    """
    Create the coordinate system correction matrix M_fix.
    Converts from OpenCV (X Right, Y Down, Z Forward) to Blender (X Right, Y Forward, Z Up).
    This is a 180-degree rotation around X-axis.
    """
    # M_fix = [[1, 0, 0, 0],
    #           [0, -1, 0, 0],
    #           [0, 0, -1, 0],
    #           [0, 0, 0, 1]]
    return Matrix([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

# ----------------- 处理 translation_scale -----------------
def get_translation_scale(translation_scale):
    """
    Extract translation scale value.
    Can be a scalar or a list (take first element if list).
    """
    if isinstance(translation_scale, (int, float)):
        return float(translation_scale)
    elif isinstance(translation_scale, list) and len(translation_scale) > 0:
        return float(translation_scale[0])
    else:
        return 1.0  # Default if invalid

# ----------------- 处理 scale -----------------
def get_scale_vector(scale):
    """
    Convert scale to Vector.
    Can be a scalar (applies to all axes) or a list [x, y, z].
    """
    if isinstance(scale, (int, float)):
        return Vector((float(scale), float(scale), float(scale)))
    elif isinstance(scale, list):
        if len(scale) == 1:
            s = float(scale[0])
            return Vector((s, s, s))
        elif len(scale) >= 3:
            return Vector((float(scale[0]), float(scale[1]), float(scale[2])))
    return Vector((1.0, 1.0, 1.0))  # Default

# ----------------- 导入 glb 并应用变换 -----------------
def import_glb_with_transform(glb_path, translation, translation_scale, rotation_quaternion, scale, name_prefix=""):
    """
    Import GLB file and apply transformation.
    
    Args:
        glb_path: Path to GLB file
        translation: Translation vector [x, y, z]
        translation_scale: Scale factor for translation (scalar or list)
        rotation_quaternion: Quaternion [w, x, y, z] format
        scale: Object scale (scalar or list [x, y, z])
        name_prefix: Prefix for object name
    """
    print(f"[INFO] Importing GLB: {glb_path}")
    if not os.path.exists(glb_path):
        print(f"[WARN] GLB file not found: {glb_path}, skipping")
        return None
    
    # 保存当前选中的对象
    bpy.ops.object.select_all(action='DESELECT')
    
    # 导入 GLB
    bpy.ops.import_scene.gltf(filepath=glb_path)
    
    # 获取导入的对象
    imported_objects = bpy.context.selected_objects
    if not imported_objects:
        print(f"[WARN] No objects imported from {glb_path}")
        return None
    
    # 找到根对象（如果有多个对象，找到根对象）
    root = None
    for obj in imported_objects:
        if obj.parent not in imported_objects:
            root = obj
            break
    if not root:
        root = imported_objects[0]
    
    if name_prefix:
        # 直接使用 name_prefix 作为物体名称，而不是添加前缀
        root.name = name_prefix
    
    # 1. 计算最终平移: T_final = translation * translation_scale
    trans_scale_val = 1
    if isinstance(translation, list) and len(translation) >= 3:
        # 处理嵌套列表
        if isinstance(translation[0], list):
            trans = translation[0]
        else:
            trans = translation
        T_final = Vector([float(trans[i]) * trans_scale_val for i in range(3)])
    else:
        print(f"[WARN] Invalid translation format: {translation}, using [0, 0, 0]")
        T_final = Vector((0.0, 0.0, 0.0))
    
    # 2. 解析四元数 [w, x, y, z]
    if not isinstance(rotation_quaternion, list) or len(rotation_quaternion) != 4:
        print(f"[WARN] Invalid quaternion format: {rotation_quaternion}, using identity")
        quat = Quaternion((1.0, 0.0, 0.0, 0.0))  # 单位四元数
    else:
        # Blender 的 Quaternion 使用 (w, x, y, z) 格式
        quat = Quaternion((float(rotation_quaternion[0]), float(rotation_quaternion[1]), 
                          float(rotation_quaternion[2]), float(rotation_quaternion[3])))
    
    # 3. 设置位置、旋转和缩放
    root.location = T_final
    root.rotation_mode = 'QUATERNION'
    root.rotation_quaternion = quat
    
    # 5. 应用缩放
    scale_vec = get_scale_vector(scale)
    root.scale = scale_vec
    
    print(f"[INFO] Applied transform - Translation: {T_final}, Rotation (quaternion): {quat}, Scale: {scale_vec}")
    print(f"[INFO] Imported {len(imported_objects)} objects from {glb_path}")
    
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
    
    # 加载位置信息
    with open(transforms_json_path, 'r') as f:
        objects_data = json.load(f)
    
    print(f"[INFO] Importing {len(objects_data)} GLB files with transforms")
    
    # 清空场景
    clear_scene()
    
    # 设置相机
    setup_camera()
    
    # 设置环境光
    setup_lighting()
    
    # 导入所有对象
    success_count = 0
    for idx, obj_data in enumerate(objects_data):
        glb_path = obj_data.get("glb") or obj_data.get("glb_path")
        if not glb_path:
            print(f"[WARN] No 'glb' or 'glb_path' key for object {idx}, skipping")
            continue
        
        # 从 glb_path 中提取文件名（不含扩展名）作为物体名称
        glb_filename = os.path.basename(glb_path)
        object_name = os.path.splitext(glb_filename)[0]  # 去掉 .glb 扩展名
        
        # 获取变换参数
        translation = obj_data.get("translation")
        translation_scale = obj_data.get("translation_scale", 1.0)
        rotation = obj_data.get("rotation")
        scale = obj_data.get("scale", 1.0)
        
        if translation is None or rotation is None:
            print(f"[WARN] Missing translation or rotation for object {idx} ({object_name}), skipping")
            continue
        
        root = import_glb_with_transform(
            glb_path=glb_path,
            translation=translation,
            translation_scale=translation_scale,
            rotation_quaternion=rotation,
            scale=scale,
            name_prefix=object_name
        )
        if root:
            success_count += 1
    
    print(f"[INFO] Successfully imported {success_count}/{len(objects_data)} GLB files")
    
    save_blend(blend_path)
    print("[INFO] Done.")

if __name__ == "__main__":
    main()
