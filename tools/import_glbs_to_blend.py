import bpy
import sys
import os
import json
# mathutils 是 Blender 内置模块，只在 Blender 环境中可用
import mathutils  # type: ignore
from mathutils import Vector, Matrix, Euler  # type: ignore

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

# ----------------- 构建模型矩阵 -----------------
def build_model_matrix(rotation_3x3, translation):
    """
    Build a 4x4 model matrix from 3x3 rotation matrix and translation vector.
    
    Args:
        rotation_3x3: List of lists representing 3x3 rotation matrix
        translation: List of 3 floats [x, y, z]
    
    Returns:
        4x4 Matrix
    """
    # 确保 rotation 是 3x3
    if len(rotation_3x3) != 3 or any(len(row) != 3 for row in rotation_3x3):
        raise ValueError(f"Invalid rotation matrix shape: {rotation_3x3}")
    
    # 构建 4x4 矩阵
    # M_model = [[R00, R01, R02, Tx],
    #             [R10, R11, R12, Ty],
    #             [R20, R21, R22, Tz],
    #             [0,   0,   0,   1]]
    model_matrix = Matrix([
        [float(rotation_3x3[0][0]), float(rotation_3x3[0][1]), float(rotation_3x3[0][2]), float(translation[0])],
        [float(rotation_3x3[1][0]), float(rotation_3x3[1][1]), float(rotation_3x3[1][2]), float(translation[1])],
        [float(rotation_3x3[2][0]), float(rotation_3x3[2][1]), float(rotation_3x3[2][2]), float(translation[2])],
        [0.0, 0.0, 0.0, 1.0]
    ])
    
    return model_matrix

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
def import_glb_with_transform(glb_path, translation, translation_scale, rotation_3x3, scale, name_prefix=""):
    """
    Import GLB file and apply transformation matrix.
    
    Args:
        glb_path: Path to GLB file
        translation: Translation vector [x, y, z]
        translation_scale: Scale factor for translation (scalar or list)
        rotation_3x3: 3x3 rotation matrix (list of lists)
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
        root.name = f"{name_prefix}_{root.name}"
    
    # 1. 计算最终平移: T_final = translation * translation_scale
    trans_scale = get_translation_scale(translation_scale)
    if isinstance(translation, list) and len(translation) >= 3:
        # 处理嵌套列表
        if isinstance(translation[0], list):
            trans = translation[0]
        else:
            trans = translation
        T_final = [float(trans[i]) * trans_scale for i in range(3)]
    else:
        print(f"[WARN] Invalid translation format: {translation}, using [0, 0, 0]")
        T_final = [0.0, 0.0, 0.0]
    
    # 2. 构建模型矩阵 M_model
    if not isinstance(rotation_3x3, list) or len(rotation_3x3) != 3:
        print(f"[WARN] Invalid rotation matrix format: {rotation_3x3}, using identity")
        rotation_3x3 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    
    M_model = build_model_matrix(rotation_3x3, T_final)
    
    # 3. 应用坐标系统修正: M_blender = M_fix * M_model
    M_fix = get_coordinate_fix_matrix()
    M_blender = M_fix @ M_model
    
    # 4. 应用矩阵变换到根对象
    # 注意：Blender 的 matrix_world 是 4x4，我们需要确保对象在正确的模式下
    root.matrix_world = M_blender
    
    # 5. 应用缩放（在矩阵变换之后）
    scale_vec = get_scale_vector(scale)
    # 缩放是相对于对象本地坐标系的，所以需要小心处理
    # 我们可以通过修改 matrix_world 来应用缩放，或者直接设置 scale
    # 为了简单，我们直接设置 scale 属性
    root.scale = scale_vec
    
    print(f"[INFO] Applied transform - Translation: {T_final}, Scale: {scale_vec}")
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
    
    # 导入所有对象
    success_count = 0
    for idx, obj_data in enumerate(objects_data):
        glb_path = obj_data.get("glb")
        if not glb_path:
            print(f"[WARN] No 'glb' key for object {idx}, skipping")
            continue
        
        # 获取变换参数
        translation = obj_data.get("translation")
        translation_scale = obj_data.get("translation_scale", 1.0)
        rotation = obj_data.get("rotation")
        scale = obj_data.get("scale", 1.0)
        
        if translation is None or rotation is None:
            print(f"[WARN] Missing translation or rotation for object {idx}, skipping")
            continue
        
        root = import_glb_with_transform(
            glb_path=glb_path,
            translation=translation,
            translation_scale=translation_scale,
            rotation_3x3=rotation,
            scale=scale,
            name_prefix=f"obj_{idx}"
        )
        if root:
            success_count += 1
    
    print(f"[INFO] Successfully imported {success_count}/{len(objects_data)} GLB files")
    
    save_blend(blend_path)
    print("[INFO] Done.")

if __name__ == "__main__":
    main()
