import bpy
import sys
import os

# ----------------- 参数解析 -----------------
def parse_args():
    argv = sys.argv
    if "--" not in argv:
        print("[ERROR] Usage:")
        print("  blender -b -P fix_glb_vertex_color.py -- input.glb output.glb")
        sys.exit(1)
    idx = argv.index("--")
    if len(argv) < idx + 3:
        print("[ERROR] Need input and output paths.")
        sys.exit(1)
    in_path = os.path.abspath(argv[idx + 1])
    out_path = os.path.abspath(argv[idx + 2])
    return in_path, out_path

# ----------------- 清空场景 -----------------
def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    # 保险起见再删一次
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)

# ----------------- 导入 glb -----------------
def import_glb(path):
    print(f"[INFO] Importing GLB: {path}")
    bpy.ops.import_scene.gltf(filepath=path)

# ----------------- 获取顶点颜色层名称 -----------------
def get_vertex_color_name(mesh):
    # Blender 3.x / 4.x: color_attributes
    if hasattr(mesh, "color_attributes") and mesh.color_attributes:
        # 找第一个颜色类型的 attribute
        for attr in mesh.color_attributes:
            if attr.domain in {"CORNER", "POINT"} and attr.data_type in {"BYTE_COLOR", "FLOAT_COLOR"}:
                return attr.name
        # fallback：第一个
        return mesh.color_attributes[0].name

    # 兼容老版本 vertex_colors
    if hasattr(mesh, "vertex_colors") and mesh.vertex_colors:
        return mesh.vertex_colors[0].name

    return None

# ----------------- 判断材质是否已经有贴图接 Base Color -----------------
def material_has_basecolor_texture(mat):
    if not mat or not mat.use_nodes:
        return False
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    principled = None
    for n in nodes:
        if n.type == "BSDF_PRINCIPLED":
            principled = n
            break
    if principled is None:
        return False

    base_input = principled.inputs.get("Base Color")
    if base_input is None:
        return False

    for link in links:
        if link.to_node == principled and link.to_socket == base_input:
            # 看看 from_node 是不是 Image Texture
            if link.from_node.type == "TEX_IMAGE":
                return True
    return False

# ----------------- 为一个材质接上顶点颜色 -----------------
def apply_vertex_color_to_material(mat, mesh, vcol_name):
    if mat is None:
        return
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # 清掉原来的节点（你也可以改成“只清除无用节点”，这里简单粗暴一点）
    for n in list(nodes):
        nodes.remove(n)

    # Attribute 节点（读取顶点颜色）
    attr = nodes.new("ShaderNodeAttribute")
    attr.attribute_name = vcol_name
    attr.location = (-300, 0)

    # Principled BSDF
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (0, 0)

    # 输出节点
    out = nodes.new("ShaderNodeOutputMaterial")
    out.location = (200, 0)

    # 连接：Attribute Color → Base Color → Output
    links.new(attr.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    print(f"[FIX] Material {mat.name}: vertex color '{vcol_name}' -> Base Color")

# ----------------- 处理所有 mesh 对象 -----------------
def process_scene():
    mesh_objects = [obj for obj in bpy.data.objects if obj.type == "MESH"]
    print(f"[INFO] Found {len(mesh_objects)} mesh objects")

    for obj in mesh_objects:
        mesh = obj.data
        vcol_name = get_vertex_color_name(mesh)
        if not vcol_name:
            print(f"[WARN] Object {obj.name} has NO vertex colors; skip.")
            continue

        print(f"[INFO] Object {obj.name} uses vertex color layer '{vcol_name}'")

        # 没有材质就新建一个
        if not obj.data.materials:
            mat = bpy.data.materials.new(name=f"{obj.name}_VertexColorMat")
            obj.data.materials.append(mat)
            mats = [mat]
        else:
            mats = [slot.material for slot in obj.material_slots if slot.material]

        for mat in mats:
            # 如果已有贴图接了 Base Color，就保持不动，避免破坏已有材质
            if material_has_basecolor_texture(mat):
                print(f"[INFO] Material {mat.name} already has texture on Base Color; keep it.")
                continue
            # 否则接上顶点颜色
            apply_vertex_color_to_material(mat, mesh, vcol_name)

# ----------------- 导出 glb -----------------
def export_glb(path):
    print(f"[INFO] Exporting fixed GLB to: {path}")
    os.makedirs(os.path.dirname(path), exist_ok=True)

    bpy.ops.export_scene.gltf(
        filepath=path,
        export_format='GLB',
        export_texcoords=True,
        export_normals=True,
        export_yup=True,
        export_apply=True,
        export_skins=True,
        export_animations=True,
        export_cameras=False,
        export_lights=False,
    )

# ----------------- 主函数 -----------------
def main():
    in_path, out_path = parse_args()
    print(f"[INFO] Input:  {in_path}")
    print(f"[INFO] Output: {out_path}")

    clear_scene()
    import_glb(in_path)
    process_scene()
    export_glb(out_path)
    print("[INFO] Done.")

if __name__ == "__main__":
    main()
