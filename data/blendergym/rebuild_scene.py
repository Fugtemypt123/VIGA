import bpy, os, json
from pathlib import Path

# ===== 配置 =====
OUTPUT_DIR = Path(bpy.data.filepath).parent if bpy.data.filepath else Path.cwd()
LIB_PATH = OUTPUT_DIR / "assets.blend"
SCRIPT_PATH = OUTPUT_DIR / "rebuild_scene.py"

# 导出哪些类型的 DataBlock（可按需增减）
DATABLOCK_KEYS = {
    "meshes": set(),
    "materials": set(),
    "images": set(),
    "node_groups": set(),
    "armatures": set(),
    "actions": set(),
    "curves": set(),
    "texts": set(),        # 可能有脚本
    "grease_pencils": set()
}

# ===== 收集 DataBlocks =====
def collect_datablocks():
    # 物体引用的数据
    for obj in bpy.data.objects:
        if obj.type == 'MESH' and obj.data:
            DATABLOCK_KEYS["meshes"].add(obj.data)
        if obj.type == 'ARMATURE' and obj.data:
            DATABLOCK_KEYS["armatures"].add(obj.data)
        if obj.type == 'CURVE' and obj.data:
            DATABLOCK_KEYS["curves"].add(obj.data)
        # 材质
        for ms in obj.material_slots:
            if ms.material:
                DATABLOCK_KEYS["materials"].add(ms.material)
                # 材质里的节点组、贴图
                if ms.material.use_nodes and ms.material.node_tree:
                    for n in ms.material.node_tree.nodes:
                        if n.type == 'GROUP' and n.node_tree:
                            DATABLOCK_KEYS["node_groups"].add(n.node_tree)
                        if hasattr(n, "image") and n.image:
                            DATABLOCK_KEYS["images"].add(n.image)
        # 骨骼动画/动作
        if obj.animation_data and obj.animation_data.action:
            DATABLOCK_KEYS["actions"].add(obj.animation_data.action)
    # 世界与其贴图/节点组
    if bpy.data.worlds:
        for w in bpy.data.worlds:
            if w.use_nodes and w.node_tree:
                for n in w.node_tree.nodes:
                    if n.type == 'GROUP' and n.node_tree:
                        DATABLOCK_KEYS["node_groups"].add(n.node_tree)
                    if hasattr(n, "image") and n.image:
                        DATABLOCK_KEYS["images"].add(n.image)

    # 场景里用到的文本(Text)等（可选）
    for t in bpy.data.texts:
        DATABLOCK_KEYS["texts"].add(t)

collect_datablocks()

# ===== 写出库文件 assets.blend =====
to_write = set()
for k, s in DATABLOCK_KEYS.items():
    to_write |= set(s)

# 也把世界、灯光、相机数据块纳入
to_write |= set(b.data for b in bpy.data.objects if b.data)

# 注意：libraries.write 需要 4.0+；老版本可用 bpy.data.libraries.write 同名 API
bpy.data.libraries.write(str(LIB_PATH), to_write, fake_user=True)

# ===== 序列化场景结构（对象、集合、父子关系、变换、约束、修改器、动画等）=====
scene_spec = {
    "render": {
        "engine": bpy.context.scene.render.engine,
        "fps": bpy.context.scene.render.fps,
        "resolution_x": bpy.context.scene.render.resolution_x,
        "resolution_y": bpy.context.scene.render.resolution_y,
        "resolution_percentage": bpy.context.scene.render.resolution_percentage,
        "film_transparent": bpy.context.scene.render.film_transparent,
    },
    "world": bpy.context.scene.world.name if bpy.context.scene.world else None,
    "collections": {},
    "objects": []
}

# 集合（层级）
def walk_collection(col, parent_path=None):
    entry = {
        "name": col.name,
        "children": [c.name for c in col.children],
        "objects": [o.name for o in col.objects],
        "parent": parent_path
    }
    scene_spec["collections"][col.name] = entry
    for c in col.children:
        walk_collection(c, col.name)

walk_collection(bpy.context.scene.collection, parent_path=None)

# 物体
def mat_slot_names(obj):
    return [ms.material.name if ms.material else None for ms in obj.material_slots]

def modifier_specs(obj):
    specs = []
    for m in obj.modifiers:
        specs.append({"name": m.name, "type": m.type})
    return specs

def constraint_specs(obj):
    specs = []
    for c in obj.constraints:
        specs.append({"name": c.name, "type": c.type, "target": getattr(c.target, "name", None)})
    return specs

for obj in bpy.data.objects:
    spec = {
        "name": obj.name,
        "type": obj.type,
        "data_name": obj.data.name if obj.data else None,
        "matrix_world": [list(row) for row in obj.matrix_world],
        "parent": obj.parent.name if obj.parent else None,
        "parent_type": obj.parent_type if obj.parent else None,
        "materials": mat_slot_names(obj),
        "modifiers": modifier_specs(obj),
        "constraints": constraint_specs(obj),
        "hide_viewport": obj.hide_viewport,
        "hide_render": obj.hide_render,
        "animation_action": obj.animation_data.action.name if (obj.animation_data and obj.animation_data.action) else None,
        "collections": [c.name for c in obj.users_collection],
    }
    # 相机/灯光额外参数（示例，可按需扩展）
    if obj.type == "CAMERA" and obj.data:
        spec["camera"] = {"lens": obj.data.lens, "clip_start": obj.data.clip_start, "clip_end": obj.data.clip_end}
    if obj.type == "LIGHT" and obj.data:
        spec["light"] = {"type": obj.data.type, "energy": obj.data.energy, "color": list(obj.data.color)}
    scene_spec["objects"].append(spec)

STRUCT_JSON = OUTPUT_DIR / "scene_structure.json"
STRUCT_JSON.write_text(json.dumps(scene_spec, indent=2), encoding="utf-8")

# ===== 生成重建脚本 rebuild_scene.py =====
rebuild = f"""# Auto-generated by export script
import bpy, json, mathutils
from pathlib import Path

HERE = Path(__file__).parent
LIB_PATH = HERE / "assets.blend"
STRUCT_JSON = HERE / "scene_structure.json"

# 读取结构描述
spec = json.loads(STRUCT_JSON.read_text(encoding="utf-8"))

# 清空当前默认场景
bpy.ops.wm.read_homefile(use_empty=True)

# 渲染/世界
bpy.context.scene.render.engine = spec["render"]["engine"]
bpy.context.scene.render.fps = spec["render"]["fps"]
bpy.context.scene.render.resolution_x = spec["render"]["resolution_x"]
bpy.context.scene.render.resolution_y = spec["render"]["resolution_y"]
bpy.context.scene.render.resolution_percentage = spec["render"]["resolution_percentage"]
bpy.context.scene.render.film_transparent = spec["render"]["film_transparent"]

# 先创建集合层级
name_to_collection = {{}}
def ensure_collection(name):
    col = name_to_collection.get(name)
    if col: return col
    col = bpy.data.collections.new(name)
    name_to_collection[name] = col
    return col

# 根集合用现有 Scene Collection
root = bpy.context.scene.collection
name_to_collection[root.name] = root

# 预创建所有集合
for cname, centry in spec["collections"].items():
    if cname not in name_to_collection:
        ensure_collection(cname)

# 建立父子层级
for cname, centry in spec["collections"].items():
    if centry["parent"] is None:
        # 根（Scene Collection）里已经有根集合，按需要链接
        if cname != root.name and cname not in [c.name for c in root.children]:
            root.children.link(name_to_collection[cname])
    else:
        p = name_to_collection[centry["parent"]]
        c = name_to_collection[cname]
        if c.name not in [x.name for x in p.children]:
            p.children.link(c)

# 打开库（assets.blend）并把需要的数据块 append 进来
with bpy.data.libraries.load(str(LIB_PATH), link=False) as (data_from, data_to):
    # 按名称全量导入（简单起见）
    data_to.meshes = list(data_from.meshes)
    data_to.materials = list(data_from.materials)
    data_to.images = list(data_from.images)
    data_to.node_groups = list(data_from.node_groups)
    data_to.armatures = list(data_from.armatures)
    data_to.actions = list(data_from.actions)
    data_to.curves = list(data_from.curves)
    data_to.texts = list(data_from.texts)
    data_to.grease_pencils = list(getattr(data_from, "grease_pencils", []))

# 创建对象并挂接数据块
name_to_object = {{}}
for o in spec["objects"]:
    data = None
    if o["data_name"] and o["type"] in {{"MESH","CURVE","ARMATURE","LIGHT","CAMERA","GPENCIL"}}:
        # 直接通过 data_name 获取
        coll = {{
            "MESH": bpy.data.meshes,
            "CURVE": bpy.data.curves,
            "ARMATURE": bpy.data.armatures,
            "LIGHT": bpy.data.lights,
            "CAMERA": bpy.data.cameras
        }}.get(o["type"], None)
        data = coll.get(o["data_name"]) if coll else None

    new = bpy.data.objects.new(o["name"], data)
    name_to_object[o["name"]] = new

    # 放进对应集合
    if o["collections"]:
        for cname in o["collections"]:
            name_to_collection[cname].objects.link(new)
    else:
        root.objects.link(new)

# 第二遍：设置父子关系、变换、材质、约束、动画
for o in spec["objects"]:
    ob = name_to_object[o["name"]]
    # 变换
    ob.matrix_world = mathutils.Matrix(o["matrix_world"])
    # 父子
    if o["parent"]:
        ob.parent = name_to_object.get(o["parent"])
        if o["parent_type"]:
            ob.parent_type = o["parent_type"]
    # 材质
    if o["materials"]:
        # 先清空再逐槽添加
        ob.data.materials.clear()
        for mname in o["materials"]:
            if mname:
                mat = bpy.data.materials.get(mname)
                if mat: ob.data.materials.append(mat)
            else:
                ob.data.materials.append(None)
    # 约束（仅恢复 target 引用和类型，复杂参数可按需扩展）
    for c in o["constraints"]:
        con = ob.constraints.new(c["type"])
        con.name = c["name"]
        if c.get("target"):
            tgt = name_to_object.get(c["target"])
            if tgt: con.target = tgt
    # 修改器（只创建类型/名字，具体参数可按需扩展）
    for m in o["modifiers"]:
        mod = ob.modifiers.new(m["name"], m["type"])
    # 动作
    if o["animation_action"]:
        ob.animation_data_create()
        act = bpy.data.actions.get(o["animation_action"])
        if act: ob.animation_data.action = act
    # 可见性
    ob.hide_viewport = o["hide_viewport"]
    ob.hide_render = o["hide_render"]

# 世界（若存在）
if spec["world"]:
    w = bpy.data.worlds.get(spec["world"])
    if w: bpy.context.scene.world = w

print("Scene rebuilt successfully.")
"""

SCRIPT_PATH.write_text(rebuild, encoding="utf-8")
print(f"[OK] Wrote library: {LIB_PATH}")
print(f"[OK] Wrote scene structure: {STRUCT_JSON}")
print(f"[OK] Wrote rebuild script: {SCRIPT_PATH}")
