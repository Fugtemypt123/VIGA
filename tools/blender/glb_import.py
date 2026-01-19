"""GLB to Blender Scene Importer.

This script imports multiple GLB files into a Blender scene based on a
transforms JSON configuration. Run with:
    blender -b -P import_glbs_to_blend.py -- transforms.json output.blend
"""

import json
import os
import sys
from typing import List, Optional, Tuple

import bpy
from mathutils import Euler, Vector


def parse_args() -> Tuple[str, str]:
    """Parse command line arguments.

    Returns:
        Tuple of (transforms_json_path, blend_output_path).

    Raises:
        SystemExit: If required arguments are not provided.
    """
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

def clear_scene() -> None:
    """Delete all existing objects in the scene."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)

def setup_camera() -> None:
    """Set up camera at world origin with correct orientation and FOV.

    Configures the camera to face the -Y direction with Z as up.
    Uses a 32mm lens for wider field of view.
    """
    if "Camera" not in bpy.data.objects:
        bpy.ops.object.camera_add()
        camera = bpy.context.active_object
        camera.name = "Camera"
    else:
        camera = bpy.data.objects["Camera"]

    camera.location = Vector((0.0, 0.0, 0.0))
    # Point camera toward -Y direction with Z as up
    # Camera's local -Z is view direction, local +Y is "up"
    # Use to_track_quat('-Z', 'Y') to align -Z to (0, -1, 0) and Y to (0, 0, 1)
    direction = Vector((0.0, -1.0, 0.0))
    camera.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler('XYZ')

    # Use 32mm wide-angle lens for larger viewing area
    camera.data.lens = 32

    print(f"[INFO] Camera setup complete: location={camera.location}, lens={camera.data.lens}mm")

def setup_lighting() -> None:
    """Set up environment lighting for the scene.

    Creates a white background and adds a sun light for illumination.
    """
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

    print("[INFO] Lighting setup complete")

def setup_render() -> None:
    """Set up render settings matching generator_script.

    Configures Cycles engine with GPU if available, 512x512 resolution,
    and 512 samples.
    """
    scene = bpy.context.scene

    # Configure render engine and device
    scene.render.engine = 'CYCLES'
    try:
        prefs = bpy.context.preferences.addons['cycles'].preferences
        prefs.compute_device_type = 'CUDA'
        prefs.get_devices()
        for device in prefs.devices:
            if device.type == 'GPU' and not device.use:
                device.use = True
        scene.cycles.device = 'GPU'
    except Exception as e:
        print(f"[WARN] Failed to configure GPU rendering: {e}")

    # Set square resolution (512x512)
    scene.render.resolution_x = 512
    scene.render.resolution_y = 512

    # Configure samples and color mode
    scene.cycles.samples = 512
    scene.render.image_settings.color_mode = 'RGB'

def import_glb(glb_path: str, name_prefix: str = "") -> Optional[bpy.types.Object]:
    """Import a GLB file into the scene.

    Args:
        glb_path: Path to the GLB file to import.
        name_prefix: Optional name to assign to the root object.

    Returns:
        The root object of the imported hierarchy, or None if import failed.
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

    # Find the root object (one without parent in imported set)
    root = None
    for obj in imported_objects:
        if obj.parent not in imported_objects:
            root = obj
            break
    if not root:
        root = imported_objects[0]

    if name_prefix:
        root.name = name_prefix

    # Set origin for all imported MESH objects
    mesh_count = 0
    for obj in imported_objects:
        if obj.type == 'MESH':
            # Must set as active to use ops
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='MEDIAN')
            mesh_count += 1
            print(f"[INFO] Set origin for mesh: {obj.name}, location: {obj.location}")

    print(f"[INFO] Imported {len(imported_objects)} objects from {glb_path} (processed {mesh_count} meshes)")
    return root

def save_blend(path: str) -> None:
    """Save the scene as a Blender file.

    Args:
        path: Output path for the .blend file.
    """
    print(f"[INFO] Saving Blender file to: {path}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=path)
    print(f"[INFO] Saved: {path}")


def main() -> None:
    """Main entry point for the GLB importer script."""
    transforms_json_path, blend_path = parse_args()
    print(f"[INFO] Loading transforms from: {transforms_json_path}")
    print(f"[INFO] Output: {blend_path}")

    with open(transforms_json_path, 'r') as f:
        objects_data = json.load(f)

    print(f"[INFO] Importing {len(objects_data)} GLB files")

    clear_scene()
    setup_camera()
    setup_lighting()
    setup_render()

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
