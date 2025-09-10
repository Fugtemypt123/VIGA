import os
import sys
import argparse
from typing import List, Dict, Any, Optional, Tuple


def _find_obj_file(asset_dir: str) -> Optional[str]:
    """Return the first .obj file path found in the directory (non-recursive)."""
    for name in sorted(os.listdir(asset_dir)):
        if name.lower().endswith(".obj"):
            return os.path.join(asset_dir, name)
    return None


def import_obj_asset_to_blend(
    asset_dir: str,
    blend_path: str,
    location: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    scale: float = 1.0,
) -> Dict[str, Any]:
    """
    Import an OBJ asset (folder containing .obj and textures) into a .blend file.
    If the .blend does not exist, create it. Name the imported asset as the folder name.

    Note: This function must be run inside Blender's Python (bpy available).
    """
    try:
        import bpy  # type: ignore
    except Exception:
        raise RuntimeError("This function must be executed inside Blender (bpy not available)")

    if not os.path.isdir(asset_dir):
        raise FileNotFoundError(f"Asset directory not found: {asset_dir}")

    obj_path = _find_obj_file(asset_dir)
    if not obj_path:
        raise FileNotFoundError(f"No .obj file found in: {asset_dir}")

    asset_name = os.path.basename(os.path.normpath(asset_dir))

    # Open or create blend file
    if os.path.exists(blend_path):
        bpy.ops.wm.open_mainfile(filepath=blend_path)
    else:
        # Start from a clean file and save to target path
        bpy.ops.wm.read_homefile(use_empty=True)
        os.makedirs(os.path.dirname(blend_path) or ".", exist_ok=True)
        bpy.ops.wm.save_mainfile(filepath=blend_path)

    # Track pre-existing selection/objects
    for obj in bpy.context.selected_objects:
        obj.select_set(False)

    # Import OBJ (textures typically resolved via .mtl next to .obj)
    before_objects = set(bpy.data.objects.keys())
    try:
        bpy.ops.import_scene.obj(filepath=obj_path)
    except Exception as e:
        raise RuntimeError(f"Failed to import OBJ: {e}")

    # Determine imported objects
    after_objects = set(bpy.data.objects.keys())
    imported_names = sorted(list(after_objects - before_objects))
    if not imported_names:
        raise RuntimeError("OBJ import did not create any objects")

    imported_objects: List[Any] = [bpy.data.objects[name] for name in imported_names]

    # Create or get a collection for the asset
    collection = bpy.data.collections.get(asset_name)
    if collection is None:
        collection = bpy.data.collections.new(asset_name)
        bpy.context.scene.collection.children.link(collection)

    # Link/move imported objects to the asset collection and set transforms
    for obj in imported_objects:
        # Move to collection
        for coll in list(obj.users_collection):
            coll.objects.unlink(obj)
        collection.objects.link(obj)
        # Set transform
        obj.location = location
        obj.scale = (scale, scale, scale)

    # Rename the primary object (heuristic: first mesh or first imported)
    primary_obj = next((o for o in imported_objects if o.type == 'MESH'), imported_objects[0])
    primary_obj.name = asset_name

    # Save the blend
    bpy.ops.wm.save_mainfile(filepath=blend_path)

    # Remove Blender backup file if any
    backup = blend_path + "1"
    try:
        if os.path.exists(backup):
            os.remove(backup)
    except Exception:
        pass

    return {
        "status": "success",
        "message": f"Imported '{asset_name}' into blend",
        "blend_path": blend_path,
        "asset_name": asset_name,
        "imported_objects": [o.name for o in imported_objects],
        "collection": collection.name,
        "primary_object": primary_obj.name,
    }


def main():
    parser = argparse.ArgumentParser(description="Import an OBJ asset folder into a .blend file")
    parser.add_argument("--asset-dir", required=True, help="Path to the asset directory containing .obj and textures")
    parser.add_argument("--blend", required=True, help="Path to the .blend file (created if missing)")
    parser.add_argument("--location", default="0,0,0", help="Import location as x,y,z")
    parser.add_argument("--scale", type=float, default=1.0, help="Uniform scale factor")
    args = parser.parse_args()

    try:
        loc = tuple(float(x.strip()) for x in args.location.split(","))
        if len(loc) != 3:
            raise ValueError
    except Exception:
        print("Invalid --location format. Use 'x,y,z'", file=sys.stderr)
        sys.exit(2)

    try:
        result = import_obj_asset_to_blend(
            asset_dir=args.asset_dir,
            blend_path=args.blend,
            location=(loc[0], loc[1], loc[2]),
            scale=args.scale,
        )
        print(result)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()


