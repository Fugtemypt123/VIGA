"""GLB Vertex Color Fixer for Blender.

This script fixes GLB files by properly connecting vertex colors to material
base colors. Run with:
    blender -b -P fix_glb.py -- input.glb output.glb
"""

import os
import sys
from typing import List, Optional, Tuple

import bpy


def parse_args() -> Tuple[str, str]:
    """Parse command line arguments for input and output paths.

    Returns:
        Tuple of (input_path, output_path).

    Raises:
        SystemExit: If required arguments are not provided.
    """
    argv = sys.argv
    if "--" not in argv:
        print("[ERROR] Usage:")
        print("  blender -b -P fix_glb.py -- input.glb output.glb")
        sys.exit(1)
    idx = argv.index("--")
    if len(argv) < idx + 3:
        print("[ERROR] Need input and output paths.")
        sys.exit(1)
    in_path = os.path.abspath(argv[idx + 1])
    out_path = os.path.abspath(argv[idx + 2])
    return in_path, out_path

def clear_scene() -> None:
    """Clear all objects from the Blender scene."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    # Remove all remaining objects to ensure clean state
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)

def import_glb(path: str) -> None:
    """Import a GLB file into the current scene.

    Args:
        path: Path to the GLB file to import.
    """
    print(f"[INFO] Importing GLB: {path}")
    bpy.ops.import_scene.gltf(filepath=path)

def get_vertex_color_name(mesh: bpy.types.Mesh) -> Optional[str]:
    """Get the name of the vertex color layer from a mesh.

    Supports both Blender 3.x/4.x color_attributes and older vertex_colors.

    Args:
        mesh: The Blender mesh object to check.

    Returns:
        The name of the vertex color layer, or None if not found.
    """
    # Blender 3.x / 4.x: color_attributes
    if hasattr(mesh, "color_attributes") and mesh.color_attributes:
        # Find first color-type attribute
        for attr in mesh.color_attributes:
            if attr.domain in {"CORNER", "POINT"} and attr.data_type in {"BYTE_COLOR", "FLOAT_COLOR"}:
                return attr.name
        # Fallback to first attribute
        return mesh.color_attributes[0].name

    # Compatibility with older vertex_colors API
    if hasattr(mesh, "vertex_colors") and mesh.vertex_colors:
        return mesh.vertex_colors[0].name

    return None

def material_has_basecolor_texture(mat: Optional[bpy.types.Material]) -> bool:
    """Check if material already has a texture connected to Base Color.

    Args:
        mat: The Blender material to check.

    Returns:
        True if material has an image texture on Base Color, False otherwise.
    """
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
            # Check if from_node is an Image Texture
            if link.from_node.type == "TEX_IMAGE":
                return True
    return False

def apply_vertex_color_to_material(
    mat: Optional[bpy.types.Material],
    mesh: bpy.types.Mesh,
    vcol_name: str
) -> None:
    """Apply vertex color to a material's Base Color input.

    Creates a new node setup with Attribute -> Principled BSDF -> Output.

    Args:
        mat: The Blender material to modify.
        mesh: The mesh containing vertex colors.
        vcol_name: Name of the vertex color layer.
    """
    if mat is None:
        return
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # Clear existing nodes
    for n in list(nodes):
        nodes.remove(n)

    # Attribute node (read vertex colors)
    attr = nodes.new("ShaderNodeAttribute")
    attr.attribute_name = vcol_name
    attr.location = (-300, 0)

    # Principled BSDF
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (0, 0)

    # Output node
    out = nodes.new("ShaderNodeOutputMaterial")
    out.location = (200, 0)

    # Connect: Attribute Color -> Base Color -> Output
    links.new(attr.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    print(f"[FIX] Material {mat.name}: vertex color '{vcol_name}' -> Base Color")

def process_scene() -> None:
    """Process all mesh objects in the scene and apply vertex colors."""
    mesh_objects = [obj for obj in bpy.data.objects if obj.type == "MESH"]
    print(f"[INFO] Found {len(mesh_objects)} mesh objects")

    for obj in mesh_objects:
        mesh = obj.data
        vcol_name = get_vertex_color_name(mesh)
        if not vcol_name:
            print(f"[WARN] Object {obj.name} has NO vertex colors; skip.")
            continue

        print(f"[INFO] Object {obj.name} uses vertex color layer '{vcol_name}'")

        # Create material if none exists
        if not obj.data.materials:
            mat = bpy.data.materials.new(name=f"{obj.name}_VertexColorMat")
            obj.data.materials.append(mat)
            mats = [mat]
        else:
            mats = [slot.material for slot in obj.material_slots if slot.material]

        for mat in mats:
            # Keep existing textures on Base Color to avoid breaking materials
            if material_has_basecolor_texture(mat):
                print(f"[INFO] Material {mat.name} already has texture on Base Color; keep it.")
                continue
            # Otherwise apply vertex color
            apply_vertex_color_to_material(mat, mesh, vcol_name)

def export_glb(path: str) -> None:
    """Export the scene to a GLB file.

    Args:
        path: Output path for the GLB file.
    """
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


def main() -> None:
    """Main entry point for the GLB fixer script."""
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
