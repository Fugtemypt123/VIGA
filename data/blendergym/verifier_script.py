"""Blender script for verifier to execute code and save blend file."""
import bpy
import os
import sys

if __name__ == "__main__":
    code_fpath = sys.argv[6]  # Path to the code file
    if len(sys.argv) > 7:
        rendering_dir = sys.argv[7] # Path to save the rendering from camera1
    else:
        rendering_dir = None
    if len(sys.argv) > 8:
        save_blend = sys.argv[8] # Path to save the blend file
    else:
        save_blend = None

    with open(code_fpath, "r") as f:
        code = f.read()
    try:
        exec(code)
    except:
        raise ValueError

    # Save the blend file
    if save_blend:
        # Set the save version to 0
        bpy.context.preferences.filepaths.save_version = 0
        # Save the blend file
        bpy.ops.wm.save_as_mainfile(filepath=save_blend)