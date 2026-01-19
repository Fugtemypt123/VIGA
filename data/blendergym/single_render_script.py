"""Blender script to render a single camera view (Camera1) after executing scene code."""
import bpy
import os
import sys

# Disable audio sync/playback at scene level
for scene in bpy.data.scenes:
    scene.use_audio = False
    scene.sync_mode = 'NONE'

# Disable audio device in preferences
try:
    bpy.context.preferences.system.audio_device = 'NONE'
except Exception:
    pass

if __name__ == "__main__":

    code_fpath = sys.argv[6]  # Path to the code file
    rendering_dir = sys.argv[7] # Path to save the rendering from camera1

    # Enable GPU rendering
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'  # or 'OPTIX' if your GPU supports it
    bpy.context.preferences.addons['cycles'].preferences.get_devices()

    # Check and select the GPUs
    for device in bpy.context.preferences.addons['cycles'].preferences.devices:
        if device.type == 'GPU' and not device.use:
            device.use = True

    # Set the rendering device to GPU
    bpy.context.scene.cycles.device = 'GPU'

    # Setting up rendering resolution
    bpy.context.scene.render.resolution_x = 512
    bpy.context.scene.render.resolution_y = 512

    # Set max samples to 1024
    bpy.context.scene.cycles.samples = 512

    # Set color mode to RGB
    bpy.context.scene.render.image_settings.color_mode = 'RGB'

    # Read and execute the code from the specified file
    with open(code_fpath, "r") as f:
        code = f.read()
    try:
        exec(code)
    except:
        raise ValueError

    # Render from camera1
    if 'Camera1' in bpy.data.objects:
        bpy.context.scene.camera = bpy.data.objects['Camera1']
        bpy.context.scene.render.image_settings.file_format = 'PNG'
        bpy.context.scene.render.filepath = os.path.join(rendering_dir, 'render1.png')
        bpy.ops.render.render(write_still=True)





