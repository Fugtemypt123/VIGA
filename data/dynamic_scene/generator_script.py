"""Blender script for dynamic scene generation with keyframe rendering."""
import bpy
import os
import sys

if __name__ == "__main__":

    # ---- Command line arguments ----
    code_fpath = sys.argv[6]  # Path to scene generation/editing code
    if len(sys.argv) > 7:
        rendering_dir = sys.argv[7]  # Rendering output directory
    else:
        rendering_dir = None
    if len(sys.argv) > 8:
        save_blend = sys.argv[8]  # Optional: path to save .blend file
    else:
        save_blend = None

    # ---- Execute external code to build scene ----
    with open(code_fpath, "r", encoding="utf-8") as f:
        code = f.read()
    try:
        exec(code)
    except Exception as e:
        raise ValueError(f"Error executing scene code: {e}")

    if not rendering_dir:
        print("[INFO] No rendering directory provided, skipping rendering.")
        exit(0)

    bpy.context.scene.render.engine = 'CYCLES'
    try:
        prefs = bpy.context.preferences.addons['cycles'].preferences
        # Optional: 'CUDA' or 'OPTIX' depending on GPU support
        prefs.compute_device_type = 'CUDA'
        prefs.get_devices()
        # Select all GPU devices
        for device in prefs.devices:
            if device.type == 'GPU':
                device.use = True
        bpy.context.scene.cycles.device = 'GPU'
    except Exception:
        # Fall back to CPU if no GPU or setup fails
        bpy.context.scene.cycles.device = 'CPU'

    # ---- Basic render parameters ----
    scene = bpy.context.scene
    scene.render.resolution_x = 512
    scene.render.resolution_y = 512
    scene.cycles.samples = 512
    scene.render.image_settings.color_mode = 'RGB'
    scene.render.image_settings.file_format = 'PNG'
    scene.render.use_file_extension = True

    # ---- Prepare output directory ----
    os.makedirs(rendering_dir, exist_ok=True)

    # ---- Read animation range and compute three keyframes ----
    # If not set by external code, Blender defaults to 1..250
    frame_start = scene.frame_start if scene.frame_start else 1
    frame_end = scene.frame_end if scene.frame_end else 250

    # Total frames (for logging or extension)
    total_frames = max(1, frame_end - frame_start + 1)

    # Midpoint frame (rounded to integer)
    frame_mid = (frame_start + frame_end) // 2

    # Three-frame set (deduplicated and sorted, handles start=end case)
    frames_to_render = sorted(set([frame_start, frame_mid, frame_end]))

    print(f"[INFO] Frame range: {frame_start}..{frame_end} (total {total_frames})")
    print(f"[INFO] Will render frames: {frames_to_render}")

    # ---- Render (render three frames for each camera) ----
    cameras = [obj for obj in bpy.data.objects if obj.type == 'CAMERA']
    for cam in cameras:
        scene.camera = cam

        for f in frames_to_render:
            scene.frame_set(f)
            # Force update (sometimes more stable for constraints/drivers/physics)
            bpy.context.view_layer.update()

            # Filename: CameraName_fXXXX.png
            scene.render.filepath = os.path.join(rendering_dir, f"{cam.name}_f{f:04d}.png")
            print(f"[RENDER] {cam.name} @ frame {f} -> {scene.render.filepath}")
            bpy.ops.render.render(write_still=True)

    # ---- Optional: save .blend ----
    if save_blend:
        bpy.context.preferences.filepaths.save_version = 0
        bpy.ops.wm.save_as_mainfile(filepath=save_blend)
        print(f"[INFO] Saved blend to: {save_blend}")

    print("[DONE] Rendered 3 key frames per camera.")
