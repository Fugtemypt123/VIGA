import bpy
import os
import sys

if __name__ == "__main__":

    # ---- 命令行参数 ----
    code_fpath   = sys.argv[6]  # 生成/编辑场景的代码文件路径
    if len(sys.argv) > 7:
        rendering_dir = sys.argv[7]  # 渲染输出目录
    else:
        rendering_dir = None
    if len(sys.argv) > 8:
        save_blend = sys.argv[8]  # 可选：保存 .blend 的路径
    else:
        save_blend = None

    # ---- 执行外部代码，构建场景 ----
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
        # 可选：'CUDA' 或 'OPTIX'（视你的显卡支持）
        prefs.compute_device_type = 'CUDA'
        prefs.get_devices()
        # 选择所有 GPU 设备
        for device in prefs.devices:
            if device.type == 'GPU':
                device.use = True
        bpy.context.scene.cycles.device = 'GPU'
    except Exception:
        # 没有 GPU 或设置失败时，回退到 CPU
        bpy.context.scene.cycles.device = 'CPU'

    # ---- 基础渲染参数 ----
    scene = bpy.context.scene
    scene.render.resolution_x = 512
    scene.render.resolution_y = 512
    scene.cycles.samples = 512
    scene.render.image_settings.color_mode = 'RGB'
    scene.render.image_settings.file_format = 'PNG'
    scene.render.use_file_extension = True

    # ---- 输出目录准备 ----            
    os.makedirs(rendering_dir, exist_ok=True)

    # ---- 读取动画范围并计算三帧 ----
    # 若外部代码未设置，Blender 默认 1..250
    frame_start = scene.frame_start if scene.frame_start else 1
    frame_end   = scene.frame_end   if scene.frame_end   else 250

    # 总帧数（用于日志或扩展）
    total_frames = max(1, frame_end - frame_start + 1)

    # 中点帧（取整到整数帧）
    frame_mid = (frame_start + frame_end) // 2

    # 三帧集合（去重并排序，防止 start=end 的情况）
    frames_to_render = sorted(set([frame_start, frame_mid, frame_end]))

    print(f"[INFO] Frame range: {frame_start}..{frame_end} (total {total_frames})")
    print(f"[INFO] Will render frames: {frames_to_render}")

    # ---- 渲染（对每台相机分别渲染三帧）----
    cameras = [obj for obj in bpy.data.objects if obj.type == 'CAMERA']
    for cam in cameras:
        scene.camera = cam

        for f in frames_to_render:
            scene.frame_set(f)
            # 强制更新（有时对约束/驱动/物理更稳）
            bpy.context.view_layer.update()

            # 文件名：CameraName_fXXXX.png
            scene.render.filepath = os.path.join(rendering_dir, f"{cam.name}_f{f:04d}.png")
            print(f"[RENDER] {cam.name} @ frame {f} -> {scene.render.filepath}")
            bpy.ops.render.render(write_still=True)

    # ---- 可选：保存 .blend ----
    if save_blend:
        bpy.context.preferences.filepaths.save_version = 0
        bpy.ops.wm.save_as_mainfile(filepath=save_blend)
        print(f"[INFO] Saved blend to: {save_blend}")

    print("[DONE] Rendered 3 key frames per camera.")
