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
        
    print("[INFO] Cycles GPU device set to: ", bpy.context.scene.cycles.device)

    # ---- 基础渲染参数 ----
    scene = bpy.context.scene
    scene.render.resolution_x = 512
    scene.render.resolution_y = 512
    scene.cycles.samples = 512
    scene.render.image_settings.color_mode = 'RGB'
    
    # ---- 视频编码设置 ----
    scene.render.image_settings.file_format = 'FFMPEG'
    scene.render.ffmpeg.format = 'MPEG4'
    scene.render.ffmpeg.codec = 'H264'
    scene.render.ffmpeg.constant_rate_factor = 'MEDIUM'
    scene.render.ffmpeg.video_bitrate = 8000
    scene.render.ffmpeg.minrate = 0
    scene.render.ffmpeg.maxrate = 0
    scene.render.ffmpeg.buffersize = 1792
    
    # ---- 输出目录准备 ----            
    os.makedirs(rendering_dir, exist_ok=True)

    # ---- 读取动画范围 ----
    # 若外部代码未设置，Blender 默认 1..250
    frame_start = 1 # if scene.frame_start is None else scene.frame_start
    frame_end   = 48 # if scene.frame_end is None else scene.frame_end
    
    scene.frame_start = frame_start
    scene.frame_end = frame_end
    
    # 总帧数
    total_frames = max(1, frame_end - frame_start + 1)
    
    # 获取 FPS（如果未设置则使用默认值）
    fps = scene.render.fps if scene.render.fps > 0 else 16
    scene.render.ffmpeg.gopsize = int(fps)

    print(f"[INFO] Frame range: {frame_start}..{frame_end} (total {total_frames} frames)")
    print(f"[INFO] FPS: {fps}")

    # ---- 渲染视频（对每台相机分别渲染完整动画）----
    cameras = [obj for obj in bpy.data.objects if obj.type == 'CAMERA']
    for cam in cameras:
        scene.camera = cam
        scene.camera.data.lens = 18
        
        # 视频文件名：CameraName.mp4
        video_path = os.path.join(rendering_dir, f"{cam.name}.mp4")
        scene.render.filepath = video_path
        
        print(f"[RENDER] {cam.name} -> rendering animation ({total_frames} frames) -> {video_path}")
        try:
            bpy.ops.render.render(animation=True)
            print(f"[RENDER] {cam.name} -> completed: {video_path}")
        except Exception as e:
            print(f"[ERROR] Failed to render {cam.name}: {e}")

    # ---- 可选：保存 .blend ----
    if save_blend:
        bpy.context.preferences.filepaths.save_version = 0
        bpy.ops.wm.save_as_mainfile(filepath=save_blend)
        print(f"[INFO] Saved blend to: {save_blend}")

    print("[DONE] Rendered video files for all cameras.")
