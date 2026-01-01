import math

try:
    import bpy
    from mathutils import Vector
except ImportError:
    raise RuntimeError(
        "This script must be run inside Blender, e.g.:\n"
        "  blender -b your_scene.blend -P generate_video.py"
    )

# ================== 可按需修改的参数 ==================
FRAMES = 300         # 总帧数
FPS = 30         # 帧率
START_POS = (-4.8, 0, 1.35)     # 摄像机起点
END_POS   = (3.0, 0, 1.35)   # 摄像机终点
OUTPUT_PATH = "//kitchen.mp4"  # 相对当前 .blend 文件所在目录
# =====================================================


def enable_gpu_for_cycles():
    """
    将 Cycles 渲染设备切换到 GPU。
    首选 CUDA，其次 OPTIX；如果都不可用则退回 CPU。
    """
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"

    prefs_all = bpy.context.preferences

    # 确保 Cycles 插件已启用
    if "cycles" not in prefs_all.addons:
        try:
            bpy.ops.preferences.addon_enable(module="cycles")
        except Exception as e:
            print(f"[WARN] Cannot enable Cycles addon automatically: {e}")
            return

    try:
        cycles_prefs = prefs_all.addons["cycles"].preferences
    except KeyError:
        print("[WARN] Cycles addon not found in preferences; fallback to CPU.")
        return

    backend_chosen = None
    # ★ 首选 CUDA，再尝试 OPTIX
    for backend in ("OPTIX", "CUDA"):
        try:
            cycles_prefs.compute_device_type = backend
            backend_chosen = backend
            break
        except TypeError:
            # 该 Blender 版本/编译不支持这个 backend
            continue

    if backend_chosen is None:
        print("[WARN] No supported GPU backend (CUDA/OPTIX); fallback to CPU.")
        return

    # 刷新设备列表（不同版本函数签名可能略有差异）
    try:
        cycles_prefs.refresh_devices()
    except TypeError:
        cycles_prefs.refresh_devices(bpy.context)

    # 只启用 GPU 设备
    for dev in cycles_prefs.devices:
        dev.use = (dev.type == "GPU" or dev.type == backend_chosen)
        print(f"[GPU] {dev.type}: {dev.name}, use={dev.use}")

    # 场景层面指定使用 GPU
    scene.cycles.device = "GPU"
    print(f"[INFO] Cycles is now using GPU with backend = {backend_chosen}.")

def clear_old_cameras():
    """删除场景中已有摄像机（防止被旧的干扰）。"""
    for obj in list(bpy.data.objects):
        if obj.type == "CAMERA":
            print(f"[INFO] Removing old camera: {obj.name}")
            bpy.data.objects.remove(obj, do_unlink=True)


def create_camera():
    """创建一个摄像机对象，不再使用空物体和 TrackTo 约束。"""
    scene = bpy.context.scene

    cam_data = bpy.data.cameras.new("TravelCamera")
    camera = bpy.data.objects.new("TravelCamera", cam_data)
    scene.collection.objects.link(camera)

    # 设置为场景主摄像机
    scene.camera = camera

    print("[INFO] Camera created.")
    return camera


def animate_linear_motion(camera, start_pos, end_pos, frames):
    """
    让摄像机从 start_pos 直线运动到 end_pos，
    且视角始终沿运动方向。
    """
    scene = bpy.context.scene

    start = Vector(start_pos)
    end = Vector(end_pos)

    # 运动方向向量（终点 - 起点）
    direction = (end - start).normalized()

    # 摄像机在 Blender 中默认是沿 -Z 轴看向前方，
    # 因此让 -Z 轴对齐到运动方向 direction。
    base_quat = direction.to_track_quat('-Z', 'Y')

    for f in range(frames):
        # t 从 0 到 1，均匀插值
        if frames > 1:
            t = f / (frames - 1)
        else:
            t = 0.0

        # 线性插值位置
        pos = start.lerp(end, t)

        scene.frame_set(f)
        camera.location = pos
        camera.rotation_euler = base_quat.to_euler()

        camera.keyframe_insert(data_path="location", frame=f)
        camera.keyframe_insert(data_path="rotation_euler", frame=f)

    print("[INFO] Linear motion keyframes (location + rotation) inserted.")


def setup_render(output_path, frames, fps):
    """设置渲染为 H264 MP4 输出."""
    scene = bpy.context.scene

    scene.frame_start = 0
    scene.frame_end = frames - 1
    scene.render.fps = fps

    # 输出路径；// 表示相对 .blend 所在目录
    scene.render.filepath = output_path

    # 引擎（这里使用 Cycles，设备由 enable_gpu_for_cycles 控制）
    scene.render.engine = "CYCLES"
    
    scene.render.resolution_x = 1080
    scene.render.resolution_y = 1080
    scene.render.resolution_percentage = 100
    scene.render.use_motion_blur = False

    scene.render.image_settings.file_format = "FFMPEG"
    ffmpeg = scene.render.ffmpeg
    ffmpeg.format = "MPEG4"
    ffmpeg.codec = "H264"
    ffmpeg.constant_rate_factor = "HIGH"
    ffmpeg.ffmpeg_preset = "GOOD"

    print(f"[INFO] Render setup done. Output: {output_path}")


def main():
    print("[INFO] Using current opened blend file:", bpy.data.filepath)

    # ★ 先启用 GPU 渲染（首选 CUDA）
    enable_gpu_for_cycles()

    clear_old_cameras()
    camera = create_camera()
    animate_linear_motion(camera, START_POS, END_POS, FRAMES)
    setup_render(OUTPUT_PATH, FRAMES, FPS)

    print("[INFO] Start rendering animation...")
    bpy.ops.render.render(animation=True)
    print("[INFO] Rendering finished.")


if __name__ == "__main__":
    main()
