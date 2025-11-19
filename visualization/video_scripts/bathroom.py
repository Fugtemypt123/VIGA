import math

try:
    import bpy
except ImportError:
    raise RuntimeError(
        "This script must be run inside Blender, e.g.:\n"
        "  blender -b your_scene.blend -P generate_video.py"
    )

# ================== 可按需修改的参数 ==================
FRAMES = 240          # 总帧数（比如 240 帧，配合 30 fps ≈ 8 秒）
FPS = 30              # 帧率
RADIUS = 1.0          # 摄像机离中心的距离
HEIGHT = 1.7          # 摄像机高度
CENTER = (0.0, 0.0, 1.2)  # 环绕中心（可以调成你浴室的中心高度）
OUTPUT_PATH = "//bathroom.mp4"  # // 表示相对当前 .blend 文件所在目录
# =====================================================


def clear_old_cameras():
    """删除场景中已有摄像机（防止被旧的干扰）。"""
    for obj in list(bpy.data.objects):
        if obj.type == "CAMERA":
            print(f"[INFO] Removing old camera: {obj.name}")
            bpy.data.objects.remove(obj, do_unlink=True)


def create_camera_and_target(center):
    """创建一个摄像机和一个空物体作为跟踪目标."""
    scene = bpy.context.scene

    # 摄像机
    cam_data = bpy.data.cameras.new("TurntableCamera")
    camera = bpy.data.objects.new("TurntableCamera", cam_data)
    scene.collection.objects.link(camera)

    # 空物体：用于让摄像机始终看向它
    target = bpy.data.objects.new("TurntableTarget", None)
    target.location = center
    scene.collection.objects.link(target)

    # Track To 约束：摄像机始终看向 target
    constr = camera.constraints.new(type="TRACK_TO")
    constr.target = target
    constr.track_axis = "TRACK_NEGATIVE_Z"
    constr.up_axis = "UP_Y"

    # 设置为场景主摄像机
    scene.camera = camera

    print("[INFO] Camera and target created.")
    return camera


def animate_turntable(camera, center, radius, height, frames):
    """给摄像机做绕 center 一圈的动画（360°）。"""
    scene = bpy.context.scene

    for f in range(frames):
        angle = 2.0 * math.pi * (f / frames)

        x = center[0] + radius * math.cos(angle)
        y = center[1] + radius * math.sin(angle)
        z = center[2] + height

        scene.frame_set(f)
        camera.location = (x, y, z)
        camera.keyframe_insert(data_path="location", frame=f)

    print("[INFO] Turntable keyframes inserted.")


def setup_render(output_path, frames, fps):
    """设置渲染为 H264 MP4 输出."""
    scene = bpy.context.scene

    scene.frame_start = 0
    scene.frame_end = frames - 1
    scene.render.fps = fps

    # 输出路径；// 表示相对 .blend 所在目录
    scene.render.filepath = output_path

    # 引擎（要快可以换成 BLENDER_EEVEE）
    scene.render.engine = "CYCLES"

    scene.render.image_settings.file_format = "FFMPEG"
    ffmpeg = scene.render.ffmpeg
    ffmpeg.format = "MPEG4"
    ffmpeg.codec = "H264"
    ffmpeg.constant_rate_factor = "MEDIUM"
    ffmpeg.ffmpeg_preset = "GOOD"

    print(f"[INFO] Render setup done. Output: {output_path}")


def main():
    print("[INFO] Using current opened blend file:", bpy.data.filepath)

    clear_old_cameras()
    camera = create_camera_and_target(CENTER)
    animate_turntable(camera, CENTER, RADIUS, HEIGHT, FRAMES)
    setup_render(OUTPUT_PATH, FRAMES, FPS)

    print("[INFO] Start rendering animation...")
    bpy.ops.render.render(animation=True)
    print("[INFO] Rendering finished.")


if __name__ == "__main__":
    main()
