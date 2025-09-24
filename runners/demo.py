import os
import shutil
import time
import subprocess
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test demo functionality")
    parser.add_argument("--target-image-path", default="data/blendergym_hard/level4/christmas1/renders/goal/visprompt1.png", type=str, help="Task name")
    args = parser.parse_args()
    
    time_stamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = f"output/demo/blendergym_hard/{time_stamp}"
    os.makedirs(output_dir, exist_ok=True)
    init_code_path = "data/blendergym_hard/level4/christmas1/start.py"
    copy_init_code_path = f"{output_dir}/start.py"
    copy_init_code_path_0 = f'{output_dir}/scripts/0.py'
    os.makedirs(f'{output_dir}/scripts', exist_ok=True)
    shutil.copy(init_code_path, copy_init_code_path)
    shutil.copy(init_code_path, copy_init_code_path_0)
    # not exist
    init_image_path = "none"
    # shutil.copytree(init_image_path, copy_init_image_path)
    target_image_path = args.target_image_path
    task_name = "level4-1"
    generator_script = "agents/generator.py"
    verifier_script = "agents/verifier.py"
    blender_server_path = "servers/generator/blender.py"
    blender_command = "utils/blender/infinigen/blender/blender"
    # Create a fresh empty .blend file as the initial editable state for each run
    copy_blender_file = f"{output_dir}/blender_file.blend"
    create_empty_blend_cmd = (
        f"{blender_command} --background --factory-startup "
        f"--python-expr \"import bpy; "
        f"bpy.ops.wm.read_factory_settings(use_empty=True); "
        f"bpy.ops.wm.save_mainfile(filepath=\\\"{copy_blender_file}\\\")\""
    )
    print(create_empty_blend_cmd)
    subprocess.run(create_empty_blend_cmd, shell=True, check=True)
    blender_script = "data/blendergym_hard/level4/christmas1/pipeline_render_script.py"
    copy_blender_script = f"{output_dir}/pipeline_render_script.py"
    shutil.copy(blender_script, copy_blender_script)
    save_blender_file = True
    scene_server_path = "servers/verifier/scene.py"
    meshy_api_key = os.getenv("MESHY_API_KEY")
    va_api_key = os.getenv("VA_API_KEY")
    api_key = os.getenv("OPENAI_API_KEY")
    openai_base_url = os.getenv("OPENAI_BASE_URL")
    max_rounds = 30
    vision_model = "gpt-5"
    cmd = f"python main.py --mode blendergym-hard --vision-model {vision_model} --api-key {api_key} --max-rounds {max_rounds} --init-code-path {copy_init_code_path} --init-image-path {init_image_path} --target-image-path {target_image_path} --output-dir {output_dir} --task-name {task_name} --generator-script {generator_script} --verifier-script {verifier_script} --blender-server-path {blender_server_path} --blender-command {blender_command} --blender-file {copy_blender_file} --blender-script {copy_blender_script} --save-blender-file --scene-server-path {scene_server_path} --meshy_api_key {meshy_api_key} --va_api_key {va_api_key}"
    print(cmd)
    subprocess.run(cmd, shell=True)