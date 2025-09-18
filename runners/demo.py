import os
import shutil
import time
import subprocess

if __name__ == "__main__":
    time_stamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = f"output/demo/christmas1/{time_stamp}"
    init_code_path = "data/blendergym_hard/level4/christmas1/start.py"
    copy_init_code_path = f"{output_dir}/start.py"
    shutil.copy(init_code_path, copy_init_code_path)
    init_image_path = "data/blendergym_hard/level4/christmas1/renders/start"
    copy_init_image_path = f"{output_dir}/renders/start"
    shutil.copytree(init_image_path, copy_init_image_path)
    target_image_path = "data/blendergym_hard/level4/christmas1/renders/goal"
    copy_target_image_path = f"{output_dir}/renders/goal"
    shutil.copytree(target_image_path, copy_target_image_path)
    task_name = "level4-1"
    generator_script = "agents/generator_mcp.py"
    verifier_script = "agents/verifier_mcp.py"
    blender_server_path = "servers/generator/blender.py"
    blender_command = "utils/blender/infinigen/blender/blender"
    blender_file = "data/blendergym_hard/level4/christmas1/blender_file.blend"
    copy_blender_file = f"{output_dir}/blender_file.blend"
    shutil.copy(blender_file, copy_blender_file)
    blender_script = "data/blendergym_hard/level4/christmas1/pipeline_render_script.py"
    copy_blender_script = f"{output_dir}/pipeline_render_script.py"
    shutil.copy(blender_script, copy_blender_script)
    save_blender_file = True
    scene_server_path = "servers/verifier/scene.py"
    meshy_api_key = os.getenv("MESHY_API_KEY")
    va_api_key = os.getenv("VA_API_KEY")
    api_key = os.getenv("OPENAI_API_KEY")
    openai_base_url = os.getenv("OPENAI_BASE_URL")
    max_rounds = 100
    vision_model = "o4-mini"
    cmd = f"python main.py --mode blendergym-hard --vision-model {vision_model} --api-key {api_key} --max-rounds {max_rounds} --init-code-path {copy_init_code_path} --init-image-path {copy_init_image_path} --target-image-path {copy_target_image_path} --output-dir {output_dir} --task-name {task_name} --generator-script {generator_script} --verifier-script {verifier_script} --blender-server-path {blender_server_path} --blender-command {blender_command} --blender-file {copy_blender_file} --blender-script {copy_blender_script} --save-blender-file --scene-server-path {scene_server_path} --meshy_api_key {meshy_api_key} --va_api_key {va_api_key}"
    print(cmd)
    subprocess.run(cmd, shell=True)