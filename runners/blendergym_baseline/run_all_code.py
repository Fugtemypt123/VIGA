import os
import subprocess

geo_path = "data/blendergym/geometry"

for i in range(1, 51):
    code_path = geo_path + f"{i}/baseline/Qwen3_VL_8B_Instruct.py"
    render_path = geo_path + f"{i}/baseline/Qwen3_VL_8B_Instruct"
    os.makedirs(render_path, exist_ok=True)
    cmd = [
        "utils/infinigen/blender/blender",
        "--background", geo_path + f"{i}/blender_file.blend",
        "--python", "data/blendergym/pipeline_render_script.py",
        "--", code_path, render_path
    ]
    subprocess.run(cmd, check=True)
    print(f"Running blender command: {cmd}")