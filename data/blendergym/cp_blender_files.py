"""Utility script to copy blender files to task directories."""
import os
import subprocess

blender_files_dir = "blender_files"

for blender_file_name in os.listdir(blender_files_dir):
    blender_file_path = os.path.join(blender_files_dir, blender_file_name)
    task, start, end = blender_file_name.split('.')[0].split('_')

    start = int(start)
    end = int(end)


    for i in range(start, end +1):
        command = f"cp {blender_file_path} {task}{i}/blender_file.blend"
        print(command)

        subprocess.run(command, shell=True)