import os

for task_name in os.listdir("."):
    if 'material' in task_name:
        for file in os.listdir(task_name):
            if file.endswith(".py"):
                with open(os.path.join(task_name, file), "r") as f:
                    content = f.read()
                head_count = 0
                for line in content.split("\n"):
                    if line.startswith("import") or line.startswith("from"):
                        head_count += 1
                    else:
                        break
                new_head_content = """import bpy
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core import surface
from infinigen.core.nodes import node_utils"""
                tail_content = content.split("\n")[head_count:]
                new_content = new_head_content + "\n" + "\n".join(tail_content)
                with open(os.path.join(task_name, file), "w") as f:
                    f.write(new_content)