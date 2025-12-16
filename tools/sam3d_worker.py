import os, sys, argparse, json
from PIL import Image
import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(ROOT, "utils", "sam3d", "notebook"))
sys.path.append(os.path.join(ROOT, "utils", "sam3d"))

if "CONDA_PREFIX" not in os.environ:
    python_bin = sys.executable
    conda_env = os.path.dirname(os.path.dirname(python_bin))
    os.environ["CONDA_PREFIX"] = conda_env

from inference import Inference, load_image, load_mask, make_scene


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--masks", required=True, nargs='+', help="One or more mask .npy file paths")
    p.add_argument("--config", required=True)
    p.add_argument("--blend", required=True, help="Output .blend file path")
    p.add_argument("--object-names", nargs='+', help="Optional object names corresponding to masks")
    args = p.parse_args()

    inference = Inference(args.config, compile=False)
    image = load_image(args.image)
    
    # Process all masks
    outputs = []
    for idx, mask_path in enumerate(args.masks):
        mask = np.load(mask_path)
        mask = mask > 0
        output = inference(image, mask, seed=42)
        outputs.append(output)
    
    # Combine all outputs into a scene using make_scene (for Gaussian Splatting)
    scene_gs = make_scene(*outputs)
    
    # Export each object's GLB and then import into Blender
    import tempfile
    import subprocess
    
    temp_dir = tempfile.mkdtemp()
    glb_paths = []
    object_names = args.object_names if args.object_names else [f"object_{i}" for i in range(len(outputs))]
    
    # Export GLB for each output
    for idx, output in enumerate(outputs):
        glb = output.get("glb")
        if glb is not None and hasattr(glb, "export"):
            glb_path = os.path.join(temp_dir, f"{object_names[idx]}.glb")
            os.makedirs(os.path.dirname(glb_path), exist_ok=True)
            glb.export(glb_path)
            glb_paths.append(glb_path)
    
    if not glb_paths:
        print(json.dumps({"status": "error", "message": "No GLB files generated"}), file=sys.stderr)
        sys.exit(1)
    
    # Create Blender import script
    blend_path_escaped = args.blend.replace("\\", "\\\\").replace('"', '\\"')
    blender_script = f"""
import bpy
import os

# Clear scene
bpy.ops.wm.read_factory_settings(use_empty=True)
for obj in list(bpy.data.objects):
    bpy.data.objects.remove(obj, do_unlink=True)

# Import all GLB files
glb_paths = {json.dumps(glb_paths)}
object_names = {json.dumps(object_names)}

for idx, glb_path in enumerate(glb_paths):
    if os.path.exists(glb_path):
        bpy.ops.import_scene.gltf(filepath=glb_path)
        # Rename the imported object
        if bpy.context.selected_objects:
            obj = bpy.context.selected_objects[0]
            if idx < len(object_names):
                obj.name = object_names[idx]

# Save blend file
os.makedirs(os.path.dirname(r"{blend_path_escaped}"), exist_ok=True)
bpy.ops.wm.save_as_mainfile(filepath=r"{blend_path_escaped}")
"""
    
    # Write script to temp file
    script_path = os.path.join(temp_dir, "import_to_blend.py")
    with open(script_path, 'w') as f:
        f.write(blender_script)
    
    # Get Blender command from environment or use default
    blender_cmd = os.environ.get("BLENDER_CMD", "blender")
    result = subprocess.run(
        [blender_cmd, "-b", "-P", script_path],
        cwd=ROOT,
        capture_output=True,
        text=True
    )
    
    # Clean up temp files
    import shutil
    shutil.rmtree(temp_dir)
    
    if result.returncode != 0:
        print(json.dumps({
            "status": "error",
            "message": f"Blender import failed: {result.stderr}"
        }), file=sys.stderr)
        sys.exit(1)
    
    # Output result
    print(
        json.dumps(
            {
                "blend_path": args.blend,
                "num_objects": len(outputs),
                "status": "success"
            }
        )
    )


if __name__ == "__main__":
    main()


# python tools/sam3d_worker.py --image data/static_scene/christmas1/target.png --mask output/test/sam3/snowman_mask.npy --config utils/sam3d/checkpoints/hf/pipeline.yaml --glb output/test/sam3/snowman.glb

# python tools/sam3d_worker.py --image data/static_scene/blackhouse/target.jpeg --mask output/static_scene/20251205_030616/blackhouse/house_mask.npy --config utils/sam3d/checkpoints/hf/pipeline.yaml --glb output/static_scene/20251205_030616/blackhouse/house.glb