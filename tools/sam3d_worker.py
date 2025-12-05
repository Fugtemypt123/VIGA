import os, sys, argparse, json, numpy as np
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(ROOT, "utils", "sam3d", "notebook"))

if "CONDA_PREFIX" not in os.environ:
    python_bin = sys.executable
    conda_env = os.path.dirname(os.path.dirname(python_bin))
    os.environ["CONDA_PREFIX"] = conda_env

from inference import Inference, load_image, load_mask


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--mask", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--glb", required=True)
    args = p.parse_args()

    inference = Inference(args.config, compile=False)
    image = load_image(args.image)
    # args.mask 现在是 .npy 文件路径，直接加载为 numpy 数组
    mask = np.load(args.mask)
    mask = mask > 0
    output = inference(image, mask, seed=42)
    
    glb_path = None
    glb = output.get("glb")
    if glb is not None and hasattr(glb, "export"):
        os.makedirs(os.path.dirname(args.glb), exist_ok=True)
        glb.export(args.glb)
        glb_path = args.glb
    print(
        json.dumps(
            {
                "glb_path": glb_path,
                "translation": None if "translation" not in output else output["translation"].cpu().numpy().tolist(),
                "rotation": None if "rotation" not in output else output["rotation"].cpu().numpy().tolist(),
                "scale": None if "scale" not in output else output["scale"].cpu().numpy().tolist(),
            }
        )
    )


if __name__ == "__main__":
    main()


# python tools/sam3d_worker.py --image data/static_scene/christmas/target.png --mask output/test/sam3/snowman_mask.npy --config utils/sam3d/checkpoints/hf/pipeline.yaml --glb output/test/sam3/snowman.ply

# python tools/sam3d_worker.py --image data/static_scene/blackhouse/target.jpeg --mask output/static_scene/20251205_030616/blackhouse/house_mask.npy --config utils/sam3d/checkpoints/hf/pipeline.yaml --glb output/static_scene/20251205_030616/blackhouse/house.glb