import os, sys, argparse, json, numpy as np
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(ROOT, "utils", "sam3d", "notebook"))

from inference import Inference


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--mask", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--glb", required=True)
    args = p.parse_args()

    inf = Inference(args.config, compile=False)
    img = np.array(Image.open(args.image))
    mask = np.load(args.mask)
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]
    mask = (mask > 0).astype("uint8") * 255
    out = inf(img, mask, seed=42)
    glb_path = None
    glb = out.get("glb")
    if glb is not None and hasattr(glb, "export"):
        os.makedirs(os.path.dirname(args.glb), exist_ok=True)
        glb.export(args.glb)
        glb_path = args.glb
    print(
        json.dumps(
            {
                "glb_path": glb_path,
                "translation": None if "translation" not in out else out["translation"].cpu().numpy().tolist(),
                "rotation": None if "rotation" not in out else out["rotation"].cpu().numpy().tolist(),
                "scale": None if "scale" not in out else out["scale"].cpu().numpy().tolist(),
            }
        )
    )


if __name__ == "__main__":
    main()


