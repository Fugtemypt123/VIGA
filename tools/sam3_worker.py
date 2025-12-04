import os, sys, argparse, numpy as np
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(ROOT, "utils", "sam3"))

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--object", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    model = build_sam3_image_model()
    proc = Sam3Processor(model)
    img = Image.open(args.image)
    state = proc.set_image(img)
    out = proc.set_text_prompt(state=state, prompt=args.object)
    masks = out["masks"]
    if not masks:
        raise SystemExit("no mask found")
    mask = masks[0]
    if hasattr(mask, "cpu"):
        mask = mask.cpu().numpy()
    elif hasattr(mask, "numpy"):
        mask = mask.numpy()
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]
    mask = (mask > 0.5).astype("uint8") * 255
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.save(args.out, mask)


if __name__ == "__main__":
    main()


