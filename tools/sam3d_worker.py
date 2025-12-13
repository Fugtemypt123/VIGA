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
    print(output)
    # output.keys: ['6drotation_normalized', 'scale', 'shape', 'translation', 'translation_scale', 'coords_original', 'coords', 'downsample_factor', 'rotation', 'mesh', 'gaussian', 'glb', 'gs', 'pointmap', 'pointmap_colors']
    
    glb_path = None
    glb = output.get("glb")
    if glb is not None and hasattr(glb, "export"):
        os.makedirs(os.path.dirname(args.glb), exist_ok=True)
        glb.export(args.glb)
        glb_path = args.glb
    
    # 提取变换信息
    translation = None
    if "translation" in output:
        trans_tensor = output["translation"]
        if hasattr(trans_tensor, "cpu"):
            translation = trans_tensor.cpu().numpy().tolist()
        elif hasattr(trans_tensor, "numpy"):
            translation = trans_tensor.numpy().tolist()
        else:
            translation = trans_tensor.tolist() if hasattr(trans_tensor, "tolist") else trans_tensor
    
    # 提取 rotation（3x3 矩阵）
    rotation_3x3 = None
    if "rotation" in output:
        rot_tensor = output["rotation"]
        if hasattr(rot_tensor, "cpu"):
            rot_np = rot_tensor.cpu().numpy()
        elif hasattr(rot_tensor, "numpy"):
            rot_np = rot_tensor.numpy()
        else:
            rot_np = rot_tensor
        
        # 如果是 quaternion，需要转换为 3x3 矩阵
        if rot_np.shape == (4,):  # quaternion [w, x, y, z] or [x, y, z, w]
            # 假设是 [w, x, y, z] 格式，转换为 3x3 矩阵
            import numpy as np
            if abs(rot_np[0]) > 0.9:  # w is first
                w, x, y, z = rot_np
            else:  # w is last
                x, y, z, w = rot_np
            # 转换为 3x3 旋转矩阵
            rotation_3x3 = np.array([
                [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
                [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
                [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
            ]).tolist()
        elif rot_np.shape == (3, 3):  # 已经是 3x3 矩阵
            rotation_3x3 = rot_np.tolist()
        else:
            rotation_3x3 = None
    
    translation_scale = None
    if "translation_scale" in output:
        scale_tensor = output["translation_scale"]
        if hasattr(scale_tensor, "cpu"):
            translation_scale = scale_tensor.cpu().numpy().tolist()
        elif hasattr(scale_tensor, "numpy"):
            translation_scale = scale_tensor.numpy().tolist()
        else:
            translation_scale = scale_tensor.tolist() if hasattr(scale_tensor, "tolist") else scale_tensor
    
    scale = None
    if "scale" in output:
        scale_tensor = output["scale"]
        if hasattr(scale_tensor, "cpu"):
            scale = scale_tensor.cpu().numpy().tolist()
        elif hasattr(scale_tensor, "numpy"):
            scale = scale_tensor.numpy().tolist()
        else:
            scale = scale_tensor.tolist() if hasattr(scale_tensor, "tolist") else scale_tensor
    
    print(
        json.dumps(
            {
                "glb_path": glb_path,
                "glb": glb_path,  # 新格式使用 "glb"
                "translation": translation,
                "translation_scale": translation_scale,
                "rotation": rotation_3x3,
                "scale": scale,
            }
        )
    )


if __name__ == "__main__":
    main()


# python tools/sam3d_worker.py --image data/static_scene/christmas1/target.png --mask output/test/sam3/snowman_mask.npy --config utils/sam3d/checkpoints/hf/pipeline.yaml --glb output/test/sam3/snowman.ply

# python tools/sam3d_worker.py --image data/static_scene/blackhouse/target.jpeg --mask output/static_scene/20251205_030616/blackhouse/house_mask.npy --config utils/sam3d/checkpoints/hf/pipeline.yaml --glb output/static_scene/20251205_030616/blackhouse/house.glb