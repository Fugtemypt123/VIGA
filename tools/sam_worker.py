import os, sys, argparse, numpy as np
import cv2
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(ROOT, "utils", "sam"))

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


def panic_filtering_process(raw_masks):
    """
    raw_masks: 列表，每个元素包含:
               - 'segmentation': 二进制掩码 (H, W)
               - 'predicted_iou': 置信度分数 (float)
               - 'area': 像素面积 (int)
    """
    # ---------------------------------------------------------
    # 第一步：预处理 (Pre-filtering)
    # 解决 "问题2: 不适当的拆分" (如盒子上的文字)
    # ---------------------------------------------------------
    min_area_threshold = 100
    candidates = []
    for m in raw_masks:
        if m['area'] > min_area_threshold:
            candidates.append(m)
    
    if len(candidates) == 0:
        return []
    
    # ---------------------------------------------------------
    # 第二步：排序 (Sorting)
    # 解决 "问题1: 不适当的合并"
    # ---------------------------------------------------------
    candidates.sort(key=lambda x: x['predicted_iou'], reverse=True)
    
    # ---------------------------------------------------------
    # 第三步：贪婪填充 (Greedy Filling)
    # ---------------------------------------------------------
    height, width = candidates[0]['segmentation'].shape
    occupancy_mask = np.zeros((height, width), dtype=bool)
    
    final_masks = []
    MIN_NEW_AREA_RATIO = 0.6
    
    for mask_data in candidates:
        current_seg = mask_data['segmentation']  # 当前 Mask 的二进制图
        mask_area = mask_data['area']
        
        # 计算当前 Mask 和"已占用区域"的重叠
        intersection = np.logical_and(current_seg, occupancy_mask)
        intersection_area = np.count_nonzero(intersection)
        
        # 计算这一轮能贡献多少"新鲜像素"
        new_area = mask_area - intersection_area
        
        # 计算新鲜度比例
        keep_ratio = new_area / mask_area if mask_area > 0 else 0
        
        # --- 核心决策逻辑 ---
        if keep_ratio > MIN_NEW_AREA_RATIO:
            # 决策：保留这个 Mask (方案 B: 保留完整 Mask，允许轻微重叠)
            final_masks.append(mask_data)
            
            # 更新占用图：将当前 Mask 的区域标记为已占用
            occupancy_mask = np.logical_or(occupancy_mask, current_seg)
    
    return final_masks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--checkpoint", default=None)
    args = p.parse_args()
    
    # 设置默认 checkpoint 路径
    if args.checkpoint is None:
        args.checkpoint = os.path.join(ROOT, "utils", "sam", "sam_vit_h_4b8939.pth")
    
    # 加载图像
    image = cv2.imread(args.image)
    if image is None:
        raise ValueError(f"Failed to load image: {args.image}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 初始化 SAM 模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_type = "vit_h"
    
    sam = sam_model_registry[model_type](checkpoint=args.checkpoint)
    sam.to(device=device)
    
    # 生成所有 mask
    mask_generator = SamAutomaticMaskGenerator(sam)
    raw_masks = mask_generator.generate(image)
    
    # 应用 panic filtering process
    filtered_masks = panic_filtering_process(raw_masks)
    
    # 将过滤后的 masks 转换为 numpy 数组并保存
    # 每个 mask 的 segmentation 是 (H, W) 的布尔数组，转换为 uint8 (0/255)
    if len(filtered_masks) == 0:
        print("Warning: No masks after filtering")
        mask_arrays = np.array([], dtype=np.uint8)
    else:
        mask_arrays = []
        for mask_data in filtered_masks:
            seg = mask_data['segmentation']  # 布尔数组
            mask_uint8 = (seg.astype(np.uint8)) * 255  # 转换为 0/255
            mask_arrays.append(mask_uint8)
        
        # 所有 mask 尺寸相同，可以堆叠为 3D 数组 (N, H, W)
        mask_arrays = np.stack(mask_arrays, axis=0)
    
    # 保存为 .npy 文件
    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else ".", exist_ok=True)
    np.save(args.out, mask_arrays)
    
    print(f"Generated {len(raw_masks)} raw masks, filtered to {len(filtered_masks)} masks")
    print(f"Saved filtered masks to: {args.out}")


if __name__ == "__main__":
    main()


# python tools/sam_worker.py --image data/static_scene/christmas/target.png --out output/test/sam/all_masks.npy

