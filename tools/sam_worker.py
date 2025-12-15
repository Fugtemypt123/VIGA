import os, sys, argparse, numpy as np
import cv2
import torch
import re
import json
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, "utils", "sam"))
sys.path.append(os.path.join(ROOT, "utils"))

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from common import build_client, get_image_base64


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
    
    # ---------------------------------------------------------
    # 第四步：按 area 排序并只返回前 k 个
    # ---------------------------------------------------------
    final_masks.sort(key=lambda x: x['area'], reverse=True)
    k = 15
    final_masks = final_masks[:k]
    
    return final_masks


def save_mask_as_png(mask_data, original_image, output_path):
    """
    将单个 mask 保存为 PNG 图片
    segmentation=1 的像素保持原图颜色，segmentation=0 的像素设为透明
    参考 utils/sam/demo.py 中的 save_ann_masks 函数
    """
    m = mask_data['segmentation']  # 布尔数组 (H, W)
    h, w = m.shape
    
    # 确保原图尺寸匹配
    if original_image.shape[:2] != (h, w):
        # 如果尺寸不匹配，调整原图
        original_image = cv2.resize(original_image, (w, h))
    
    # 创建 RGBA 图像
    rgba_image = np.zeros((h, w, 4), dtype=np.uint8)
    
    # segmentation=1 的地方：使用原图颜色，alpha=255（不透明）
    rgba_image[m, :3] = original_image[m]  # RGB 通道
    rgba_image[m, 3] = 255  # Alpha 通道
    
    # segmentation=0 的地方：保持为 0（透明），alpha=0
    
    # 保存为 PNG
    pil_image = Image.fromarray(rgba_image, 'RGBA')
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    pil_image.save(output_path)
    return output_path


def sanitize_filename(name):
    """
    将文件名中的非法字符替换为下划线
    保留字母、数字、下划线和连字符
    """
    # 替换所有非字母数字、下划线、连字符的字符为下划线
    sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
    # 移除连续的下划线
    sanitized = re.sub(r'_+', '_', sanitized)
    # 移除开头和结尾的下划线
    sanitized = sanitized.strip('_')
    # 如果为空，使用默认名称
    if not sanitized:
        sanitized = "object"
    return sanitized

def get_object_name_from_vlm(image_path, ori_img_path, model="gpt-4o", existing_names=None):
    """
    使用 VLM 识别图片中的物体并返回一个唯一的名称
    
    Args:
        image_path: PNG 图片路径
        ori_img_path: 原始图片路径
        model: VLM 模型名称
        existing_names: 已存在的名称列表，用于确保不重复
    
    Returns:
        物体名称（字符串）
    """
    if existing_names is None:
        existing_names = []
    
    try:
        # 编码图片
        image_b64 = get_image_base64(image_path)
        ori_img_b64 = get_image_base64(ori_img_path)
        # 初始化 OpenAI client
        client = build_client(model)
        
        # 构建提示词，包含已存在的名称以避免重复
        existing_names_str = ""
        if existing_names:
            existing_names_str = f"\n\nAlready identified objects (do not use these names): {', '.join(existing_names)}"
        
        # 创建消息
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_b64
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": ori_img_b64
                        }
                    },
                    {
                        "type": "text",
                        "text": f"Look at the first image showing a segmented object, and the second image showing the original image that contains this object. Identify what this object is and provide a concise, descriptive name for it (e.g., 'red_chair', 'wooden_table'). If the first image is not clear, check the second image to get the whole context.\n\nIf you think the first image is not a object, but a background, please return 'background' as the object name.\n\n Use only lowercase letters, numbers, and underscores. The name should be a single word or short phrase (2-3 words max, use underscores to separate words).{existing_names_str}\n\nRespond with ONLY the object name, nothing else."
                    }
                ]
            }
        ]
        
        # 调用 API
        response = client.chat.completions.create(model=model, messages=messages)
        
        # 解析响应
        object_name = response.choices[0].message.content.strip()
        
        # 清理名称：移除引号、多余空格等
        object_name = object_name.strip('"\'')
        object_name = re.sub(r'\s+', '_', object_name)  # 空格替换为下划线
        object_name = sanitize_filename(object_name)
        
        # 确保不重复：如果重复，添加数字后缀
        base_name = object_name
        counter = 1
        while object_name in existing_names:
            object_name = f"{base_name}_{counter}"
            counter += 1
        
        return object_name
        
    except Exception as e:
        print(f"VLM naming failed: {e}, using fallback name")
        # 如果 VLM 调用失败，使用默认名称
        base_name = "object"
        counter = 1
        while f"{base_name}_{counter}" in existing_names:
            counter += 1
        return f"{base_name}_{counter}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--vlm-model", default="gpt-4o", help="VLM model to use for object identification")
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
    
    if len(filtered_masks) == 0:
        print("Warning: No masks after filtering")
        return
    
    # 确定输出目录
    output_dir = os.path.dirname(args.out)
    os.makedirs(output_dir, exist_ok=True)
    
    # 为每个 mask 保存 PNG 并使用 VLM 命名
    object_names = []
    mask_files = []  # 存储每个 mask 的文件路径和名称信息
    
    print(f"Processing {len(filtered_masks)} filtered masks...")
    for idx, mask_data in enumerate(filtered_masks):
        # 1. 保存 PNG（临时文件名，稍后会重命名）
        temp_png_path = os.path.join(output_dir, f"temp_mask_{idx}.png")
        ori_img_path = args.image
        save_mask_as_png(mask_data, image, temp_png_path)
        
        # 2. 使用 VLM 识别物体并获取名称
        print(f"  Identifying object {idx+1}/{len(filtered_masks)}...")
        object_name = get_object_name_from_vlm(temp_png_path, ori_img_path, model=args.vlm_model, existing_names=object_names)
        if object_name == "background":
            os.remove(temp_png_path)
            continue
        object_names.append(object_name)
        
        # 3. 重命名 PNG 文件为 object_ID.png
        final_png_path = os.path.join(output_dir, f"{object_name}.png")
        if os.path.exists(final_png_path):
            # 如果文件已存在，添加数字后缀
            counter = 1
            while os.path.exists(final_png_path):
                final_png_path = os.path.join(output_dir, f"{object_name}_{counter}.png")
                counter += 1
        os.rename(temp_png_path, final_png_path)
        print(f"    Identified as: {object_name}")
        
        # 4. 保存对应的 npy 文件为 object_ID.npy
        seg = mask_data['segmentation']  # 布尔数组
        mask_uint8 = (seg.astype(np.uint8)) * 255  # 转换为 0/255
        
        npy_path = os.path.join(output_dir, f"{object_name}.npy")
        if os.path.exists(npy_path):
            # 如果文件已存在，添加数字后缀
            counter = 1
            while os.path.exists(npy_path):
                npy_path = os.path.join(output_dir, f"{object_name}_{counter}.npy")
                counter += 1
        np.save(npy_path, mask_uint8)
        
        mask_files.append({
            "object_id": object_name,
            "png_path": final_png_path,
            "npy_path": npy_path,
            "mask_data": mask_uint8
        })
    
    # 5. 如果指定了输出文件，也保存一个包含所有 masks 的堆叠数组（向后兼容）
    if args.out:
        mask_arrays = [item["mask_data"] for item in mask_files]
        if mask_arrays:
            mask_arrays = np.stack(mask_arrays, axis=0)
            np.save(args.out, mask_arrays)
            print(f"Also saved combined masks to: {args.out}")
        
        # 保存物体名称映射信息，用于后续 sam_init.py 使用
        object_names_json_path = args.out.replace('.npy', '_object_names.json')
        # 从 npy_path 中提取最终的文件名（不带扩展名），因为可能包含了冲突后缀
        object_mapping = []
        for item in mask_files:
            npy_path = item["npy_path"]
            # 提取文件名（不包括路径和扩展名）
            npy_filename = os.path.basename(npy_path)
            object_name_final = os.path.splitext(npy_filename)[0]  # 去掉 .npy 扩展名
            object_mapping.append(object_name_final)
        
        object_names_info = {
            "object_names": object_names,  # 按顺序的物体基础名称列表
            "object_mapping": object_mapping  # 每个 mask 的最终文件名（按顺序，包含冲突后缀如 _1, _2 等）
        }
        with open(object_names_json_path, 'w') as f:
            json.dump(object_names_info, f, indent=2)
        print(f"Also saved object names mapping to: {object_names_json_path}")
    
    print(f"\nGenerated {len(raw_masks)} raw masks, filtered to {len(filtered_masks)} masks")
    print(f"Identified objects: {', '.join(object_names)}")
    print(f"Saved {len(mask_files)} mask files to: {output_dir}")


if __name__ == "__main__":
    main()


# python tools/sam_worker.py --image data/static_scene/christmas1/target.png --out output/test/sam/all_masks.npy

