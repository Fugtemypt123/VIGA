#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_video.py
把每一步的 thought+code（左半屏）与 render 图（右半屏）拼成视频。
用法示例：
    python make_video.py --steps steps.json --renders_dir renders \
        --out video.mp4 --width 1920 --height 1080 --fps 30 --step_duration 2.5

注意：
- 默认在 renders_dir 中按 step_001.png, step_002.png... 寻找右半屏图像；
  若 steps.json 的该步含有 image_path 且文件存在，会优先使用该路径。
- 左半屏会自动换行、代码高亮；超长会出现可控滚动（以确保内容展示）。
"""

import os
import json
import math
import argparse
from pathlib import Path
from typing import Tuple

from PIL import Image, ImageDraw, ImageFont
import imageio.v2 as imageio

# 可选：代码高亮（若未安装 Pygments，会退化为无高亮的等宽绘制）
try:
    from pygments import highlight
    from pygments.lexers import PythonLexer
    from pygments.formatters import ImageFormatter
    HAS_PYGMENTS = True
except Exception:
    HAS_PYGMENTS = False

# ---------- 可调样式 ----------
LEFT_BG = (20, 22, 26)     # 左半背景色
TEXT_COLOR = (235, 235, 235)
THOUGHT_COLOR = (180, 220, 255)
SEPARATOR_COLOR = (70, 70, 80)
PADDING = 28               # 左半内边距
GAP = 18                   # thought 与 code 之间的间距
SCROLL_MARGIN = 20
DEFAULT_FONT = None        # 自动找系统等宽字体；也可自行设为具体 .ttf 路径
MONO_FALLBACKS = ["Menlo.ttc", "Consolas.ttf", "DejaVuSansMono.ttf", "JetBrainsMono-Regular.ttf"]

def find_mono_font():
    if DEFAULT_FONT and Path(DEFAULT_FONT).exists():
        return DEFAULT_FONT
    # 简单探测常见字体
    for name in MONO_FALLBACKS:
        for d in ["/usr/share/fonts", "/Library/Fonts", str(Path.home() / "Library/Fonts"), "C:\\Windows\\Fonts"]:
            p = Path(d) / name
            if p.exists():
                return str(p)
    return None

def wrap_text(draw: ImageDraw.ImageDraw, text: str, font, max_width: int):
    lines = []
    for para in text.split("\n"):
        if not para:
            lines.append("")
            continue
        # 基于字符宽度的简单折行
        buf = ""
        for ch in para:
            test = buf + ch
            w, _ = draw.textsize(test, font=font)
            if w <= max_width:
                buf = test
            else:
                lines.append(buf)
                buf = ch
        lines.append(buf)
    return lines

def render_left_panel(size: Tuple[int,int], thought: str, code: str, scroll_px: int = 0):
    W, H = size
    img = Image.new("RGB", (W, H), LEFT_BG)
    draw = ImageDraw.Draw(img)

    font_path = find_mono_font()
    font_thought = ImageFont.truetype(font_path, size=28) if font_path else ImageFont.load_default()
    font_code   = ImageFont.truetype(font_path, size=24) if font_path else ImageFont.load_default()
    font_label  = ImageFont.truetype(font_path, size=22) if font_path else ImageFont.load_default()

    x = PADDING
    y = PADDING

    # 标题
    draw.text((x, y), "Thought", fill=THOUGHT_COLOR, font=font_label)
    y += 8 + font_label.size

    # Thought 文本
    max_text_w = W - 2 * PADDING
    thought_lines = wrap_text(draw, thought, font_thought, max_text_w)
    for line in thought_lines:
        draw.text((x, y), line, fill=TEXT_COLOR, font=font_thought)
        y += font_thought.size + 6

    y += GAP

    # 分隔线
    draw.line((PADDING, y, W - PADDING, y), fill=SEPARATOR_COLOR, width=2)
    y += GAP

    # Code 标题
    draw.text((x, y), "Code (Python)", fill=(180, 255, 200), font=font_label)
    y += 8 + font_label.size

    # 代码区矩形（可滚动）
    code_top_y = y
    code_box_h = H - PADDING - y
    code_box = Image.new("RGB", (max_text_w, code_box_h), (16, 16, 18))

    if HAS_PYGMENTS:
        # 使用 Pygments 生成图片后，粘贴进来（同宽）
        formatter = ImageFormatter(
            font_name="DejaVu Sans Mono",
            line_numbers=False,
            image_format="PNG",
            line_pad=2,
            font_size=20,
            style="native",
        )
        code_img_bytes = highlight(code, PythonLexer(), formatter)
        code_img = Image.open(imageio.core.asarray(code_img_bytes)).convert("RGB")
        # 缩放到宽度适配
        ratio = max_text_w / code_img.width if code_img.width > 0 else 1.0
        new_h = max(1, int(code_img.height * ratio))
        code_img = code_img.resize((max_text_w, new_h), Image.BICUBIC)
    else:
        # 简单等宽绘制
        code_img = Image.new("RGB", (max_text_w, 10), (16,16,18))
        d2 = ImageDraw.Draw(code_img)
        code_lines = code.split("\n")
        h_lines = []
        for ln in code_lines:
            h_lines.append(ln if ln else " ")
        # 估高
        code_img_h = max(10, (font_code.size + 6) * len(h_lines) + 10)
        code_img = Image.new("RGB", (max_text_w, code_img_h), (16,16,18))
        d2 = ImageDraw.Draw(code_img)
        yy = 6
        for ln in h_lines:
            d2.text((8, yy), ln, fill=(220,220,220), font=font_code)
            yy += font_code.size + 6

    # 处理滚动（超出高度时）
    max_scroll = max(0, code_img.height - code_box_h)
    s = max(0, min(scroll_px, max_scroll))
    code_crop = code_img.crop((0, s, max_text_w, min(s + code_box_h, code_img.height)))
    code_box.paste(code_crop, (0,0))
    img.paste(code_box, (x, code_top_y))

    # 滚动条简易指示
    if max_scroll > 0:
        bar_h = max(30, int(code_box_h * (code_box_h / code_img.height)))
        bar_y = code_top_y + int((code_box_h - bar_h) * (s / max_scroll))
        draw.rectangle(
            (W - PADDING - 6, bar_y, W - PADDING - 2, bar_y + bar_h),
            fill=(90, 90, 100)
        )

    return img

def compose_step_frame(total_size: Tuple[int,int], left_img: Image.Image, right_img: Image.Image):
    W, H = total_size
    half_w = W // 2
    canvas = Image.new("RGB", (W, H), (0,0,0))

    # 右图等比适配右半屏
    if right_img is None:
        right_img = Image.new("RGB", (half_w, H), (10, 10, 12))
    else:
        ratio = min(half_w / right_img.width, H / right_img.height)
        new_w, new_h = max(1, int(right_img.width * ratio)), max(1, int(right_img.height * ratio))
        right_img = right_img.resize((new_w, new_h), Image.BICUBIC)
    # 居中放置右图
    rx = W - half_w + (half_w - right_img.width)//2
    ry = (H - right_img.height)//2

    canvas.paste(left_img.resize((half_w, H), Image.BICUBIC), (0,0))
    canvas.paste(right_img, (rx, ry))
    return canvas

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=str, required=True, help="/home/shaofengyin/AgenticVerifier/output/static_scene/demo/20251028_133713/christmas1/generator_memory.json")
    ap.add_argument("--out", type=str, default="video.mp4")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--step_duration", type=float, default=1.0, help="每步停留时长（秒）")
    ap.add_argument("--scroll_code", action="store_true", help="若代码过长则缓慢向下滚动")
    args = ap.parse_args()
    
    args.renders_dir = Path(args.steps).parent / "renders"

    steps = json.loads(Path(args.steps).read_text(encoding="utf-8"))
    frames = []
    frames_per_step = max(1, int(args.fps * args.step_duration))

    for i, complete_step in enumerate(steps, start=1):
        if 'tool_calls' not in complete_step:
            continue
        tool_call = complete_step['tool_calls'][0]
        if tool_call['function']['name'] != "execute_and_evaluate":
            continue
        step = json.loads(tool_call['function']['arguments'])
        thought = step.get("thought", "").strip()
        if "code" in step:
            code = step.get("code", "").strip()
        else:
            code = step.get("full_code", "").strip()
            
        if i+2 >= len(steps):
            continue
        user_message = steps[i+2]
        if user_message['role'] != 'user':
            continue
        if len(user_message['content']) < 3:
            continue
        if 'Image loaded from local path: ' not in user_message['content'][2]['text']:
            continue
        image_path = user_message['content'][2]['text'].split("Image loaded from local path: ")[1]
        right_img = Image.open(image_path).convert("RGB")

        # 计算滚动范围（如果开启滚动）
        max_scroll = 0
        if args.scroll_code:
            # 先渲染一次左半，估算代码图高度
            tmp_left = render_left_panel((args.width//2, args.height), thought, code, 0)
            # 粗略：再渲染一次获取最大滚动（这里简单取高度差的 2 倍当作上限）
            # 为简化，我们在循环里渐进增加 scroll_px
            max_scroll = args.height // 2

        for f in range(frames_per_step):
            scroll_px = 0
            if args.scroll_code and max_scroll > 0:
                # 匀速滚动到 max_scroll
                scroll_px = int(max_scroll * (f / max(1, frames_per_step-1)))
            left_img = render_left_panel((args.width//2, args.height), thought, code, scroll_px)
            composed = compose_step_frame((args.width, args.height), left_img, right_img)
            frames.append(composed)

    # 写视频
    writer = imageio.get_writer(args.out, fps=args.fps, codec="libx264", quality=8)
    for im in frames:
        writer.append_data(imageio.core.asarray(im))
    writer.close()
    print(f"[OK] 写出视频：{args.out}")

if __name__ == "__main__":
    main()
