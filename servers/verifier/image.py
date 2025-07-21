import os
import base64
import io
import re
import sys
import traceback
from io import StringIO
from PIL import Image, ImageChops, ImageEnhance
import numpy as np
from typing import Dict
from mcp.server.fastmcp import FastMCP
import logging
import openai

class PILExecutor:
    def __init__(self):
        self._setup_environment()

    def _setup_environment(self):
        self.globals = {
            'Image': Image,
            'io': io,
            'base64': base64,
            'current_image': None,
            'result': None
        }

    def _image_to_base64(self, image: Image.Image) -> str:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    def execute(self, code: str) -> Dict:
        stdout_capture = StringIO()
        stderr_capture = StringIO()
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = stdout_capture, stderr_capture

        try:
            exec(code, self.globals)
            result = self.globals.get('result', None)
            if isinstance(result, Image.Image):
                result = self._image_to_base64(result)
            return {
                'success': True,
                'result': result,
                'stdout': stdout_capture.getvalue(),
                'stderr': stderr_capture.getvalue()
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'stdout': stdout_capture.getvalue(),
                'stderr': stderr_capture.getvalue() + traceback.format_exc()
            }
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr

class ImageDifferentiationTool:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided.")
        openai.api_key = self.api_key

    def pil_to_base64(self, image: Image.Image) -> str:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _highlight_differences(self, img1, img2, diff, threshold=50):
        img1_array = np.array(img1)
        img2_array = np.array(img2)
        diff_array = np.array(diff)
        mask = np.any(diff_array > threshold, axis=2)

        highlight = np.array([255, 0, 0])
        img1_high = img1_array.copy()
        img2_high = img2_array.copy()

        img1_high[mask] = ((img1_high[mask] * 0.5 + highlight * 0.5)).astype(np.uint8)
        img2_high[mask] = ((img2_high[mask] * 0.5 + highlight * 0.5)).astype(np.uint8)

        return Image.fromarray(img1_high), Image.fromarray(img2_high)

    def describe_difference(self, path1: str, path2: str) -> str:
        img1 = Image.open(path1).convert("RGB")
        img2 = Image.open(path2).convert("RGB")
        if img1.size != img2.size:
            img2 = img2.resize(img1.size)

        diff = ImageChops.difference(img1, img2)
        img1_high, img2_high = self._highlight_differences(img1, img2, diff)

        enhancer = ImageEnhance.Brightness(diff)
        diff_bright = enhancer.enhance(4.0)

        b64s = [self.pil_to_base64(im) for im in [img1, img2, img1_high, img2_high]]

        messages = [
            {"role": "system", "content": "You are an expert in image comparison."},
            {"role": "user", "content": [
                {"type": "text", "text": "Compare the two original images and describe the highlighted red difference."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64s[0]}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64s[1]}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64s[2]}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64s[3]}"}},
            ]}
        ]

        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=512
        )
        return response.choices[0].message.content

def main():
    server = FastMCP("image-server")

    @server.tool()
    def initialize_executor(api_key: str) -> dict:
        """
        初始化ImageDifferentiationTool，设置api_key。
        """
        try:
            global _image_tool
            _image_tool = ImageDifferentiationTool(api_key=api_key)
            return {"status": "success", "message": "ImageDifferentiationTool initialized with api_key."}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    @server.tool()
    def exec_pil_code(code: str) -> dict:
        tool = PILExecutor()
        return tool.execute(code)

    @server.tool()
    def compare_images(path1: str, path2: str) -> dict:
        try:
            global _image_tool
            if '_image_tool' not in globals() or _image_tool is None:
                raise RuntimeError("ImageDifferentiationTool not initialized. Call initialize_executor first.")
            result = _image_tool.describe_difference(path1, path2)
            return {"description": result}
        except Exception as e:
            logging.error(f"Comparison failed: {e}")
            return {"error": str(e)}

    server.run(transport="stdio")

if __name__ == "__main__":
    main()
