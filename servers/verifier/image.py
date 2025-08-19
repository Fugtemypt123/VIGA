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

# 创建全局 MCP 实例
mcp = FastMCP("image-server")

# 全局工具实例
_image_tool = None
_pil_executor = None

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
    def __init__(self, api_key: str = None, api_base_url: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided.")
        openai.api_key = self.api_key
        # Allow overriding OpenAI-compatible base URL (e.g., Azure, local proxy)
        self.api_base_url = api_base_url or os.getenv("OPENAI_BASE_URL")
        if self.api_base_url:
            try:
                # For openai>=1.0 clients, base_url is set on client, but the legacy interface can respect env var
                openai.base_url = self.api_base_url
            except Exception:
                pass

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
            {"role": "system", "content": "You are an expert in image comparison. You will receive two original images and a difference-highlighted version. Describe the visual differences in natural language in detail. If there is almost no difference, just say 'No difference'."},
            {"role": "user", "content": [
                {"type": "text", "text": "Here are the two original images and their highlighted (red) visual difference parts. Please focus on the highlighted parts and describe the visual difference in these parts. If there is almost no difference, just say 'No difference'."},
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

@mcp.tool()
def initialize_executor(api_key: str, api_base_url: str = None) -> dict:
    """
    初始化ImageDifferentiationTool，设置api_key。
    """
    global _image_tool
    try:
        _image_tool = ImageDifferentiationTool(api_key=api_key, api_base_url=api_base_url)
        return {"status": "success", "message": "ImageDifferentiationTool initialized with api_key."}
    except Exception as e:
        return {"status": "error", "error": str(e)}

@mcp.tool()
def exec_pil_code(code: str) -> dict:
    """
    执行传入的 PIL Python 代码，并返回执行结果。
    """
    global _pil_executor
    if _pil_executor is None:
        _pil_executor = PILExecutor()
    
    try:
        result = _pil_executor.execute(code)
        return {"status": "success", "result": result}
    except Exception as e:
        return {"status": "error", "error": str(e)}

@mcp.tool()
def compare_images(path1: str, path2: str) -> dict:
    """
    比较两个图像并返回差异描述。
    需要先调用 initialize_executor 进行初始化。
    """
    global _image_tool
    if _image_tool is None:
        return {"status": "error", "error": "ImageDifferentiationTool not initialized. Call initialize_executor first."}
    
    try:
        result = _image_tool.describe_difference(path1, path2)
        return {"status": "success", "description": result}
    except Exception as e:
        logging.error(f"Comparison failed: {e}")
        return {"status": "error", "error": str(e)}

def main():
    # 检查是否直接运行此脚本（用于测试）
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        print("Running image.py tools test...")
        test_tools()
    else:
        # 正常运行 MCP 服务器
        mcp.run(transport="stdio")

def test_tools():
    """测试所有工具函数"""
    print("=" * 50)
    print("Testing Image Tools")
    print("=" * 50)
    
    # 测试 1: 初始化执行器
    print("\n1. Testing initialize_executor...")
    try:
        # 尝试从环境变量获取 API key，如果没有则使用测试 key
        api_key = os.getenv("OPENAI_API_KEY", "test_key_for_testing")
        result = initialize_executor(api_key)
        print(f"Result: {result}")
        if result.get("status") == "success":
            print("✓ initialize_executor passed")
        else:
            print("✗ initialize_executor failed")
    except Exception as e:
        print(f"✗ initialize_executor failed with exception: {e}")
    
    # 测试 2: 执行 PIL 代码
    print("\n2. Testing exec_pil_code...")
    try:
        # 创建一个简单的测试图像
        test_code = """
from PIL import Image
import numpy as np

# 创建一个简单的测试图像
img = Image.new('RGB', (100, 100), color='red')
result = img
"""
        result = exec_pil_code(test_code)
        print(f"Result: {result}")
        if result.get("status") == "success":
            print("✓ exec_pil_code passed")
        else:
            print("✗ exec_pil_code failed")
    except Exception as e:
        print(f"✗ exec_pil_code failed with exception: {e}")
    
    # 测试 3: 比较图像（需要创建测试图像文件）
    print("\n3. Testing compare_images...")
    try:
        # 创建两个测试图像文件
        test_img1 = Image.new('RGB', (100, 100), color='red')
        test_img2 = Image.new('RGB', (100, 100), color='blue')
        
        test_path1 = "/home/shaofengyin/AgenticVerifier/output/renders/1/render1.png"
        test_path2 = "/home/shaofengyin/AgenticVerifier/output/renders/2/render1.png"
        
        test_img1.save(test_path1)
        test_img2.save(test_path2)
        
        # 只有在有有效 API key 时才测试图像比较
        if os.getenv("OPENAI_API_KEY"):
            result = compare_images(test_path1, test_path2)
            print(f"Result: {result}")
            if result.get("status") == "success":
                print("✓ compare_images passed")
            else:
                print("✗ compare_images failed")
        else:
            print("⚠ Skipping compare_images test (no OPENAI_API_KEY)")
        
        # 清理测试文件
        if os.path.exists(test_path1):
            os.remove(test_path1)
        if os.path.exists(test_path2):
            os.remove(test_path2)
            
    except Exception as e:
        print(f"✗ compare_images failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print("Test completed!")
    print("=" * 50)
    print("\nTo run the MCP server normally, use:")
    print("python image.py")
    print("\nTo run tests, use:")
    print("python image.py --test")

if __name__ == "__main__":
    main()
