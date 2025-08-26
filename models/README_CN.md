### 本地部署 Qwen2-VL-7B-Instruct（OpenAI 兼容接口）

该模块使用 vLLM 启动 `Qwen/Qwen2-VL-7B-Instruct` 的本地 OpenAI 兼容 HTTP 服务，并提供基于 OpenAI 客户端的文本与多模态（图片）调用示例。

### 环境要求

- Linux，NVIDIA GPU 与合适的 NVIDIA 驱动
- 与 PyTorch 匹配的 CUDA/cuDNN 环境
- Python 3.10+

说明：vLLM 会安装匹配的 `torch`，如果你需要特定 CUDA 版本的 wheel，请先单独安装对应的 PyTorch。

### 安装步骤

```bash
# 创建 Python 虚拟环境（位于项目根目录的 .venv 文件夹）
python -m venv .venv
# 激活虚拟环境（后续安装将仅作用于该环境）
source .venv/bin/activate
# 升级 pip 以避免旧版导致的安装问题
pip install --upgrade pip
# 安装本模块所需依赖（vLLM、openai 等）
pip install -r models/requirements.txt
```

如遇到 CUDA/Torch 安装问题，可先手动安装指定的 torch 版本，再安装 vLLM：

```bash
# 以 CUDA 12.1 为例（按需调整），先安装与 CUDA 匹配的 PyTorch
pip install --upgrade --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
# 再安装（或升级） vLLM，以使用你的本地 CUDA/PyTorch 配置
pip install --upgrade vllm
```

### 启动服务

```bash
#!/usr/bin/env bash
# 激活虚拟环境
source .venv/bin/activate
# 启动 vLLM 的 OpenAI 兼容服务（监听 0.0.0.0:8000）
python models/server.py --host 0.0.0.0 --port 8000 --model Qwen/Qwen2-VL-7B-Instruct --served-model-name Qwen2-VL-7B-Instruct --tensor-parallel-size 1 --gpu-memory-utilization 0.90 --max-model-len 32768
```

服务将暴露 OpenAI 兼容端点：`http://<host>:<port>/v1`

为 OpenAI 客户端设置环境变量（使用任意占位值即可）：

```bash
# 设置 OpenAI 客户端所需的 API Key（本地服务不校验真实有效性）
export OPENAI_API_KEY="not-needed"
```

### 使用 OpenAI 客户端（文本对话）

```bash
#!/usr/bin/env bash
# 激活虚拟环境
source .venv/bin/activate
# 使用 OpenAI 客户端调用本地服务进行文本对话
python models/client_chat.py --base-url http://localhost:8000/v1 --model Qwen2-VL-7B-Instruct \
  --prompt "用三句话介绍一下长城"
```

### 使用 OpenAI 客户端（多模态/视觉）

```bash
#!/usr/bin/env bash
# 激活虚拟环境
source .venv/bin/activate
# 使用 OpenAI 客户端调用本地服务进行视觉理解（图片 URL + 文本指令）
python models/client_vision.py --base-url http://localhost:8000/v1 --model Qwen2-VL-7B-Instruct \
  --image-url "https://raw.githubusercontent.com/openai/gpt-4-vision-preview/main/cats.jpg" \
  --prompt "这张图片里有几只猫？"
```

### 常见说明

- 多卡部署：将 `--tensor-parallel-size` 调大为 GPU 数以进行张量并行切分。
- 磁盘空间：首次运行会将模型权重下载到本地 Hugging Face 缓存目录。
- 视觉能力：Qwen2-VL 需要 `--trust-remote-code` 来启用视觉处理，启动脚本已默认开启。


