### Local OpenAI-compatible server for Qwen2-VL-7B-Instruct (Vision + Text)

This module launches a local OpenAI-compatible HTTP server powered by vLLM to serve `Qwen/Qwen2-VL-7B-Instruct`. It also includes minimal OpenAI client examples for chat and vision.

### Prerequisites

- Linux with NVIDIA GPU and recent NVIDIA drivers
- CUDA/cuDNN compatible with your PyTorch build
- Python 3.10+

Note: vLLM will install a matching `torch` build. If you need a specific CUDA wheel, install that first.

### Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r models/requirements.txt
```

If you run into CUDA/Torch issues, install a specific torch build first, then reinstall vLLM:

```bash
# Example for CUDA 12.1 (adjust if needed)
pip install --upgrade --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
pip install --upgrade vllm
```

### Start the server

```bash
source .venv/bin/activate
python models/server.py --host 0.0.0.0 --port 8000 \
  --model Qwen/Qwen2-VL-7B-Instruct \
  --served-model-name Qwen2-VL-7B-Instruct \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 32768
```

Server will expose OpenAI-compatible endpoints at `http://<host>:<port>/v1`.

Environment variable for OpenAI clients (a dummy key is fine):

```bash
export OPENAI_API_KEY="not-needed"
```

### Test with OpenAI client (chat)

```bash
source .venv/bin/activate
python models/client_chat.py --base-url http://localhost:8000/v1 --model Qwen2-VL-7B-Instruct \
  --prompt "用三句话介绍一下长城"
```

### Test with OpenAI client (vision)

```bash
source .venv/bin/activate
python models/client_vision.py --base-url http://localhost:8000/v1 --model Qwen2-VL-7B-Instruct \
  --image-url "https://raw.githubusercontent.com/openai/gpt-4-vision-preview/main/cats.jpg" \
  --prompt "这张图片里有几只猫？"
```

### Notes

- Multi-GPU: increase `--tensor-parallel-size` to the number of GPUs to shard across.
- Disk space: first run will download model weights to your Hugging Face cache.
- Vision: vLLM requires `--trust-remote-code` to enable Qwen2-VL vision processing; this is enabled by the launcher.


