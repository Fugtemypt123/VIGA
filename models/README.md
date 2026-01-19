# Models

Local OpenAI-compatible server for self-hosted vision-language models.

## Overview

This module launches a vLLM-powered HTTP server that serves `Qwen/Qwen2-VL-7B-Instruct` with an OpenAI-compatible API. Useful for running VIGA with local models instead of cloud APIs.

## Files

| File | Description |
|------|-------------|
| `server.py` | vLLM server launcher |
| `client_chat.py` | Example chat client |
| `client_vision.py` | Example vision client |
| `requirements.txt` | Python dependencies |

## Prerequisites

- Linux with NVIDIA GPU
- CUDA/cuDNN compatible with PyTorch
- Python 3.10+

## Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r models/requirements.txt
```

If you encounter CUDA/Torch issues:

```bash
pip install --upgrade --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
pip install --upgrade vllm
```

## Usage

### Start Server

```bash
python models/server.py --host 0.0.0.0 --port 8000 \
  --model Qwen/Qwen2-VL-7B-Instruct \
  --served-model-name Qwen2-VL-7B-Instruct \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 32768
```

Server exposes OpenAI-compatible endpoints at `http://<host>:<port>/v1`.

### Test Chat

```bash
export OPENAI_API_KEY="not-needed"
python models/client_chat.py --base-url http://localhost:8000/v1 \
  --model Qwen2-VL-7B-Instruct \
  --prompt "Describe the Eiffel Tower"
```

### Test Vision

```bash
python models/client_vision.py --base-url http://localhost:8000/v1 \
  --model Qwen2-VL-7B-Instruct \
  --image-url "https://example.com/image.jpg" \
  --prompt "What is in this image?"
```

## Notes

- **Multi-GPU**: Increase `--tensor-parallel-size` to shard across GPUs
- **Disk space**: First run downloads model weights to HuggingFace cache
- **Vision**: vLLM enables `--trust-remote-code` automatically for Qwen2-VL
