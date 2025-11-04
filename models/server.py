import argparse
import os
import shlex
import subprocess
import sys


def build_command(
    host: str,
    port: int,
    model: str,
    served_model_name: str,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    max_model_len: int,
    additional_args: str,
):
    base_cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--host",
        host,
        "--port",
        str(port),
        "--model",
        model,
        "--served-model-name",
        served_model_name,
        "--trust-remote-code",
        "--tensor-parallel-size",
        str(tensor_parallel_size),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--max-model-len",
        str(max_model_len),
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "hermes"
    ]

    if additional_args:
        base_cmd.extend(shlex.split(additional_args))

    return base_cmd


def main():
    parser = argparse.ArgumentParser(
        description="Launch a local OpenAI-compatible server for Qwen2-VL-7B-Instruct via vLLM"
    )
    parser.add_argument("--host", default="0.0.0.0", type=str)
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument(
        "--model", default="Qwen/Qwen2-VL-7B-Instruct", type=str,
        help="HF model id or local path"
    )
    parser.add_argument(
        "--served-model-name", default="Qwen2-VL-7B-Instruct", type=str,
        help="Name exposed to OpenAI clients"
    )
    parser.add_argument("--tensor-parallel-size", default=1, type=int)
    parser.add_argument("--gpu-memory-utilization", default=0.90, type=float)
    parser.add_argument("--max-model-len", default=32768, type=int)
    parser.add_argument(
        "--additional-args",
        default="",
        type=str,
        help="Extra args passed through to vLLM server (e.g., --quantization awq)",
    )

    args = parser.parse_args()

    # Allow users to set HF token if needed
    if "HF_TOKEN" in os.environ:
        os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]

    cmd = build_command(
        host=args.host,
        port=args.port,
        model=args.model,
        served_model_name=args["served_model_name"] if isinstance(args, dict) else args.served_model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        additional_args=args.additional_args,
    )

    print("Launching vLLM OpenAI server:")
    print(" ", " ".join(shlex.quote(token) for token in cmd))
    print("Base URL: http://%s:%d/v1" % (args.host, args.port))

    # Stream subprocess output
    process = subprocess.Popen(cmd)
    try:
        process.wait()
    except KeyboardInterrupt:
        process.terminate()
        process.wait()


if __name__ == "__main__":
    main()


