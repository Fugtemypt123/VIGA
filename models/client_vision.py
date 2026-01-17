"""Vision Chat Client for OpenAI-compatible Server.

Command-line client for vision-based chat with local OpenAI-compatible servers
such as vLLM or other inference backends that support multimodal inputs.
"""

import argparse
import os

from openai import OpenAI


def main() -> None:
    """Run the vision chat client.

    Parses command-line arguments and sends a vision chat completion request
    to the specified OpenAI-compatible server with an image URL.
    """
    parser = argparse.ArgumentParser(description="Vision chat with local OpenAI-compatible server")
    parser.add_argument("--base-url", default="http://localhost:8000/v1", type=str)
    parser.add_argument("--model", default="Qwen2-VL-7B-Instruct", type=str)
    parser.add_argument("--image-url", required=True, type=str)
    parser.add_argument("--prompt", default="Describe the image.", type=str)
    parser.add_argument("--max-tokens", default=512, type=int)
    parser.add_argument("--temperature", default=0.2, type=float)
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "not-needed")
    client = OpenAI(base_url=args.base_url, api_key=api_key)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": args.prompt},
                {"type": "image_url", "image_url": {"url": args.image_url}},
            ],
        }
    ]

    completion = client.chat.completions.create(
        model=args.model,
        messages=messages,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    print(completion.choices[0].message.content)


if __name__ == "__main__":
    main()
