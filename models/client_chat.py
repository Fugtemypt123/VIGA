"""Chat Client for OpenAI-compatible Server.

Command-line client for text-based chat with local OpenAI-compatible servers
such as vLLM or other inference backends.
"""

import argparse
import os

from openai import OpenAI


def main() -> None:
    """Run the chat client.

    Parses command-line arguments and sends a chat completion request
    to the specified OpenAI-compatible server.
    """
    parser = argparse.ArgumentParser(description="Chat with local OpenAI-compatible server")
    parser.add_argument("--base-url", default="http://localhost:8000/v1", type=str)
    parser.add_argument("--model", default="Qwen2-VL-7B-Instruct", type=str)
    parser.add_argument("--prompt", required=True, type=str)
    parser.add_argument("--max-tokens", default=512, type=int)
    parser.add_argument("--temperature", default=0.7, type=float)
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "not-needed")
    client = OpenAI(base_url=args.base_url, api_key=api_key)

    completion = client.chat.completions.create(
        model=args.model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": args.prompt},
        ],
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    print(completion.choices[0].message.content)


if __name__ == "__main__":
    main()
