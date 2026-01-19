"""LLM wrapper utilities for GPT-4o-mini."""
from typing import Tuple

from openai import OpenAI


class LLM:
    """LLM wrapper class for chat completions."""

    @classmethod
    def __init_llm__(cls) -> Tuple[OpenAI, str]:
        """Initialize OpenAI client and code prompt."""
        client = OpenAI()
        code_prompt = "Directly Generate executable python code for the following request:\n"
        return client, code_prompt

    @classmethod
    def get_answer(cls, question: str) -> str:
        """Get answer from LLM for a question."""
        client, code_prompt = cls.__init_llm__()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question}
            ]
        )
        return response.choices[0].message.content

    @classmethod
    def get_code(cls, request: str, examples: str = "") -> str:
        """Generate code for a request using LLM."""
        client, code_prompt = cls.__init_llm__()
        code = cls.get_answer(code_prompt + examples + request)
        return code