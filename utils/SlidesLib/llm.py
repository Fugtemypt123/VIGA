from openai import OpenAI
import requests

class LLM():
    """Calls the LLM"""
    @classmethod
    def __init_llm__(cls):
        client = OpenAI()
        code_prompt = "Directly Generate executable python code for the following request:\n"
        return client, code_prompt
    @classmethod
    def get_answer(cls, question: str):
        """Calls the LLM by inputing a question, 
        then get the response of the LLM as the answer"""
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
    def get_code(cls, request:str, examples:str = ""):
        """ 
        Calls the LLM to generate code for a request. 
        request: the task that the model should conduct
        examples: few-shot code examples for the request
        """
        client, code_prompt = cls.__init_llm__()
        code = cls.get_answer(code_prompt + examples + request)
        return code