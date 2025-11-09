import os
import json
from openai import OpenAI
from typing import Dict, Any
from agents.tool_client import ExternalToolClient
from agents.verifier import VerifierAgent
from agents.prompt_builder import PromptBuilder
from utils.common import get_image_base64, get_model_response

class GeneratorAgent:
    def __init__(self, args, verifier: VerifierAgent):
        self.config = args
        self.memory = []
        self.init_plan = None
        self.verifier = verifier
        
        # Initialize chat args
        self.init_chat_args = {}
        if 'gpt' in self.config.get("model"):
            self.init_chat_args['parallel_tool_calls'] = False
            
        # Initialize tool client
        self.tool_client = ExternalToolClient(self.config.get("generator_tools"), self.config)
        
        # Initialize OpenAI client
        client_kwargs = {"api_key": self.config.get("api_key"), 'base_url': self.config.get("api_base_url") or os.getenv("OPENAI_BASE_URL") or 'https://api.openai.com/v1'}
        self.client = OpenAI(**client_kwargs)
        
        # Initialize system prompt
        self.prompt_builder = PromptBuilder(self.client, self.config)
        self.system_prompt = self.prompt_builder.build_prompt("generator", "system")
        self.memory.extend(self.system_prompt)

    async def run(self) -> Dict[str, Any]:
        """
        Generate code based on current memory and optional feedback.
        Now enforces tool calling and returns verifier flag.
        
        Args:
            verifier: Verifier agent instance
            
        Returns:
            Dict containing the generated code, metadata, and verifier flag
        """
        print("\n=== Running generator agent ===\n")
        for i in range(self.config.get("max_rounds")):
            print(f"=== Round {i} ===\n")
            
            # Prepare chat args
            print("Prepare chat args...")
            memory = self.prompt_builder.build_memory(self.memory)
            tool_configs = self.tool_client.tool_configs
            tool_configs = [x for v in tool_configs.values() for x in v]
            chat_args = {"model": self.config.get("model"), "messages": memory, "tools": tool_configs, "tool_choice": "auto", **self.init_chat_args}

            with open(self.config.get("output_dir") + "/_chat_args.json", "w") as f:
                json.dump(chat_args, f, indent=4, ensure_ascii=False)

            # Generate response
            print("Generate response...")
            response = get_model_response(self.client, chat_args)
            message = response.choices[0].message
            
            # Handle tool call
            print("Handle tool call...")
            if not message.tool_calls and not self.config.get("no_tools"):
                if message.content != '':
                    self.memory.append({"role": "assistant", "content": message.content})
                else:
                    self.memory.append({"role": "assistant", "content": "No output"})
                self.memory.append({"role": "user", "content": "Every single output must contain a 'tool_call' field. Your previous message did not contain a 'tool_call' field. Please reconsider."})
                self._save_memory()
                continue
            elif self.config.get("no_tools"):
                content = message.content
                try:
                    if '```json' in content:
                        content = content.split('```json')[1].split('```')[0]
                        content = json.loads(content)
                    tool_name = "execute_and_evaluate"
                    tool_response = await self.tool_client.call_tool("execute_and_evaluate", content)
                    
                    if tool_response.get('require_verifier', False):
                        verifier_result = await self.verifier.run({"argument": content, "execution": tool_response})
                        tool_response['verifier_result'] = verifier_result
                except Exception as e:
                    print(f"Error executing tool: {e}")
                    self.memory.append({"role": "assistant", "content": content})
                    self.memory.append({"role": "user", "content": f"Error executing tool: {e}. Please try again."})
                    self._save_memory()
                    continue
            else:
                tool_call = message.tool_calls[0]
                tool_name = tool_call.function.name
                print(f"Call tool {tool_name}...")
                tool_arguments = json.loads(tool_call.function.arguments)
                
                with open(self.config.get("output_dir") + f"/_tool_call.json", "w") as f:
                    json.dump({'name': tool_name, 'arguments': tool_arguments}, f, indent=4, ensure_ascii=False)
                    
                tool_response = await self.tool_client.call_tool(tool_name, tool_arguments)
                # If the tool is execute_and_evaluate, run the verifier
                if tool_response.get('require_verifier', False):
                    verifier_result = await self.verifier.run({"argument": tool_arguments, "execution": tool_response, "init_plan": self.init_plan})
                    tool_response['verifier_result'] = verifier_result
                    
            # Update and save memory
            print("Update and save memory...")
            self._update_memory({"assistant": message, "user": tool_response})
            self._save_memory()
            
            if tool_name == "end":
                break
        
        print("\n=== Finish generator process ===\n")
    
    def _update_memory(self, message: Dict):
        """Update the memory with the new message"""
        # Add tool calling
        assistant_content = message['assistant'].content
        assistant_tool_calls = message['assistant'].tool_calls[0].model_dump()
        self.memory.append({"role": "assistant", "content": assistant_content, "tool_calls": [assistant_tool_calls]})
        
        # Add tool response
        tool_call_id = message['assistant'].tool_calls[0].id
        tool_call_name = message['assistant'].tool_calls[0].function.name
        tool_response = []
        user_response = []
        
        if 'image' in message['user']:
            for text, image in zip(message['user']['text'], message['user']['image']):
                user_response.append({"type": "text", "text": text})
                user_response.append({"type": "image_url", "image_url": {"url": get_image_base64(image)}})
                user_response.append({"type": "text", "text": f"Image loaded from local path: {image}"})
        else:
            for text in message['user']['text']:
                tool_response.append({"type": "text", "text": text})
        if 'verifier_result' in message['user']:
            tool_response.append({"type": "text", "text": "The following information is what the verifier agent returns to you: (1) Visual difference analysis between the current scene and the target scene (2) Suggested code modifications to follow."})
            for text in message['user']['verifier_result']['text']:
                tool_response.append({"type": "text", "text": text})
        if 'image' in message['user']:
            tool_response.append({"type": "text", "text": "The next user message contains the image result of the tool call."})
        
        if self.config.get("no_tools"):
            self.memory.append({"role": "assistant", "content": tool_response})
        else:
            self.memory.append({"role": "tool", "content": tool_response, "name": tool_call_name, "tool_call_id": tool_call_id})
        if user_response:
            self.memory.append({"role": "user", "content": user_response})
        
        # Add initial plan
        if tool_call_name == "initialize_plan":
            have_plan = False
            self.init_plan = "\n".join(message['user']['plan'])
            for info in self.memory[1]['content']:
                if info['type'] == 'text' and info['text'].startswith('Initial plan:'):
                    info['text'] = f"Initial plan: {self.init_plan}"
                    have_plan = True
                    break
            if not have_plan:
                self.memory[1]['content'].append({"type": "text", "text": f"Initial plan: {self.init_plan}"})

        # Add downloaded assets
        if tool_call_name == "get_better_object":
            try:
                object_name = json.loads(message['assistant'].tool_calls[0].function.arguments)['object_name']
                object_path = message['user']['text'][0].split('downloaded to: ')[1]
                self.memory[1]['content'].append({"type": "text", "text": f"Downloaded {object_name} to {object_path}"})
            except Exception as e:
                print(f"Error adding downloaded assets: {e}")
    
    def _save_memory(self):
        """Save the memory to the file"""
        output_file = self.config.get("output_dir") + "/generator_memory.json"
        with open(output_file, "w") as f:
            json.dump(self.memory, f, indent=4, ensure_ascii=False)
    
    async def cleanup(self) -> None:
        """Clean up external connections."""
        await self.tool_client.cleanup()