"""Verifier Agent for analyzing generated scenes in the VIGA system.

The Verifier Agent analyzes rendered scenes from multiple viewpoints,
comparing them against target images and providing structured feedback
for the Generator Agent to refine its output.
"""

import json
import os
from typing import Any, Dict, List, Optional

from openai import OpenAI

from agents.prompt_builder import PromptBuilder
from agents.tool_client import ExternalToolClient
from utils.common import get_image_base64, get_model_response


class VerifierAgent:
    """Agent responsible for analyzing and verifying generated scenes.

    The Verifier Agent examines rendered outputs from the Generator,
    comparing them against targets using camera manipulation tools
    and providing actionable feedback for refinement.

    Attributes:
        config: Configuration dictionary containing model settings and paths.
        memory: Working conversation memory (may be cleared between rounds).
        saved_memory: Persistent conversation memory for logging.
        tool_client: Client for calling external MCP tools.
    """

    def __init__(self, args: Dict[str, Any]) -> None:
        """Initialize the Verifier Agent.

        Args:
            args: Configuration dictionary with keys like 'model', 'api_key',
                  'verifier_tools', 'max_rounds', 'clear_memory', etc.
        """
        self.config = args
        self.memory: List[Dict[str, Any]] = []
        self.saved_memory: List[Dict[str, Any]] = []

        # Initialize chat args
        self.init_chat_args: Dict[str, Any] = {}
        if 'gpt' in self.config.get("model") and not self.config.get("no_tools"):
            self.init_chat_args['parallel_tool_calls'] = False

        # Initialize tool client
        self.tool_client = ExternalToolClient(self.config.get("verifier_tools"), self.config)

        # Initialize OpenAI client
        client_kwargs = {
            "api_key": self.config.get("api_key"),
            "base_url": self.config.get("api_base_url") or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        }
        self.client = OpenAI(**client_kwargs)

        # Initialize system prompt
        self.prompt_builder = PromptBuilder(self.client, self.config)
        self.system_prompt = self.prompt_builder.build_prompt("verifier", "system")
        self.memory.extend(self.system_prompt)
        self.saved_memory.extend(self.system_prompt)
        
    async def run(self, user_message: Dict[str, Any]) -> Dict[str, Any]:
        """Verify the generated scene and provide feedback.

        Analyzes the rendered output using chain-of-thought reasoning and
        camera manipulation tools to compare against the target.

        Args:
            user_message: Dictionary containing 'argument' (generator's inputs),
                          'execution' (tool execution results), and optionally 'init_plan'.

        Returns:
            Dictionary with 'text' containing feedback for the generator.
        """
        print("\n=== Running verifier agent ===\n")
        
        # Reload scene if needed
        if "reload_scene" in self.tool_client.tool_to_server:
            print("Reload scene...")
            await self.tool_client.call_tool("reload_scene", {})
        print("Build user message...")
        user_message = self.prompt_builder.build_prompt("verifier", "user", user_message)
        if self.config.get("clear_memory"):
            print("Clear memory...")
            self.memory = self.system_prompt + user_message
        else:
            print("Extend memory...")
            self.memory.extend(user_message)
        self.saved_memory.extend(user_message)
        print("Save memory...")
        self._save_memory()
        result = None
        
        for i in range(self.config.get("max_rounds")):
            print(f"=== Round {i} ===\n")
            
            # Prepare chat args
            print("Prepare chat args...")
            memory = self.prompt_builder.build_memory(self.memory)
            tool_configs = self.tool_client.tool_configs
            tool_configs = [x for v in tool_configs.values() for x in v]
            if self.config.get("no_tools"):
                chat_args = {"model": self.config.get("model"), "messages": memory, **self.init_chat_args}
            else:
                chat_args = {"model": self.config.get("model"), "messages": memory, "tools": tool_configs, "tool_choice": "auto", **self.init_chat_args}
            
            # Generate response
            print("Generate response...")
            response = get_model_response(self.client, chat_args, 1)
            message = response[0].choices[0].message
            
            # Handle tool call
            print("Handle tool call...")
            if not message.tool_calls and not self.config.get("no_tools"):
                self.memory.append({"role": "assistant", "content": message.content})
                self.memory.append({"role": "user", "content": "Each return message must contain a tool call. Your previous message did not contain a tool call. Please reconsider."})
                self.saved_memory.append({"role": "assistant", "content": message.content})
                self.saved_memory.append({"role": "user", "content": "Each return message must contain a tool call. Your previous message did not contain a tool call. Please reconsider."})
                self._save_memory()
                continue
            elif self.config.get("no_tools"):
                content = message.content
                try:
                    json_content = content.split('```json')[1].split('```')[0]
                    json_content = json.loads(json_content)
                    json_content = {'visual_difference': str(json_content.get('visual_difference', '')), 'edit_suggestion': str(json_content.get('edit_suggestion', ''))}
                    tool_name = "end"
                    tool_response = await self.tool_client.call_tool("end", json_content)
                except Exception as e:
                    print(f"Error executing tool: {e}")
                    self.memory.append({"role": "assistant", "content": content})
                    self.memory.append({"role": "user", "content": f"Error executing tool: {e}. Please try again."})
                    self.saved_memory.append({"role": "assistant", "content": content})
                    self.saved_memory.append({"role": "user", "content": f"Error executing tool: {e}. Please try again."})
                    self._save_memory()
                    continue
            else:
                tool_call = message.tool_calls[0]
                tool_name = tool_call.function.name
                print(f"Call tool {tool_name}...")
                tool_response = await self.tool_client.call_tool(tool_name, json.loads(tool_call.function.arguments))
                
            # Update and save memory
            print("Update and save memory...")
            self._update_memory({"assistant": message, "user": tool_response})
            self._save_memory()

            if tool_name == "end":
                result = tool_response
                break
            
        print("\n=== Finish verification process ===\n")
        
        if result:
            return result
        else:
            return {"text": ["No valid response, please observe the output image and adjust your code accordingly."]}
    
    def _update_memory(self, message: Dict[str, Any]) -> None:
        """Update both working and saved memory with assistant and tool messages.

        Args:
            message: Dictionary containing 'assistant' (the model response) and
                     'user' (the tool response with text and optional images).
        """
        # Add tool calling
        assistant_content = message['assistant'].content
        if not self.config.get("no_tools"):
            assistant_tool_calls = message['assistant'].tool_calls[0].model_dump()
            self.memory.append({"role": "assistant", "content": assistant_content, "tool_calls": [assistant_tool_calls]})
            self.saved_memory.append({"role": "assistant", "content": assistant_content, "tool_calls": [assistant_tool_calls]})
        else:
            self.memory.append({"role": "assistant", "content": assistant_content})
            self.saved_memory.append({"role": "assistant", "content": assistant_content})
        
        # Add tool response
        if not self.config.get("no_tools"):
            tool_call_id = message['assistant'].tool_calls[0].id
            tool_call_name = message['assistant'].tool_calls[0].function.name
        else:
            tool_call_id = ''
            tool_call_name = ''
        tool_response = []
        user_response = []
        if 'image' in message['user']:
            tool_response.append({"type": "text", "text": "The next user message contains the image result of the tool call."})
            for text, image in zip(message['user']['text'], message['user']['image']):
                user_response.append({"type": "text", "text": text})
                user_response.append({"type": "image_url", "image_url": {"url": get_image_base64(image)}})
                user_response.append({"type": "text", "text": f"Image loaded from local path: {image}"})
        else:
            for text in message['user']['text']:
                tool_response.append({"type": "text", "text": text}) 
        
        if self.config.get("no_tools"):
            self.memory.append({"role": "assistant", "content": tool_response})
            self.saved_memory.append({"role": "assistant", "content": tool_response})
        else:
            self.memory.append({"role": "tool", "content": tool_response, "name": tool_call_name, "tool_call_id": tool_call_id})
            self.saved_memory.append({"role": "tool", "content": tool_response, "name": tool_call_name, "tool_call_id": tool_call_id})
        if user_response:
            self.memory.append({"role": "user", "content": user_response})
            self.saved_memory.append({"role": "user", "content": user_response})

    def _save_memory(self) -> None:
        """Save the persistent memory to a JSON file in the output directory."""
        output_file = self.config.get("output_dir") + "/verifier_memory.json"
        with open(output_file, "w") as f:
            json.dump(self.saved_memory, f, indent=4, ensure_ascii=False)

    async def cleanup(self) -> None:
        """Clean up external connections."""
        await self.tool_client.cleanup()