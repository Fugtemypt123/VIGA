#!/usr/bin/env python3
"""
Main entry for dual-agent interactive framework (generator/verifier).
Supports 3D (Blender) and 2D (PPTX) modes.
Uses MCP stdio connections instead of HTTP servers.
"""
import argparse
import os
import sys
import shutil
import time
import json
import asyncio
from pathlib import Path
from typing import Optional
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import TextContent

# ========== MCP Session Management ==========

class McpSession:
    """Manages a single MCP session with its own task and cleanup."""
    
    def __init__(self, name: str, client: ClientSession, task: asyncio.Task, stop_event: asyncio.Event):
        self.name = name
        self.client = client
        self.task = task
        self.stop_event = stop_event
        self.initialized = False

    async def close(self) -> None:
        """Close the MCP session by setting stop event and waiting for task completion."""
        print(f"Sending stop event to {self.name}")
        self.stop_event.set()
        print(f"Waiting for task {self.name} to finish")
        await self.task
        print(f"Task {self.name} finished")

# ========== Agent Client Wrappers ==========

class GeneratorAgentClient:
    def __init__(self, script_path: str):
        self.script_path = script_path
        self.mcp_session: Optional[McpSession] = None
        self.initialized = False

    async def connect(self):
        """Connect to the Generator MCP server in a background task."""
        ready_event = asyncio.Event()

        async def mcp_session_runner() -> None:
            try:
                server_params = StdioServerParameters(
                    command="python",
                    args=[self.script_path],
                    env={**os.environ, "PYTHONPATH": os.getcwd()}
                )
                
                exit_stack = AsyncExitStack()
                stdio_transport = await exit_stack.enter_async_context(stdio_client(server_params))
                stdio, write = stdio_transport
                client = await exit_stack.enter_async_context(ClientSession(stdio, write))
                
                # Initialize the session
                await client.initialize()
                
            except Exception as e:
                print(f"Error during MCP connection setup: {e}")
                raise ConnectionError(f"Failed to connect to Generator MCP server: {e}") from e
            finally:
                print("Sending Generator MCP connection ready event")
                ready_event.set()

            try:
                stop_event = asyncio.Event()
                
                # Store the session
                current_task = asyncio.current_task()
                assert current_task is not None, "Current task should not be None"
                
                self.mcp_session = McpSession(
                    name="generator",
                    client=client,
                    task=current_task,
                    stop_event=stop_event,
                )
                
                print(f"Connected to Generator: {self.script_path}")
                
                # Wait for the stop event
                await stop_event.wait()
                
            except asyncio.CancelledError:
                print("Generator MCP session cancelled")
                raise
            except Exception as e:
                print(f"Error during Generator MCP session: {e}")
                raise
            finally:
                print("Closing Generator MCP session")
                try:
                    await exit_stack.aclose()
                except Exception as e:
                    print(f"Error during Generator exit stack close: {e}")
                print("Generator MCP session closed")

        # Run the session runner in a separate task
        asyncio.create_task(mcp_session_runner())
        print("Waiting for Generator MCP connection to be ready")
        await ready_event.wait()
        print("Generator MCP connection is ready")

    async def create_session(self, **kwargs):
        """Initialize the generator with session parameters."""
        if not self.mcp_session:
            raise RuntimeError("Not connected. Call connect() first.")
        
        result = await self.mcp_session.client.call_tool("initialize_generator", {'args': kwargs})
        if result.content and len(result.content) > 0:
            content = result.content[0].text
            if '"status": "success"' in content or '"status":"success"' in content:
                self.initialized = True
                return "success"
        raise RuntimeError(f"Failed to create session: {result.content}")

    async def call(self, no_memory: bool = False):
        """Generate code, optionally with feedback."""
        if not self.initialized:
            raise RuntimeError("Generator not initialized. Call create_session() first.")
        
        params = {}
        if no_memory:
            params["no_memory"] = no_memory
        
        result = await self.mcp_session.client.call_tool("call", params)
        if result.content and len(result.content) > 0:
            content = result.content[0].text
            return json.loads(content)
        raise RuntimeError(f"Failed to generate code: {result.content}")

    async def add_feedback(self, feedback: str):
        """Add feedback to the generator."""
        if not self.initialized:
            raise RuntimeError("Generator not initialized. Call create_session() first.")
        
        result = await self.mcp_session.client.call_tool("add_feedback", {"feedback": feedback})
        if not (result.content and len(result.content) > 0 and 
                ('"status": "success"' in result.content[0].text or '"status":"success"' in result.content[0].text)):
            raise RuntimeError(f"Failed to add feedback: {result.content}")

    async def save_thought_process(self):
        """Save the thought process."""
        if not self.initialized:
            raise RuntimeError("Generator not initialized. Call create_session() first.")
        
        result = await self.mcp_session.client.call_tool("save_thought_process", {})
        if not (result.content and len(result.content) > 0 and 
                ('"status": "success"' in result.content[0].text or '"status":"success"' in result.content[0].text)):
            raise RuntimeError(f"Failed to save thought process: {result.content}")

    async def cleanup(self):
        """Clean up resources."""
        if self.mcp_session:
            try:
                # Call cleanup tool first if initialized
                if self.initialized:
                    try:
                        await self.mcp_session.client.call_tool("cleanup_generator", {})
                        print("Generator cleanup tool called successfully")
                    except Exception as e:
                        print(f"Warning: Generator cleanup tool failed: {e}")
                
                # Then cleanup MCP session
                await self.mcp_session.close()
            except Exception as e:
                print(f"Warning: Generator cleanup error: {e}")


class VerifierAgentClient:
    def __init__(self, script_path: str):
        self.script_path = script_path
        self.mcp_session: Optional[McpSession] = None
        self.initialized = False

    async def connect(self):
        """Connect to the Verifier MCP server in a background task."""
        ready_event = asyncio.Event()

        async def mcp_session_runner() -> None:
            try:
                server_params = StdioServerParameters(
                    command="python",
                    args=[self.script_path],
                    env={**os.environ, "PYTHONPATH": os.getcwd()}
                )
                
                exit_stack = AsyncExitStack()
                stdio_transport = await exit_stack.enter_async_context(stdio_client(server_params))
                stdio, write = stdio_transport
                client = await exit_stack.enter_async_context(ClientSession(stdio, write))
                
                # Initialize the session
                await client.initialize()
                
            except Exception as e:
                print(f"Error during MCP connection setup: {e}")
                raise ConnectionError(f"Failed to connect to Verifier MCP server: {e}") from e
            finally:
                print("Sending Verifier MCP connection ready event")
                ready_event.set()

            try:
                stop_event = asyncio.Event()
                
                # Store the session
                current_task = asyncio.current_task()
                assert current_task is not None, "Current task should not be None"
                
                self.mcp_session = McpSession(
                    name="verifier",
                    client=client,
                    task=current_task,
                    stop_event=stop_event,
                )
                
                print(f"Connected to Verifier: {self.script_path}")
                
                # Wait for the stop event
                await stop_event.wait()
                
            except asyncio.CancelledError:
                print("Verifier MCP session cancelled")
                raise
            except Exception as e:
                print(f"Error during Verifier MCP session: {e}")
                raise
            finally:
                print("Closing Verifier MCP session")
                try:
                    await exit_stack.aclose()
                except Exception as e:
                    print(f"Error during Verifier exit stack close: {e}")
                print("Verifier MCP session closed")

        # Run the session runner in a separate task
        asyncio.create_task(mcp_session_runner())
        print("Waiting for Verifier MCP connection to be ready")
        await ready_event.wait()
        print("Verifier MCP connection is ready")

    async def create_session(self, **kwargs):
        """Create a verification session with automatic tool server connection."""
        if not self.mcp_session:
            raise RuntimeError("Not connected. Call connect() first.")
        
        result = await self.mcp_session.client.call_tool("initialize_verifier", {'args': kwargs})
        if result.content and len(result.content) > 0:
            content = result.content[0].text
            try:
                json_data = json.loads(content)
                if json_data.get("status") == "success":
                    self.initialized = True
                    return "success"
            except json.JSONDecodeError:
                pass
        raise RuntimeError(f"Failed to create session: {result.content}")

    async def call(self, code: str, render_path: str, round_num: int):
        """Verify the scene with given parameters."""
        if not self.initialized:
            raise RuntimeError("Verifier not initialized. Call create_session() first.")
        
        result = await self.mcp_session.client.call_tool("call", {
            "code": code,
            "render_path": render_path,
            "round_num": round_num
        })
        if result.content and len(result.content) > 0:
            content = result.content[0].text
            print("verifier result: ", content)
            return json.loads(content)
        raise RuntimeError(f"Failed to verify scene: {result.content}")

    async def save_thought_process(self):
        """Save the thought process."""
        if not self.initialized:
            raise RuntimeError("Verifier not initialized. Call create_session() first.")
        
        result = await self.mcp_session.client.call_tool("save_thought_process", {})
        if not (result.content and len(result.content) > 0 and 
                ('"status": "success"' in result.content[0].text or '"status":"success"' in result.content[0].text)):
            raise RuntimeError(f"Failed to save thought process: {result.content}")

    async def cleanup(self):
        """Clean up resources."""
        if self.mcp_session:
            try:
                # Call cleanup tool first if initialized
                if self.initialized:
                    try:
                        await self.mcp_session.client.call_tool("cleanup_verifier", {})
                        print("Verifier cleanup tool called successfully")
                    except Exception as e:
                        print(f"Warning: Verifier cleanup tool failed: {e}")
                
                # Then cleanup MCP session
                await self.mcp_session.close()
            except Exception as e:
                print(f"Warning: Verifier cleanup error: {e}")