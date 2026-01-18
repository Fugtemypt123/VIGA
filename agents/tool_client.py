"""MCP Tool Client for connecting to external tool servers.

This module provides async clients for Model Context Protocol (MCP) servers,
enabling the agent system to call external tools like Blender, slides generator, etc.
"""

import asyncio
import json
import logging
import os
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from utils.path import path_to_cmd

class ServerHandle:
    """Async wrapper for a single MCP server connection via stdio.

    Manages the lifecycle of an MCP server process, including startup,
    tool discovery, tool execution, and graceful shutdown.

    Attributes:
        path: Path to the MCP server script.
        session: Active MCP client session once connected.
        ready: Event signaling when the server is ready for requests.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()
        self.ready = asyncio.Event()
        self.session: ClientSession | None = None
        self.stack: AsyncExitStack | None = None

    async def start(self) -> None:
        """Start the MCP server and wait for it to be ready."""
        self._task = asyncio.create_task(self._runner())
        await self.ready.wait()

    async def _runner(self) -> None:
        """Internal runner that maintains the server connection."""
        self.stack = AsyncExitStack()
        async with self.stack:
            env = os.environ.copy()
            params = StdioServerParameters(command=path_to_cmd[self.path], args=[self.path], env=env)
            stdio, write = await self.stack.enter_async_context(stdio_client(params))
            session = await self.stack.enter_async_context(ClientSession(stdio, write))
            await session.initialize()
            self.session = session
            self.ready.set()
            await self._stop.wait()

    async def stop(self) -> None:
        """Stop the server and clean up resources."""
        self._stop.set()
        if self._task:
            await self._task

    async def list_tools(self) -> List[str]:
        """List all available tools from this server.

        Returns:
            List of tool names available on this server.

        Raises:
            RuntimeError: If the server is not started.
        """
        if not self.session:
            raise RuntimeError(f"Server {self.path} is not started.")
        tools = await self.session.list_tools()
        return [t.name for t in tools.tools]

    async def call_tool(
        self,
        tool_name: str,
        args: Optional[Dict[str, Any]] = None,
        timeout: int = 3600
    ) -> Any:
        """Call a tool on this server.

        Args:
            tool_name: Name of the tool to call.
            args: Arguments to pass to the tool.
            timeout: Maximum time to wait for the tool to complete (seconds).

        Returns:
            The tool's response.

        Raises:
            RuntimeError: If the server is not started.
            asyncio.TimeoutError: If the tool call exceeds the timeout.
        """
        if not self.session:
            raise RuntimeError(f"Server {self.path} is not started.")
        return await asyncio.wait_for(self.session.call_tool(tool_name, args), timeout=timeout)

class ExternalToolClient:
    """Client for connecting to multiple external MCP tool servers.

    Orchestrates connections to multiple MCP servers (e.g., Blender, slides, image processing)
    and provides a unified interface for tool discovery and execution.

    Attributes:
        tool_to_server: Mapping from tool name to server path.
        tool_configs: Tool configurations indexed by server path.
        handles: Active ServerHandle instances indexed by server path.
        tool_servers: List of server script paths to connect to.
    """

    def __init__(self, tool_servers: str, args: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the external tool client.

        Args:
            tool_servers: Comma-separated list of MCP server script paths.
            args: Configuration arguments to pass to each server during initialization.
        """
        self.tool_to_server: Dict[str, str] = {}
        self.tool_configs: Dict[str, List[Dict[str, Any]]] = {}
        self.handles: Dict[str, ServerHandle] = {}
        self.tool_servers = tool_servers.split(",")
        self.args = args

    async def connect_servers(self) -> None:
        """Connect to all configured MCP servers and discover their tools."""
        self.handles = {p: ServerHandle(p) for p in self.tool_servers}
        await asyncio.gather(*(h.start() for h in self.handles.values()))

        # Build tool -> server mapping
        for path, handle in self.handles.items():
            tool_names = await handle.list_tools()
            for tool_name in tool_names:
                self.tool_to_server[tool_name] = path
            print(f"MCP Server {path} connected. Tools: {tool_names}")
            tool_configs = await handle.call_tool("initialize", {"args": self.args})
            tool_configs = json.loads(tool_configs.content[0].text)
            self.tool_configs[path] = tool_configs['output']['tool_configs']

    async def call_tool(
        self,
        tool_name: str,
        tool_args: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Call a tool by name, routing to the appropriate server.

        Args:
            tool_name: Name of the tool to call.
            tool_args: Arguments to pass to the tool.

        Returns:
            The tool's output response.

        Raises:
            RuntimeError: If the tool is not found in any connected server.
        """
        server_path = self.tool_to_server.get(tool_name)
        if not server_path:
            raise RuntimeError(f"Tool {tool_name} not found in any server.")
        handle = self.handles[server_path]
        result = await handle.call_tool(tool_name, tool_args)
        result = json.loads(result.content[0].text)
        return result['output']

    async def cleanup(self) -> None:
        """Clean up connections by stopping all MCP servers."""
        await asyncio.gather(*(h.stop() for h in self.handles.values()))
        print("All MCP servers stopped.")