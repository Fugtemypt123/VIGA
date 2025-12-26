import asyncio
import json
import logging
import os
from re import T
from typing import Dict, Any, Optional, List
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from contextlib import AsyncExitStack
from utils.path import path_to_cmd

class ServerHandle:
    def __init__(self, path: str):
        self.path = path
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()
        self.ready = asyncio.Event()
        self.session: ClientSession | None = None
        self.stack: AsyncExitStack | None = None

    async def start(self):
        self._task = asyncio.create_task(self._runner())
        await self.ready.wait()  # 等待初始化完成

    async def _runner(self):
        self.stack = AsyncExitStack()
        async with self.stack:
            # 传递当前环境变量
            env = os.environ.copy()
            params = StdioServerParameters(command=path_to_cmd[self.path], args=[self.path], env=env)
            stdio, write = await self.stack.enter_async_context(stdio_client(params))
            session = await self.stack.enter_async_context(ClientSession(stdio, write))
            await session.initialize()
            self.session = session
            self.ready.set()
            await self._stop.wait()  # 阻塞直到 stop

    async def stop(self):
        self._stop.set()
        if self._task:
            await self._task  # 自然退出，安全清理资源

    async def list_tools(self):
        if not self.session:
            raise RuntimeError(f"Server {self.path} is not started.")
        tools = await self.session.list_tools()
        return [t.name for t in tools.tools]

    async def call_tool(self, tool_name: str, args: dict = None, timeout: int = 3600):
        if not self.session:
            raise RuntimeError(f"Server {self.path} is not started.")
        return await asyncio.wait_for(self.session.call_tool(tool_name, args), timeout=timeout)

class ExternalToolClient:
    """Client for connecting to external MCP tool servers (blender/slides/image/scene)."""
    def __init__(self, tool_servers: str, args: Optional[Dict[str, dict]] = None):
        self.tool_to_server: Dict[str, str] = {}
        self.tool_configs = {}
        self.handles = {}
        self.tool_servers = tool_servers.split(",")
        self.args = args

    async def connect_servers(self) -> List[Any]:
        """Connect to multiple MCP servers given a server_name->script map."""
        self.handles = {p: ServerHandle(p) for p in self.tool_servers}
        await asyncio.gather(*(h.start() for h in self.handles.values()))

        # 建立 tool -> server 映射
        for path, handle in self.handles.items():
            tool_names = await handle.list_tools()
            for t in tool_names:
                self.tool_to_server[t] = path
            print(f"MCP Server {path} connected. Tools: {tool_names}")
            tool_configs = await handle.call_tool("initialize", {"args": self.args})
            tool_configs = json.loads(tool_configs.content[0].text)
            self.tool_configs[path] = tool_configs['output']['tool_configs']
        
    async def call_tool(self, tool_name: str, tool_args: dict = None) -> Any:
        server_path = self.tool_to_server.get(tool_name)
        if not server_path:
            raise RuntimeError(f"Tool {tool_name} not found in any server.")
        handle = self.handles[server_path]
        result = await handle.call_tool(tool_name, tool_args)
        result = json.loads(result.content[0].text)
        return result['output']
    
    async def cleanup(self):
        """Clean up connections by closing all MCP sessions."""
        await asyncio.gather(*(h.stop() for h in self.handles.values()))
        print("All MCP servers stopped.")