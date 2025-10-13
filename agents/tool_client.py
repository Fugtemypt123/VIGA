import asyncio
import json
import logging
from re import T
from typing import Dict, Any, Optional, List
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from contextlib import AsyncExitStack

class ExternalToolClient:
    """Client for connecting to external MCP tool servers (blender/slides/image/scene)."""
    def __init__(self, tool_servers: Dict[str, str], args: Optional[Dict[str, dict]] = None):
        self.mcp_sessions = {}  # server_name -> McpSession
        self.tool_to_server: Dict[str, str] = {}
        self.tool_configs = {}
        self.tool_servers = tool_servers.split(",")
        self.args = args
    
    async def connect_server(path: str):
        async with AsyncExitStack() as stack:
            server_params = StdioServerParameters(command="python", args=[path])
            stdio, write = await stack.enter_async_context(stdio_client(server_params))
            session = await stack.enter_async_context(ClientSession(stdio, write))
            await session.initialize()
            print(f"Connected to MCP server: {path}")
            return session  # 可以存起来调用工具

    async def connect_servers(self) -> List[Any]:
        """Connect to multiple MCP servers given a server_name->script map."""
        for path in self.tool_servers:
            session = await self.connect_server(path=path)
            tools = session.get_tools()
            for tool in tools:
                self.tool_to_server[tool.name] = path
                self.mcp_sessions[path] = session
            self.tool_configs[path] = tools
            session.call_tool(tool_name="initialize", tool_args={"args": self.args})
        
    async def call_tool(self, tool_name: str, tool_args: dict = None) -> Any:
        """Call a specific tool by name with timeout. Server is inferred from known mappings."""
        server_name = self.tool_to_server.get(tool_name)
        session = self.mcp_sessions.get(server_name)
        if not session:
            raise RuntimeError(f"Server '{server_name}' for tool '{tool_name}' not connected")
        try:
            call_tool_result = await session.call_tool(tool_name, tool_args)
            result = json.loads(call_tool_result.content[0].text)
            return result['output']
        except Exception as e:
            raise RuntimeError(f"Tool '{tool_name}' call failed: {str(e)}")
    
    async def cleanup(self):
        """Clean up connections by closing all MCP sessions."""
        for server_name, mcp_session in self.mcp_sessions.items():
            try:
                await mcp_session.close()
            except Exception as e:
                logging.warning(f"Cleanup error for {server_name}: {e}")