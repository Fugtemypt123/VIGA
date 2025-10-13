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
        self.stack = AsyncExitStack()
        self.mcp_sessions = {}  # server_name -> McpSession
        self.tool_to_server: Dict[str, str] = {}
        self.tool_configs = {}
        self.tool_servers = tool_servers.split(",")
        self.args = args
    
    async def connect_server(self, path: str):
        server_params = StdioServerParameters(command="python", args=[path])
        stdio, write = await self.stack.enter_async_context(stdio_client(server_params))
        session = await self.stack.enter_async_context(ClientSession(stdio, write))
        await session.initialize()
        print(f"Connected to MCP server: {path}")
        return session

    async def connect_servers(self) -> List[Any]:
        """Connect to multiple MCP servers given a server_name->script map."""
        for path in self.tool_servers:
            print(f"Connecting to MCP server: {path}")
            session = await self.connect_server(path)
            tool_list = await session.list_tools()
            for tool in tool_list.tools:
                print(f"Adding tool: {tool.name} to server")
                self.tool_to_server[tool.name] = path
                self.mcp_sessions[path] = session
            tool_configs = await session.call_tool(name="initialize", arguments={"args": self.args})
            tool_configs = json.loads(tool_configs.content[0].text)
            self.tool_configs[path] = tool_configs['output']['tool_configs']
        
    async def call_tool(self, tool_name: str, tool_args: dict = None) -> Any:
        """Call a specific tool by name with timeout. Server is inferred from known mappings."""
        server_name = self.tool_to_server.get(tool_name)
        session = self.mcp_sessions.get(server_name)
        if not session:
            raise RuntimeError(f"Server '{server_name}' for tool '{tool_name}' not connected")
        try:
            call_tool_result = await session.call_tool(name=tool_name, arguments=tool_args)
            result = json.loads(call_tool_result.content[0].text)
            return result['output']
        except Exception as e:
            raise RuntimeError(f"Tool '{tool_name}' call failed: {str(e)}")
    
    async def cleanup(self):
        """Clean up connections by closing all MCP sessions."""
        await self.stack.aclose()