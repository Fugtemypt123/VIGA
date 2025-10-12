import asyncio
import json
import logging
from typing import Dict, Any, Optional, List
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from contextlib import AsyncExitStack

class McpSession:
    """Manages a single MCP session with its own task and cleanup."""
    def __init__(self, name: str, client: ClientSession, task: asyncio.Task, stop_event: asyncio.Event):
        self.name = name
        self.client = client
        self.task = task
        self.stop_event = stop_event

    async def close(self) -> None:
        """Close the MCP session by setting stop event and waiting for task completion."""
        print(f"Sending stop event to {self.name}")
        self.stop_event.set()
        print(f"Waiting for task {self.name} to finish")
        await self.task
        print(f"Task {self.name} finished")

class ExternalToolClient:
    """Client for connecting to external MCP tool servers (blender/slides/image/scene)."""
    def __init__(self, tool_servers: Dict[str, str], init_args: Optional[Dict[str, dict]] = None):
        self.mcp_sessions = {}  # server_name -> McpSession
        self.connection_timeout = 30  # 30 seconds timeout
        self.tool_to_server: Dict[str, str] = {}
        self.tool_configs = []
        self.tool_servers = tool_servers
        self.init_args = init_args
    
    async def connect_server(self, server_name: str, server_path: str):
        """Connect to the specified MCP server with timeout in a background task.
        Multiple servers can be connected; indexed by server_name key.
        """
        print(f"Connecting to {server_name} server at {server_path}")
        ready_event = asyncio.Event()
        
        async def mcp_session_runner() -> None:
            try:
                server_params = StdioServerParameters(command="python", args=[server_path])
                exit_stack = AsyncExitStack()
                stdio_transport = await asyncio.wait_for(exit_stack.enter_async_context(stdio_client(server_params)), timeout=self.connection_timeout)
                stdio, write = stdio_transport
                session = await asyncio.wait_for(exit_stack.enter_async_context(ClientSession(stdio, write)), timeout=self.connection_timeout)
                await asyncio.wait_for(session.initialize(), timeout=self.connection_timeout)
                
            except Exception as e:
                print(f"Error during MCP connection setup: {e}")
                raise ConnectionError(f"Failed to connect to {server_name} server: {e}") from e
            finally:
                print(f"Sending {server_name} MCP connection ready event")
                ready_event.set()

            try:
                stop_event = asyncio.Event()
                # Store the session
                current_task = asyncio.current_task()
                assert current_task is not None, "Current task should not be None"
                
                self.mcp_sessions[server_name] = McpSession(name=server_name, client=session, task=current_task, stop_event=stop_event)
                
                # List available tools
                response = await asyncio.wait_for(session.list_tools(), timeout=10)
                tools = response.tools
                # Record tool -> server mapping
                for tool in tools:
                    if tool and getattr(tool, "name", None) and tool.name != "initialize":
                        self.tool_to_server[tool.name] = server_name

                # Auto-initialize once per server if an 'initialize' tool exists
                try:
                    tool_names = {t.name for t in tools if getattr(t, "name", None)}
                    if "initialize" in tool_names:
                        args = (self.init_args or {}).get(server_name) or {}
                        tool_configs = await asyncio.wait_for(session.call_tool("initialize", {"args": args} if isinstance(args, dict) else args), timeout=30)
                    self.tool_configs.extend(tool_configs)
                except Exception as e:
                    print(f"Warning: initialize failed for {server_name}: {e}")
                
                # Wait for the stop event
                await stop_event.wait()
                
            except Exception as e:
                print(f"Error during {server_name} MCP session: {e}")
                raise
            finally:
                print(f"Closing {server_name} MCP session")
                try:
                    await exit_stack.aclose()
                except Exception as e:
                    print(f"Error during {server_name} exit stack close: {e}")
                print(f"{server_name} MCP session closed")

        # Run the session runner in a separate task
        asyncio.create_task(mcp_session_runner())
        await ready_event.wait()

    async def connect_servers(self) -> List[Any]:
        """Connect to multiple MCP servers given a server_name->script map."""
        if not self.tool_servers:
            return
        # Launch connections concurrently
        await asyncio.gather(*[
            self.connect_server(server_name=stype, server_path=spath)
            for stype, spath in self.tool_servers.items()
        ])
    
    async def call_tool(self, tool_name: str, tool_args: dict = None, timeout: int = 3600, **kwargs) -> Any:
        """Call a specific tool by name with timeout. Server is inferred from known mappings."""
        server_name = self.tool_to_server.get(tool_name)
        if not server_name:
            available = ", ".join(sorted(self.tool_to_server.keys()))
            raise RuntimeError(f"No server mapping for tool '{tool_name}'. Known tools: [{available}]")
        session = self.mcp_sessions.get(server_name)
        if not session:
            raise RuntimeError(f"Server '{server_name}' for tool '{tool_name}' not connected")
        
        try:
            result = await asyncio.wait_for(session.client.call_tool(tool_name, tool_args or {}), timeout=timeout)
            return result
        except asyncio.TimeoutError:
            raise RuntimeError(f"Tool '{tool_name}' call timeout after {timeout}s")
        except Exception as e:
            raise RuntimeError(f"Tool '{tool_name}' call failed: {str(e)}")
    
    async def cleanup(self):
        """Clean up connections by closing all MCP sessions."""
        for server_name, mcp_session in self.mcp_sessions.items():
            try:
                await mcp_session.close()
            except Exception as e:
                logging.warning(f"Cleanup error for {server_name}: {e}")