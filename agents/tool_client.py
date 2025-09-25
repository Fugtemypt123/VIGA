import asyncio
import json
import logging
from typing import Dict, Any
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from contextlib import AsyncExitStack

class ExternalToolClient:
    """Client for connecting to external MCP tool servers (blender/slides/image/scene)."""
    
    def __init__(self):
        self.mcp_sessions = {}  # server_type -> McpSession
        self.connection_timeout = 30  # 30 seconds timeout
    
    async def connect_server(self, server_type: str, server_path: str, api_key: str = None):
        """Connect to the specified MCP server with timeout in a background task."""
        if server_type in self.mcp_sessions:
            return  # Already connected
            
        ready_event = asyncio.Event()
        
        async def mcp_session_runner() -> None:
            try:
                env = {"OPENAI_API_KEY": api_key} if api_key else None
                server_params = StdioServerParameters(
                    command="python",
                    args=[server_path],
                    env = env
                )
                
                exit_stack = AsyncExitStack()
                stdio_transport = await asyncio.wait_for(
                    exit_stack.enter_async_context(stdio_client(server_params)),
                    timeout=self.connection_timeout
                )
                stdio, write = stdio_transport
                session = await asyncio.wait_for(
                    exit_stack.enter_async_context(ClientSession(stdio, write)),
                    timeout=self.connection_timeout
                )
                await asyncio.wait_for(
                    session.initialize(),
                    timeout=self.connection_timeout
                )
                
            except Exception as e:
                print(f"Error during MCP connection setup: {e}")
                raise ConnectionError(f"Failed to connect to {server_type} server: {e}") from e
            finally:
                print(f"Sending {server_type} MCP connection ready event")
                ready_event.set()

            try:
                stop_event = asyncio.Event()
                
                # Store the session
                current_task = asyncio.current_task()
                assert current_task is not None, "Current task should not be None"
                
                self.mcp_sessions[server_type] = McpSession(
                    name=server_type,
                    client=session,
                    task=current_task,
                    stop_event=stop_event,
                )
                
                # List available tools
                response = await asyncio.wait_for(
                    session.list_tools(),
                    timeout=10
                )
                tools = response.tools
                print(f"Connected to {server_type.capitalize()} server with tools: {[tool.name for tool in tools]}")
                
                # Wait for the stop event
                await stop_event.wait()
                
            except asyncio.CancelledError:
                print(f"{server_type} MCP session cancelled")
                raise
            except Exception as e:
                print(f"Error during {server_type} MCP session: {e}")
                raise
            finally:
                print(f"Closing {server_type} MCP session")
                try:
                    await exit_stack.aclose()
                except Exception as e:
                    print(f"Error during {server_type} exit stack close: {e}")
                print(f"{server_type} MCP session closed")

        # Run the session runner in a separate task
        asyncio.create_task(mcp_session_runner())
        print(f"Waiting for {server_type} MCP connection to be ready")
        await ready_event.wait()
        print(f"{server_type} MCP connection is ready")
    
    async def initialize_executor(self, server_type: str, **kwargs) -> Dict:
        """Initialize the executor using external server with timeout."""
        session = self.mcp_sessions.get(server_type)
        if not session:
            raise RuntimeError(f"{server_type.capitalize()} server not connected")
        try:
            result = await asyncio.wait_for(
                session.client.call_tool("initialize_executor", {'args': kwargs}),
                timeout=30
            )
            content = json.loads(result.content[0].text) if result.content else {}
            return content
        except asyncio.TimeoutError:
            raise RuntimeError(f"{server_type.capitalize()} executor initialization timeout after 30s")
        except Exception as e:
            raise RuntimeError(f"{server_type.capitalize()} executor initialization failed: {str(e)}")
        
    async def initialize_investigator(self, server_type: str, **kwargs) -> Dict:
        """Initialize the executor using external server with timeout."""
        session = self.mcp_sessions.get(server_type)
        if not session:
            raise RuntimeError(f"{server_type.capitalize()} server not connected")
        try:
            result = await asyncio.wait_for(
                session.client.call_tool("initialize_investigator", {'args': kwargs}),
                timeout=30
            )
            content = json.loads(result.content[0].text) if result.content else {}
            return content
        except asyncio.TimeoutError:
            raise RuntimeError(f"{server_type.capitalize()} investigator initialization timeout after 30s")
        except Exception as e:
            raise RuntimeError(f"{server_type.capitalize()} investigator initialization failed: {str(e)}")
    
    async def exec_script(self, server_type: str, code: str, round_num: int, **kwargs) -> Dict:
        """Execute script using external server with timeout."""
        session = self.mcp_sessions.get(server_type)
        if not session:
            raise RuntimeError(f"{server_type.capitalize()} server not connected")
        if server_type == "blender":
            tool_name = "exec_script"
            tool_args = {"code": code, "round": round_num}
        elif server_type == "slides":
            tool_name = "exec_pptx"
            tool_args = {"code": code, "round": round_num, "code_save": kwargs.get("code_save")}
        elif server_type == "html":
            tool_name = "exec_html"
            tool_args = {"code": code, "round": round_num}
        else:
            raise ValueError(f"Unknown server_type: {server_type}")
        try:
            result = await asyncio.wait_for(
                session.client.call_tool(tool_name, tool_args),
                timeout=120
            )
            content = json.loads(result.content[0].text) if result.content else {}
            return content
        except asyncio.TimeoutError:
            raise RuntimeError(f"{server_type.capitalize()} script execution timeout after 120s")
        except Exception as e:
            raise RuntimeError(f"{server_type.capitalize()} script execution failed: {str(e)}")
    
    async def call_tool(self, server_type: str, tool_name: str, tool_args: dict, timeout: int = 3600) -> Any:
        """Call a specific tool on the external server with timeout."""
        session = self.mcp_sessions.get(server_type)
        if not session:
            raise RuntimeError(f"{server_type.capitalize()} server not connected")
        try:
            result = await asyncio.wait_for(
                session.client.call_tool(tool_name, tool_args),
                timeout=timeout
            )
            content = json.loads(result.content[0].text) if result.content else {}
            return content
        except asyncio.TimeoutError:
            raise RuntimeError(f"{server_type.capitalize()} tool call timeout after {timeout}s")
        except Exception as e:
            raise RuntimeError(f"{server_type.capitalize()} tool call failed: {str(e)}")
    
    async def cleanup(self):
        """Clean up connections by closing all MCP sessions."""
        for server_type, mcp_session in self.mcp_sessions.items():
            try:
                await mcp_session.close()
            except Exception as e:
                logging.warning(f"Cleanup error for {server_type}: {e}")


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
