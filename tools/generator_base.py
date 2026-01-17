"""Generator Base MCP Server.

Provides the base MCP server for the Generator agent with an 'end' tool
to signal process completion.
"""

from typing import Dict, List

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("generator-base")

# Tool configurations for the agent
tool_configs: List[Dict[str, object]] = [
    {
        "type": "function",
        "function": {
            "name": "end",
            "description": "No-op tool used to indicate the process should end. If the scene has no remaining issues, stop making changes and call this tool.",
        }
    }
]


@mcp.tool()
def initialize(args: Dict[str, object]) -> Dict[str, object]:
    """Initialize the generator base."""
    return {"status": "success", "output": {"text": ["Generator base initialized successfully"], "tool_configs": tool_configs}}


@mcp.tool()
def end() -> Dict[str, object]:
    """Signal that the generation process should end."""
    return {"status": "success", "output": {"text": ["END THE PROCESS"]}}


def main() -> None:
    """Run the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()