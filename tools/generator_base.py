from mcp.server.fastmcp import FastMCP

mcp = FastMCP("generator-base")

# tool_configs for agent (only the function w/ @mcp.tools)
tool_configs = [
    {
        "type": "function",
        "function": {
            "name": "end",
            "description": "No-op tool used to indicate the process should end. If the scene has no remaining issues, stop making changes and call this tool.",
        }
    }
]

@mcp.tool()
def initialize(args: dict) -> dict:
    """
    Initialize the generator base.
    """
    return {"status": "success", "output": {"text": ["Generator base initialized successfully"], "tool_configs": tool_configs}}

@mcp.tool()
def end() -> dict:
    """
    No-op tool used to indicate the process should end.
    """
    return {"status": "success", "output": {"text": ["END THE PROCESS"]}}

def main():
    mcp.run()

if __name__ == "__main__":
    main()