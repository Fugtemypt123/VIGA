"""Verifier Base MCP Server.

Provides the base MCP server for the Verifier agent with an 'end' tool
to output visual difference analysis and edit suggestions.
"""

from typing import Dict, List, Optional

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("verifier-base")

# Tool configurations for the agent
tool_configs: List[Dict[str, object]] = [
    {
        "type": "function",
        "function": {
            "name": "end",
            "description": "If you think your observations are sufficient, call the tool to end the process and output the answer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "visual_difference": {
                        "type": "string", 
                        "description": "Visual difference between the current scene and the target scene. You can answer from the following four aspects:\n1) Camera\n• If the generator’s camera choice is poor (occlusions, missing key objects, suboptimal angle), use setup_camera and investigate to find a better viewpoint.\n• Provide the exact observer camera coordinates and orientation as part of your edit_suggestion so the generator can replicate or adapt them.\n2) Objects\n• Verify that all key objects in the target image exist in the current scene.\n• If objects are missing or extraneous, recommend additions or removals and specify whether to generate a new asset or duplicate an existing one.\n• If a present object diverges materially from the target (e.g., target chair is black but current chair is white), recommend replacement or material edits as appropriate.\n3) Layout\n• Check whether spatial layout matches the target.\n• If not, recommend concrete transforms (move/rotate/scale) and, when possible, indicate relative or absolute adjustments (e.g., “move desk 0.3 m toward +Y,” “rotate lamp −15° around Z”).\n4) Environment\n• Assess background, lighting direction/intensity, and overall ambience.\n• If these do not match the target, recommend changes (e.g., environment map, key/fill/rim balance, wall/floor backdrop corrections).\n5) Animation (only required in dynamic mode)\n• Assess whether animated objects exhibit correct motion; verify inter-object interactions (e.g., contact timing, grasp/kick/hold), per-frame physical plausibility, and whether the overall keyframe sequence satisfies the intended action.\n• Use set_key_frame to inspect critical frames; reference exact frame indices from the scene metadata in your suggestions."
                    },
                    "edit_suggestion": {
                        "type": "string", 
                        "description": "Edit suggestion for the current scene. Refer to the visual difference to propose the edit suggestion."
                    },
                    # "image_paths": {
                    #     "type": "array", 
                    #     "description": "Output the observation image paths that you think will be helpful to the generator here. If you think the observation image is not helpful, you do not need to output anything here.",
                    #     "items": {
                    #         "type": "string",
                    #         "description": "The path of the observation image."
                    #     }
                    # },
                },
                "required": ["visual_difference", "edit_suggestion"]
            }
        }
    }
]

@mcp.tool()
def end(
    visual_difference: str,
    edit_suggestion: str,
    image_paths: Optional[str] = None
) -> Dict[str, object]:
    """End verification and output analysis results."""
    result: Dict[str, object] = {
        "status": "success",
        "output": {"text": [f"Visual difference: {visual_difference}\nEdit suggestion: {edit_suggestion}"]}
    }
    if image_paths:
        result['output']['image'] = image_paths
    return result


@mcp.tool()
def initialize(args: Dict[str, object]) -> Dict[str, object]:
    """Initialize the verifier base."""
    return {"status": "success", "output": {"text": ["Verifier base initialized successfully"], "tool_configs": tool_configs}}


def main() -> None:
    """Run the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()