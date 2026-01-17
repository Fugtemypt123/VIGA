"""Initialize Plan MCP Server.

Provides tools for creating and storing detailed scene plans that guide
the Generator agent's subsequent actions. Supports different plan formats
for demo scenes, slides, and BlenderGym tasks.
"""

from typing import Dict, List

from mcp.server.fastmcp import FastMCP

# Tool configuration for demo scene planning
demo_plan_tool: Dict[str, object] = {
    "type": "function",
    "function": {
        "name": "initialize_plan",
        "description": "From the given inputs, imagine and articulate the scene in detail. This tool does not return new information. It stores your detailed description as your own plan to guide subsequent actions. You must call this tool first.",
        "parameters": {
            "type": "object",
            "properties": {
                "overall_description": {"type": "string", "description": "A thorough, comprehensive depiction of the entire scene.\nExample (Simple Room — Overall Description): “A compact, modern study room measuring 4.0 m (X) × 3.0 m (Y) × 2.8 m (Z), with the world origin at the center of the floor. Walls are matte white (slightly warm); the floor is light-gray concrete with subtle roughness; the ceiling is white. The +Y side is the ‘north wall,’ −Y is ‘south,’ +X is ‘east,’ −X is ‘west.’ A single rectangular window (1.2 m × 1.0 m) is centered on the west wall (X = −2.0 m plane), sill height 0.9 m from the floor, with a thin black metal frame and frosted glass that softly diffuses daylight. Primary furniture: a medium-tone oak desk against the north wall, a simple black task chair, a slim floor lamp to the desk’s right, and a low potted plant softening the corner. A framed A2 poster hangs above the desk, and a 1.6 m × 1.0 m flat-woven rug (light beige) sits beneath the desk area. Lighting combines soft daylight from the window with a warm key from the floor lamp; the ambience is calm, minimal, and functional.”"},
                "detailed_plan": {"type": "string", "description": "Consider a detailed plan for scene construction. This plan should follow this format:\n1. Preparation Stage: Use the appropriate tool to generate and download the necessary 3D assets, which are typically complex objects that cannot be constructed using basic geometry.\n2. Rough Stage: Establish the global layout and basic environment components, including the floor, walls or background, camera, and main light source.\n3. Intermediate Stage: Import the downloaded objects into the scene, adjusting their positions, scales, and orientations to align with the global layout. Construct any missing objects using basic geometry.\n4. Refinement Stage: Refine details, enhance materials, add auxiliary lights and props, and make precise local adjustments to enhance realism and accuracy."}
            },
            "required": ["overall_description", "detailed_plan"]
        }
    }
}

# Tool configuration for slide planning
slide_plan_tool: Dict[str, object] = {
    "type": "function",
    "function": {
        "name": "initialize_plan",
        "description": "From the given inputs, imagine and articulate the scene in detail. This tool does not return new information. It stores your detailed description as your own plan to guide subsequent actions. You must call this tool first.",
        "parameters": {
            "type": "object",
            "properties": {
                "detailed_plan": {"type": "string", "description": "Consider a detailed plan for slide construction. This plan should follow this format:\n1. Maintain proper spacing and arrangements of elements in the slide: make sure to keep sufficient spacing between different elements; do not make elements overlap or overflow beyond the slide area.\n2. Carefully select the colors of text, shapes, and backgrounds to ensure all contents are clearly readable. Use high-contrast color combinations and avoid overly bright or clashing tones.\n3. Ensure each slide looks complete and visually engaging. Do not leave large empty areas or make slides appear unfinished. Maintain a strong sense of visual balance, hierarchy, and logical composition when filling in content.\nAt the end of the plan, include:\n- A concise summary of the slide sequence and design logic.\n- Justifications for layout, color, and composition choices.\n- Clear instructions for how each content element will be visually represented.\nThe plan should reflect the thinking process of a professional presentation designer who emphasizes aesthetics, clarity, and functionality."}
            },
            "required": ["detailed_plan"]
        }
    }
}

# Tool configuration for BlenderGym scene editing
blendergym_plan_tool: Dict[str, object] = {
    "type": "function",
    "function": {
        "name": "initialize_plan",
        "description": "From the given inputs, imagine and articulate the scene in detail. This tool does not return new information. It stores your detailed description as your own plan to guide subsequent actions. You must call this tool first.",
        "parameters": {
            "type": "object",
            "properties": {
                "detailed_plan": {"type": "string", "description": "Consider a detailed plan for scene editing. This plan should follow this format:\n1. Visual Analysis: Analyze the visual differences between the initial scene and the target scene. Describe these differences clearly and precisely, focusing on elements such as lighting, camera angle, object placement, texture, or color tone.\n2. Code Analysis: Examine the initial code of the scene and identify which specific lines may be responsible for the visual differences you observed above. Highlight these lines as key areas to focus on during the upcoming experiments.\n3. Action Plan: Provide a detailed plan for the specific code lines and parameters you intend to modify. When adjusting parameters, start with small, incremental changes to observe their effect on the visual output, and gradually refine them based on feedback and observations.\nAt the end of the plan, include:\n- A concise summary linking the visual observations to the planned code modifications.\n- Explanations for why each targeted change is expected to reduce the visual gap.\n- A clear step-by-step schedule for testing and iteratively refining the scene.\nThe plan should reflect the thinking process of a professional scene editor who emphasizes aesthetics, clarity, and functionality."}
            },
            "required": ["detailed_plan"]
        }
    }
}

# Create MCP instance
mcp = FastMCP("initialize-plan-executor")


@mcp.tool()
def initialize(args: Dict[str, object]) -> Dict[str, object]:
    """Initialize the plan tool with mode-specific configuration.

    Args:
        args: Configuration dictionary with 'mode' key to select plan type.

    Returns:
        Dictionary with status and mode-specific tool configuration.
    """
    mode = args.get("mode", "")
    if "blendergym" in mode:
        tool_configs = [blendergym_plan_tool]
    elif "autopresent" in mode:
        tool_configs = [slide_plan_tool]
    else:
        tool_configs = [demo_plan_tool]
    return {
        "status": "success",
        "output": {"text": ["Initialize plan completed"], "tool_configs": tool_configs}
    }


@mcp.tool()
def initialize_plan(
    overall_description: str = '',
    detailed_plan: str = ''
) -> Dict[str, object]:
    """Store the detailed scene plan for guiding subsequent actions.

    Args:
        overall_description: Comprehensive description of the entire scene.
        detailed_plan: Step-by-step plan for scene construction.

    Returns:
        Dictionary with the stored plan and success status.
    """
    output_text = f"{detailed_plan}\nPlease follow the plan carefully."
    return {
        "status": "success",
        "output": {"plan": [output_text], "text": ["Plan initialized successfully"]}
    }


def main() -> None:
    """Run the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()