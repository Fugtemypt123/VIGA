from mcp.server.fastmcp import FastMCP

tool_configs = [
    {
        "type": "function",
        "function": {
            "name": "initialize_plan",
            "description": "From the given inputs, imagine and articulate the scene in detail. This tool does not return new information. It stores your detailed description as your own plan to guide subsequent actions. You must call this tool first when building a new scene.",
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
]

mcp = FastMCP("initialize-plan-executor")

@mcp.tool()
def initialize(args: dict) -> dict:
    return {"status": "success", "output": {"text": ["Initialize plan completed"], "tool_configs": tool_configs}}

@mcp.tool()
def initialize_plan(overall_description: str, detailed_plan: str) -> dict:
    """
    Store the detailed scene plan to a file and return the path.
    """
    output_text = f"{detailed_plan}\nPlease follow the plan carefully."
    return {"status": "success", "output": {"plan": [output_text], "text": ["Plan initialized successfully"]}}
