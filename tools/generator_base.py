from mcp.server.fastmcp import FastMCP

mcp = FastMCP("generator-base")

# tool_configs for agent (only the function w/ @mcp.tools)
tool_configs = [
    {
        "type": "function",
        "function": {
            "name": "init_plan",
            "description": "From the given inputs, imagine and articulate the scene in detail. This tool does not return new information. It stores your detailed description as your own plan to guide subsequent actions. You MUST call this tool first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "overall_description": {"type": "string", "description": "A thorough, comprehensive depiction of the entire scene.\nExample (Simple Room — Overall Description): “A compact, modern study room measuring 4.0 m (X) × 3.0 m (Y) × 2.8 m (Z), with the world origin at the center of the floor. Walls are matte white (slightly warm); the floor is light-gray concrete with subtle roughness; the ceiling is white. The +Y side is the ‘north wall,’ −Y is ‘south,’ +X is ‘east,’ −X is ‘west.’ A single rectangular window (1.2 m × 1.0 m) is centered on the west wall (X = −2.0 m plane), sill height 0.9 m from the floor, with a thin black metal frame and frosted glass that softly diffuses daylight. Primary furniture: a medium-tone oak desk against the north wall, a simple black task chair, a slim floor lamp to the desk’s right, and a low potted plant softening the corner. A framed A2 poster hangs above the desk, and a 1.6 m × 1.0 m flat-woven rug (light beige) sits beneath the desk area. Lighting combines soft daylight from the window with a warm key from the floor lamp; the ambience is calm, minimal, and functional.”"},
                    "object_list": {"type": "string", "description": "List of objects in the scene, including camera and lights.\nExample (Simple Room — Object List):  \n• Architectural: floor plane (4×3 m), four walls, ceiling plane, west-wall window (frame + glass).\n• Furniture: oak desk (rectangular top), black task chair (five-star base or four-leg variant), slim floor lamp (cylindrical shade).\n• Props: closed laptop, mouse, ceramic mug, framed A2 poster, potted plant (pot + medium leafy foliage), flat rug.\n• Lighting: environment daylight (soft), key light approximating window bounce or lamp emission, optional fill."},
                    "object_relations": {"type": "string", "description": "For each object, list related objects and spatial relations, including camera and lights.\nExample (Simple Room — Object Relations):\n• Desk: centered along the north wall (+Y); back edge ~0.05 m from wall; desk top Z ≈ 0.75 m.\n• Chair: in front of desk; seat center ~0.6 m from desk front edge; faces +Y.\n• Floor Lamp: to the desk’s right (east side); base ~0.4 m from desk right edge; shade center Z ≈ 1.5 m.\n• Poster: centered above desk; bottom edge ~0.25 m above desk top. \n• Rug: centered under desk/chair zone; long side aligned with X axis. \n• Window: centered on west wall (X = −2.0 m); lower edge Z = 0.9 m; daylight enters −X → +X. \n• Laptop, mouse, mug: on desk; laptop centered, mouse to the right, mug at left-rear corner. \n• Plant: near northwest corner (−X, +Y), ~0.4 m offset from both walls."},
                    "initial_layout": {"type": "string", "description": "A rough spatial layout for each object, with numeric coordinates (meters, Z-up), including camera and lights.\nExample (Simple Room — Initial Layout Plan):\n• Room envelope: floor X ∈ [−2.0, +2.0], Y ∈ [−1.5, +1.5]; walls at Y = ±1.5, X = ±2.0; ceiling Z = 2.8.\n• Window frame: on X = −2.0; center ≈ (−2.0, 0.0, 1.4); size 1.2 × 1.0 (negligible thickness).\n• Desk: center ≈ (0.0, +1.2, 0.75); size ≈ 1.4 (X) × 0.7 (Y) × 0.75 (Z top).\n• Chair: center ≈ (0.0, +0.5, 0.45); faces +Y; seat top Z ≈ 0.45.\n• Floor Lamp: base ≈ (+0.9, +1.1, 0.0); shade center Z ≈ 1.5.\n• Poster: center ≈ (0.0, +1.48, 1.35); A2 (~0.594 × 0.420 m). \n• Rug: center ≈ (0.0, +0.85, 0.0); size 1.6 (X) × 1.0 (Y). \n• Plant: pot center ≈ (−1.5, +1.2, 0.0); plant height ~0.6–0.8. \n• Camera (initial suggestion): (X, Y, Z) ≈ (+3.6, −2.4, 1.6), target (0, +0.9, 0.9). \n• Key light (if not using emissive lamp): soft area light aligned −X → +X, intensity balanced for exposure."}
                },
                "required": ["overall_description", "object_list", "object_relations", "initial_layout"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "end",
            "description": "No-op tool used to indicate the process should end. If the scene has no remaining issues, stop making changes and call this tool.",
        }
    }
]

@mcp.tool()
def init_plan(overall_description: str, object_list: list, object_relations: list, initial_layout: list) -> dict:
    """
    Store the detailed scene plan to a file and return the path.
    """
    output_text = f"Overall Description: {overall_description}\nObject List: {object_list}\nObject Relations: {object_relations}\nInitial Layout: {initial_layout}"
    return {"status": "success", "output": {"text": [output_text]}}

@mcp.tool()
def end() -> dict:
    """
    No-op tool used to indicate the process should end.
    """
    return {"status": "success", "output": {"text": ["END THE PROCESS"]}}
