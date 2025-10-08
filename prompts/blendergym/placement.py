generator_hints = """1. Understand Start vs. Goal Scene Layout
Carefully compare the start and goal scenes to identify which objects have moved and where they need to go. Focus on relative positions, distances, and alignment between objects.

2. Read the Placement Functions in the Code
Understand the behavior of each function provided in the script. Wisely use them—whether to set absolute positions, apply offsets, or place objects relative to others—to accomplish the spatial rearrangement.

3. Choose the Most Interpretable Operation
Prefer the method that most clearly reflects the spatial change. For example, if the object moves a fixed amount, use vector offset; if it aligns with another object, use relative placement.

4. Reason About Spatial Relationships
Placement is not just about coordinates—it reflects spatial logic (e.g., “to the left of”, “behind”, “stacked on top of”). Use this to guide your choice of direction and distance in relative placement.

5. Validate Placement Visually
After each move, observe the visual scene. Check if the new position matches the goal scene in both location and context (e.g., symmetry, grouping, contact with surfaces)."""

verifier_hints = """0. Use `compare_image` tool first to identify the difference between current scene and target image.

1. Understand Start vs. Goal Scene Layout
Carefully compare the start and goal scenes to identify which objects have moved and where they need to go. Focus on relative positions, distances, and alignment between objects.

2. Read the Placement Functions in the Code
Understand the behavior of each function provided in the script. Wisely use them—whether to set absolute positions, apply offsets, or place objects relative to others—to accomplish the spatial rearrangement.

3. Choose the Most Interpretable Operation
Prefer the method that most clearly reflects the spatial change. For example, if the object moves a fixed amount, use vector offset; if it aligns with another object, use relative placement.

4. Reason About Spatial Relationships
Placement is not just about coordinates—it reflects spatial logic (e.g., “to the left of”, “behind”, “stacked on top of”). Use this to guide your choice of direction and distance in relative placement.

5. Validate Placement Visually
After each move, observe the visual scene. Check if the new position matches the goal scene in both location and context (e.g., symmetry, grouping, contact with surfaces)."""