generator_hints = """1. Reason About 3D Geometry, Not Just Appearance
Geometry edits often affect underlying 3D structure, not just surface textures. Consider how shape and topology change in 3D space, especially with respect to occlusion and depth.

2. Handle Occlusion Relationships Carefully
Pay attention to occlusion—e.g., when a core is enclosed within a shell, edits must preserve or reveal this containment relationship. Avoid mistakenly breaking spatial hierarchies.

3. Interpret Shape, Scale, and Position Semantically
Use blend shape names or code variables (e.g., "CoreRadius", "ShellThickness") as cues for what spatial attributes they control. Match these to visual features in the rendered result.

4. Think in Terms of Spatial Composition
Adjust object attributes (scale, distance, alignment) in a way that preserves or transforms spatial distribution as intended (e.g., symmetrical scaling vs. directional stretching).

5. Edit, Observe, Reason Iteratively
Geometry edits require careful, stepwise updates. After each edit, check how spatial relationships (e.g., overlaps, containment, relative distances) have changed in the image, and refine accordingly."""

verifier_hints = """0. Use `compare_image` tool first to identify the difference between current scene and target image.

1. Reason About 3D Geometry, Not Just Appearance
Geometry edits often affect underlying 3D structure, not just surface textures. Consider how shape and topology change in 3D space, especially with respect to occlusion and depth.

2. Handle Occlusion Relationships Carefully
Pay attention to occlusion—e.g., when a core is enclosed within a shell, edits must preserve or reveal this containment relationship. Avoid mistakenly breaking spatial hierarchies.

3. Interpret Shape, Scale, and Position Semantically
Use blend shape names or code variables (e.g., "CoreRadius", "ShellThickness") as cues for what spatial attributes they control. Match these to visual features in the rendered result.

4. Think in Terms of Spatial Composition
Adjust object attributes (scale, distance, alignment) in a way that preserves or transforms spatial distribution as intended (e.g., symmetrical scaling vs. directional stretching).

5. Edit, Observe, Reason Iteratively
Geometry edits require careful, stepwise updates. After each edit, check how spatial relationships (e.g., overlaps, containment, relative distances) have changed in the image, and refine accordingly."""