generator_hints = """1. Focus on Color and Texture Details
This task always centers on the visual appearance of the material—especially its base color, roughness, metallic, and normal map settings. Even subtle changes in these parameters can significantly alter the surface look.

2. Match Visual Attributes to Code Parameters
Use semantic cues in parameter names (e.g., "Base Color", "Roughness", "Bump Strength") to align code edits with observed visual differences. Adjust values to produce desired texture and reflectivity.

3. Observe Fine-Grained Texture Cues
Pay close attention to micro-structure details in the image—such as glossiness, bumpiness, or surface noise. These are often controlled by procedural nodes (e.g., noise, voronoi, or musgrave).

4. Edit Iteratively, Validate Visually
Since procedural materials are highly sensitive to small changes, apply edits incrementally and check the rendered output for the desired visual effect before proceeding further.

5. Think Physically and Aesthetically
Consider how material parameters affect realistic physical behavior (e.g., metallic surfaces reflect sharply; rough surfaces scatter light). Adjust accordingly to achieve the intended material feel."""

verifier_hints = """0. Use `compare_image` tool first to identify the difference between current scene and target image.

1. Focus on Color and Texture Details
This task always centers on the visual appearance of the material—especially its base color, roughness, metallic, and normal map settings. Even subtle changes in these parameters can significantly alter the surface look.

2. Match Visual Attributes to Code Parameters
Use semantic cues in parameter names (e.g., "Base Color", "Roughness", "Bump Strength") to align code edits with observed visual differences. Adjust values to produce desired texture and reflectivity.

3. Observe Fine-Grained Texture Cues
Pay close attention to micro-structure details in the image—such as glossiness, bumpiness, or surface noise. These are often controlled by procedural nodes (e.g., noise, voronoi, or musgrave).

4. Edit Iteratively, Validate Visually
Since procedural materials are highly sensitive to small changes, apply edits incrementally and check the rendered output for the desired visual effect before proceeding further.

5. Think Physically and Aesthetically
Consider how material parameters affect realistic physical behavior (e.g., metallic surfaces reflect sharply; rough surfaces scatter light). Adjust accordingly to achieve the intended material feel."""