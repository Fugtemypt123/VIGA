generator_hints = """1. Control Light Color, Intensity, Position, and Direction
Lighting is defined not only by how bright or what color the light is, but also by where it comes from and where it's pointed. Adjust all four dimensions to shape the scene’s illumination.

2. Infer Light Source from Shadows
In many cases, light sources are not directly visible in the camera view. Instead, use shadow orientation and softness to infer the direction, distance, and sharpness of the light.

3. Match Lighting to Scene Goals
Adjust lighting to create intended visual effects, such as highlighting an object, creating dramatic contrast, or softening harsh shadows. Consider how each edit affects scene mood and visibility.

4. Iterate Using Visual Cues
After each lighting change, examine the resulting shadows, highlights, and overall balance. Use these cues to refine light parameters, especially when indirect lighting is involved.

5. Avoid Overexposure or Flat Lighting
Be cautious with intensity. Lights that are too bright or too diffuse can wash out details. Aim for balanced contrast that maintains depth and material fidelity."""

verifier_hints = """0. Use `compare_image` tool first to identify the difference between current scene and target image.

1. Control Light Color, Intensity, Position, and Direction
Lighting is defined not only by how bright or what color the light is, but also by where it comes from and where it's pointed. Adjust all four dimensions to shape the scene’s illumination.

2. Infer Light Source from Shadows
In many cases, light sources are not directly visible in the camera view. Instead, use shadow orientation and softness to infer the direction, distance, and sharpness of the light.

3. Match Lighting to Scene Goals
Adjust lighting to create intended visual effects, such as highlighting an object, creating dramatic contrast, or softening harsh shadows. Consider how each edit affects scene mood and visibility.

4. Iterate Using Visual Cues
After each lighting change, examine the resulting shadows, highlights, and overall balance. Use these cues to refine light parameters, especially when indirect lighting is involved.

5. Avoid Overexposure or Flat Lighting
Be cautious with intensity. Lights that are too bright or too diffuse can wash out details. Aim for balanced contrast that maintains depth and material fidelity."""