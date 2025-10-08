generator_hints = """1. Understand Blend Shape Semantics
Use the blend shape name (e.g., "BellySag", "ChestEnlarge") as a linguistic cue to infer what part or feature it affects. Match user-desired features (from prompts or comparisons) with blend shape names.

2. Adjust with Care: Continuous Parameters
Each blend shape has a continuous value (e.g., 0.0 to 5.0). Start with small changes (±1.0) to observe impact. Gradually refine based on feedback.

3. Avoid Redundant Edits
If a shape key already has no effect (value 0) and the visual result aligns with the target, do not modify it. Focus only on shape keys that contribute meaningfully.

4. Edit One or Few Keys at a Time
To isolate the effect of each blend shape, modify only one or a small group of related shape keys per step. This helps ensure interpretable changes.

5. Think Iteratively
This is not a one-shot task. Use a loop of (edit → observe → evaluate) to converge toward the desired shape."""

verifier_hints = """0. Use `compare_image` tool first to identify the difference between current scene and target image.

1. Understand Blend Shape Semantics
Use the blend shape name (e.g., "BellySag", "ChestEnlarge") as a linguistic cue to infer what part or feature it affects. Match user-desired features (from prompts or comparisons) with blend shape names.

2. Adjust with Care: Continuous Parameters
Each blend shape has a continuous value (e.g., 0.0 to 5.0). Start with small changes (±1.0) to observe impact. Gradually refine based on feedback.

3. Avoid Redundant Edits
If a shape key already has no effect (value 0) and the visual result aligns with the target, do not modify it. Focus only on shape keys that contribute meaningfully.

4. Edit One or Few Keys at a Time
To isolate the effect of each blend shape, modify only one or a small group of related shape keys per step. This helps ensure interpretable changes.

5. Think Iteratively
This is not a one-shot task. Use a loop of (edit → observe → evaluate) to converge toward the desired shape."""