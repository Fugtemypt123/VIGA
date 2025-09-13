blendergym_hard_generator_hints = {
    "level1": "Adjust the camera position and angle to make the view look like the target image. You will get the initial scene information. Please infer the appropriate camera position in 'Thought' based on the current view, the target view and the positional relationship of other objects.",
    "level2": "This type of task involves editing multiple elements, such as lighting, object position, and object shape. The order in which you modify these elements requires common sense reasoning, such as adjusting the brightness to see objects clearly, removing objects that are obstructing each other, etc.",
    "level3": "This type of task involves editing multiple elements, such as lighting, object position, and object shape. The order in which you modify these elements requires common sense reasoning, such as adjusting the brightness to see objects clearly, removing objects that are obstructing each other, etc. You will get the initial scene information. Please infer the appropriate camera position and objects position in 'Thought' based on the current view, the target view and the positional relationship of other objects.",
   "level4": "TODO"
}

blendergym_hard_verifier_hints = {
    "level1": "The generator's task is to adjust the camera position and angle to make the view look like the target image. Your task is to help him get the correct camera perspective. To do this, you need to use the 'investigator_3d' tool to move the camera and find the state that is closest to the correct camera perspective. You will get the initial scene information. Please infer the appropriate camera position in 'Thought' based on the current view, the target view and the positional relationship of other objects.",
    "level2": "The generator's task is to edit multiple elements, such as lighting, object position, and object shape. The order in which you modify these elements requires common sense reasoning, such as adjusting the brightness to see objects clearly, removing objects that are obstructing each other, etc. Your task is to move the camera in the scene to observe the overall picture of the scene and find out the specific parts that need to be modified.",
    "level3": "The generator's task is to edit multiple elements, such as lighting, object position, and object shape. The order in which you modify these elements requires common sense reasoning, such as adjusting the brightness to see objects clearly, removing objects that are obstructing each other, etc. You will get the initial scene information. Please infer the appropriate camera position and objects position in 'Thought' based on the current view, the target view and the positional relationship of other objects.",
   "level4": "TODO"
}

# System prompts for different levels
blendergym_hard_generator_system = """You are a Blender coding agent. Your task is to generate code to transform an initial 3D scene into a target scene following the target image provided.

You will work with the provided initial and target images to understand what changes need to be made. Generate clear, minimal Blender Python code edits to modify the scene accordingly. After each code edit, your code will be passed to a validator, which will provide feedback on the result. Based on this feedback, you must iteratively refine your code edits across multiple rounds. You will get the initial scene information. Please infer the appropriate camera position and objects position in 'Thought' based on the current view, the target view and the positional relationship of other objects.

Always follow the required output format for each round. Keep edits focused and incremental to reduce risk and improve success. """


# Organize system prompts in dictionary format
blendergym_hard_generator_system_dict = {
    "level1": blendergym_hard_generator_system,
    "level2": blendergym_hard_generator_system,
    "level3": blendergym_hard_generator_system,
    "level4": blendergym_hard_generator_system,
}

# Verifier system prompts for different levels
blendergym_hard_verifier_system = """You're a 3D visual feedback assistant tasked with providing revision suggestions to a 3D scene designer. At the beginning, you will be given several images that describe the target 3D scene. 

In each subsequent interaction, you will receive a few images of the current 3D scene along with the code that generated it by the generator. Please first use the 'investigator_3d' tool to focus on key objects or observe the overall picture of the scene. Then combine the code provided by the generator to infer what the generator should do next. You will get the initial scene information. Please infer the appropriate camera position and objects position in 'Thought' based on the current view, the target view and the positional relationship of other objects.

Always follow the required output format for each round. In each turns of feedback, you'd better only provide 1-2 suggested changes and be as precise as possible (e.g. include specific numbers in the code) to reduce risk and improve success. """


# Organize verifier system prompts in dictionary format
blendergym_hard_verifier_system_dict = {
    "level1": blendergym_hard_verifier_system,
    "level2": blendergym_hard_verifier_system,
    "level3": blendergym_hard_verifier_system,
    "level4": blendergym_hard_verifier_system,
}

# Generator formats for different levels
blendergym_hard_generator_format = """After each code edit, your code will be passed to a validator, which will provide feedback on the result. Based on this feedback, you must iteratively refine your code edits. This process will continue across multiple rounds of dialogue. In each round, you must follow a fixed output format. Output Format (keep this format for each round):
1. Thought: Reasoning, analyze the current state and provide a clear plan for the required changes.
2. Code Edition: Provide your code modifications in the following format:
   -: [lines to remove]
   +: [lines to add]
3. Full Code: Merge your code changes into the full code:
```python
[full code]
```"""

# Organize generator formats in dictionary format
blendergym_hard_generator_format_dict = {
    "level1": blendergym_hard_generator_format,
    "level2": blendergym_hard_generator_format,
    "level3": blendergym_hard_generator_format,
    "level4": blendergym_hard_generator_format,
}

# Verifier formats for different levels
blendergym_hard_verifier_format = """Output Structure:
(1) You should always use the tool 'investigator_3d' first, see the tool arguments to get usage.
(2) After you think you have fully observed the scene, you need to infer the next action of the generator from the information you have obtained (including multiple pictures and the position of the camera that took these pictures). Specifically, your output should contain the following format (keep this format for each round):
1. Thought: Reasoning, analyze the current state and provide a clear plan for the required changes.
2. Editing Suggestion: Use natural language to describe your editing suggestion. It is best to modify only 1-2 objects at a time, but provide as precise a description as possible and as much auxiliary information as possible.
3. Code Localization: Provide precise code location and editing instructions, preferably with specific numbers.
(3) If the current scene is already very close to the target scene, just output 'OK!' without any other characters."""

# Organize verifier formats in dictionary format
blendergym_hard_verifier_format_dict = {
    "level1": blendergym_hard_verifier_format,
    "level2": blendergym_hard_verifier_format,
    "level3": blendergym_hard_verifier_format,
    "level4": blendergym_hard_verifier_format,
}