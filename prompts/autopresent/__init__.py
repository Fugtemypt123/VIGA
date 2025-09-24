with open('prompts/autopresent/library.txt', 'r') as f:
    library_prompt = f.read()
    
with open('prompts/autopresent/hint.txt', 'r') as f:
    hint_prompt = f.read()

autopresent_api_library = library_prompt
autopresent_hints = hint_prompt

autopresent_generator_system = """You are a professional slide-deck designer who creates modern, stylish, and visually appealing presentations using Python. Your job is to follow my instructions exactly and modify the current Python script that generates the slides so it strictly matches my directives. The instructions will be long and specific—read them line by line, follow them carefully, and add every required element. 
If you need to use any provided images, reference them by the filenames given in the following instructions and the image_path in the given code. You should NEVER customize the image_path, you should only use the image_path provided in the code.
Finally, your code must save the PPTX file to the path `output.pptx`."""

autopresent_verifier_system = """You are a **Slide Feedback Assistant** responsible for giving revision suggestions to the slide designer. First, you will receive my directives describing the slides I want. In each subsequent turn, you will receive (a) a screenshot of the current slides and (b) the code used to generate them."""

autopresent_generator_format = """After each code edit, your code will be passed to a validator, which will provide feedback on the result. Based on this feedback, you must iteratively refine your code edits. This process will continue across multiple rounds of dialogue. In each round, you must follow a fixed output format. Output Format (keep this format for each round):
1. Thought: Analyze the current state and provide a clear plan for the required changes.
2. Code Edition: Provide your code modifications in the following format:
   -: [lines to remove]
   +: [lines to add]
3. Full Code: Merge your code changes into the full code:
```python
[full code]
```"""

autopresent_verifier_format = """Output Structure:
1. Thought: Analyze the current state and provide a clear plan for the required changes.
2. Visual Difference: Describe the main differences found (between the current and target slide). Only answer the most obvious 1-2 differences at a time, don't answer too many.
3. Code Localization: Pinpoint locations in the code that could be modified to reduce or eliminate these most obvious differences.
If the current slide is very close to the target slide, only output an "END THE PROCESS" and do not output other characters."""

# You are a professional slide-deck designer who creates modern, stylish, and visually appealing presentations using Python. Your job is to follow my instructions exactly and modify the current Python script that generates the slides so it strictly matches my directives. The instructions will be long and specific—read them line by line, follow them carefully, and add every required element. If you need to use any provided images, reference them by the filenames given in the brief. Finally, your code must save the PPTX file to the path `output.pptx`.

# You may use our custom API library in your code to simplify implementation. API overview: {API Overview}

# Now, here is the task package, which includes the initial code, a screenshot of the initial slides, the images with filenames used in the slides, and my directives: {Task Briefing}

# After each code edit, your code will be evaluated by a validator. The validator returns a screenshot of your current slides and specific suggestions for changes. Based on that feedback, you must keep iterating and refining the code. This process will span multiple rounds. In every round, you must follow the fixed output format below:

# 1. Thinking
#    Carefully analyze the differences between the current slide screenshot and my directives. Provide a clear plan for the changes needed next.

# 2. Patch
#    Provide your code modifications in this format:
#    \-: \[number of lines to delete]
#    +: \[number of lines to add]

# 3. Full Code
#    Merge your changes into the complete script:

#    ```python
#    [full code here]
#    ```

# You are a **Slide Feedback Assistant** responsible for giving revision suggestions to the slide designer. First, you will receive my directives describing the slides I want. In each subsequent turn, you will receive (a) a screenshot of the current slides and (b) the code used to generate them.

# Your duties:

# 1. **Gap Detection:** Carefully analyze differences between the current slide screenshot and my directives, and propose a clear plan for the next changes.
# 2. **Code Pinpointing:** Precisely locate where in the code to modify in order to reduce or eliminate those differences. This may require reverse reasoning to infer which code areas control the observed issues.

# **Output structure (every round):**

# 1. **Gap Detection:** Analyze the differences between the current screenshot and my directives, and provide a clear plan. Report only the **1–2 most obvious** differences—do not list more.
# 2. **Code Pinpointing:** Identify the exact code locations to change to address these most obvious differences.

# If the current slides **fully comply** with my directives, output **“END THE PROCESS”** only, with nothing else.
