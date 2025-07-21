from .prompt import api_library, hints

autopresent_generator_hints = {
    'refinement': hints
}

autopresent_verifier_hints = None

autopresent_generator_system = """You are a slide design agent. Your task is to edit code to transform an initial slide into a target slide following the target description provided. After each code edit, your code will be passed to a validator, which will provide feedback on the result. Based on this feedback, you must iteratively refine your code edits. This process will continue across multiple rounds of dialogue. In each round, you must adhere to a fixed output format."""

autopresent_verifier_system = """You're a slide feedback assistant tasked with providing revision suggestions to a slide designer. At the beginning, you will be given a task instruction that describes the target slide. In each subsequent interaction, you will receive an image of the current slide along with the code that generated it.
Your responsibilities include:
1. **Visual Difference Identification**: Identify differences between the target slide and the current slide.
2. **Code Localization**: Pinpoint locations in the code that could be modified to reduce or eliminate these differences. This may require counterfactual reasoning and inference from the visual discrepancies."""

autopresent_api_library = api_library

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
If the current slide is very close to the target slide, only output an "OK!" and do not output other characters."""