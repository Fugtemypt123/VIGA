"""Library Documentation Generator.

Extracts function signatures and docstrings from library modules.
"""
import ast


def extract_functions(filename: str) -> str:
    """Extract function signatures and docstrings from a Python file.

    Args:
        filename: Path to the Python file to parse.

    Returns:
        Formatted string containing function signatures and docstrings.
    """
    with open(filename, "r") as f:
        tree = ast.parse(f.read())

    functions = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            signature = node.name + "(" + ", ".join(arg.arg for arg in node.args.args) + ")"
            docstring = ast.get_docstring(node) or "No docstring"
            functions.append('\n'.join([signature, f'"""{docstring}\n"""']))

    return '\n\n'.join(functions)


if __name__ == "__main__":
    instruction = """You can import all functions above by importing from the library.
For example, `from library import *` or `from library import {function_name}`."""

    # Usage
    basic_content = extract_functions("library_basic.py")
    with open("library_basic.txt", 'w') as fw:
        fw.write("## Helper functions\n\n" + basic_content + '\n\n' + instruction)

    image_content = extract_functions("library_image.py")
    with open("library_image.txt", 'w') as fw:
        fw.write("## Helper functions\n\n" + image_content + '\n\n' + instruction)

    with open("library.txt", 'w') as fw:
        fw.write("## Helper functions\n\n" + basic_content + '\n\n' + image_content + '\n\n' + instruction)
