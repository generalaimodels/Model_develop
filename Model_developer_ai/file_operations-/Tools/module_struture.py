import os
import inspect
from typing import List

def generate_module_structure(module_name: str, output_file: str) -> None:
    """
    Generate the structure of a given module in a Markdown file.

    Args:
        module_name (str): The name of the module to generate the structure for.
        output_file (str): The path to the output Markdown file.
    """
    try:
        module = __import__(module_name)
        module_path = os.path.dirname(module.__file__)

        with open(output_file, "w", encoding='utf-8') as file:
            file.write(f"# {module_name} Module Structure\n\n")
            file.write("```\n")
            traverse_directory(module_path, file, module_name)
            file.write("```\n")
    except (ModuleNotFoundError, ImportError, AttributeError):
        print(f"Module '{module_name}' or one of its dependencies not found or has missing attributes. Skipping...")

def traverse_directory(directory: str, file, module_name: str, level: int = 0) -> None:
    """
    Recursively traverse a directory and write its structure to the Markdown file.

    Args:
        directory (str): The directory to traverse.
        file (TextIOWrapper): The file object to write the structure to.
        module_name (str): The name of the current module.
        level (int): The indentation level for the current directory.
    """
    indent = "│   " * level
    entries = os.listdir(directory)
    folders: List[str] = []
    files: List[str] = []

    for entry in entries:
        if os.path.isdir(os.path.join(directory, entry)):
            folders.append(entry)
        else:
            files.append(entry)

    for folder in folders:
        file.write(f"{indent}├── {folder}/\n")
        traverse_directory(os.path.join(directory, folder), file, f"{module_name}.{folder}", level + 1)

    for file_name in files:
        if file_name.endswith(".py") and file_name != "__init__.py":
            module_path = os.path.join(directory, file_name)
            module_name_with_file = f"{module_name}.{file_name[:-3]}"
            file.write(f"{indent}├── {file_name}\n")
            write_classes_and_functions(module_name_with_file, file, level + 1)

    if level > 0:
        file.write(f"{indent[:-4]}└──\n")

def write_classes_and_functions(module_name: str, file, level: int) -> None:
    """
    Write the classes and functions of a module to the Markdown file.

    Args:
        module_name (str): The name of the module.
        file (TextIOWrapper): The file object to write the classes and functions to.
        level (int): The indentation level for the classes and functions.
    """
    indent = "│   " * level
    try:
        module = __import__(module_name, fromlist=[''])

        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj):
                file.write(f"{indent}│   Class: {name}\n")
            elif inspect.isfunction(obj):
                file.write(f"{indent}│   Function: {name}\n")
    except (ModuleNotFoundError, ImportError, AttributeError):
        file.write(f"{indent}│   (Module or dependency not found or has missing attributes)\n")

# Example usage
module_name = "trl"
output_file = "torch_structure.md"
generate_module_structure(module_name, output_file)
print(f"Module structure generated in {output_file}")