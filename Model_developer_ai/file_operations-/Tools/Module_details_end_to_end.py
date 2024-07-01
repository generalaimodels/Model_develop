import os
import inspect
from typing import List, TextIO


def generate_module_structure(module_name: str, output_file: str) -> None:
    """
    Generate the structure of a given module in a Markdown file.

    This function imports the specified module and generates a Markdown file that represents
    the structure of the module. It includes directories, files, classes, and functions.

    Args:
        module_name (str): The name of the module to generate the structure for.
        output_file (str): The path to the output Markdown file.

    Raises:
        ModuleNotFoundError: If the specified module is not found.
        ImportError: If there is an error importing the specified module.
        AttributeError: If the specified module has missing attributes.
    """
    try:
        module = __import__(module_name)
        module_path = os.path.dirname(module.__file__)

        with open(output_file, "w", encoding='utf-8') as file:
            file.write(f"# {module_name} Module Structure\n\n")
            traverse_directory(module_path, file, module_name)
    except (ModuleNotFoundError, ImportError, AttributeError) as e:
        print(f"Error: {str(e)}. Skipping...")


def traverse_directory(directory: str, file: TextIO, module_name: str, level: int = 0) -> None:
    """
    Recursively traverse a directory and write its structure to the Markdown file.

    This function traverses the directories and files within the specified directory and
    writes the structure to the Markdown file. It uses indentation to represent the hierarchy
    of directories and files.

    Args:
        directory (str): The directory to traverse.
        file (TextIOWrapper): The file object to write the structure to.
        module_name (str): The name of the current module.
        level (int, optional): The indentation level for the current directory. Defaults to 0.
    """
    indent = "  " * level
    entries = os.listdir(directory)
    folders: List[str] = []
    files: List[str] = []

    for entry in entries:
        if os.path.isdir(os.path.join(directory, entry)):
            folders.append(entry)
        else:
            files.append(entry)

    for folder in folders:
        file.write(f"{indent}- {folder}/\n")
        traverse_directory(os.path.join(directory, folder), file, f"{module_name}.{folder}", level + 1)

    for file_name in files:
        if file_name.endswith(".py") and file_name != "__init__.py":
            module_path = os.path.join(directory, file_name)
            module_name_with_file = f"{module_name}.{file_name[:-3]}"
            file.write(f"{indent}- {file_name}\n")
            write_classes_and_functions(module_name_with_file, file, level + 1)


def write_classes_and_functions(module_name: str, file: TextIO, level: int) -> None:
    """
    Write the classes and functions of a module to the Markdown file, including their signatures and docstrings.

    Args:
        module_name (str): The name of the module.
        file (TextIO): The file object to write the classes and functions to.
        level (int): The indentation level for the classes and functions.

    Raises:
        ModuleNotFoundError: If the specified module or one of its dependencies is not found.
        ImportError: If there is an error importing the specified module.
        AttributeError: If the specified module has missing attributes.
    """
    indent = "  " * level
    try:
        module = __import__(module_name, fromlist=[''])

        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) or inspect.isfunction(obj):
                # Retrieve the signature
                try:
                    sig = inspect.signature(obj)
                except ValueError:
                    sig = '(Unable to retrieve signature)'
                
                # Retrieve the docstring
                docstring = inspect.getdoc(obj) or 'No docstring available'
                docstring = ' '.join(docstring.split())  # Clean up whitespace

                # Write the class or function name, signature, and docstring
                file.write(f"{indent}  -- {'Class' if inspect.isclass(obj) else 'Function'}: `{name}{sig}`\n")
                file.write(f"{indent}  -- Description: {docstring}\n")
    except (ModuleNotFoundError, ImportError, AttributeError) as e:
        file.write(f"{indent}- (Error: {str(e)})\n")


# Example usage
module_name = "torchvision"
output_file = "torch_structure.md"
generate_module_structure(module_name, output_file)
print(f"Module structure generated in {output_file}")