import json
import importlib
import inspect
import os
from typing import Dict, Any, Callable
import argparse

def get_public_functions(module: Any, base_path: str = "") -> Dict[str, Any]:
    """
    Recursively retrieves a dictionary of public functions from a module and its submodules,
    along with their details.

    :param module: Module from which to retrieve functions.
    :param base_path: String representing the path to the module, used for nested modules.
    :return: Dictionary of function names and their details, keyed by full module path.
    """
    functions_info = {}
    for name, obj in inspect.getmembers(module):
        if inspect.ismodule(obj) and obj.__name__.startswith(module.__name__):
            # Recursive call for submodules
            submodule_path = f"{base_path}.{name}" if base_path else name
            functions_info.update(get_public_functions(obj, submodule_path))
        elif inspect.isfunction(obj) and not name.startswith('_'):
            # Process only public functions
            if base_path:
                full_name = f"{base_path}.{name}"
            else:
                full_name = name
            functions_info[full_name] = get_function_details(obj)
    return functions_info

def get_function_details(func: Callable) -> Dict[str, Any]:
    """
    Retrieves the details of a function including its name, signature, and docstring.

    :param func: Callable function to retrieve details of.
    :return: Dictionary with function details.
    """
    signature = str(inspect.signature(func))
    docstring = inspect.getdoc(func) or "No documentation available."
    func_details = {
        'name': func.__name__,
        'signature': signature,
        'docstring': docstring
    }
    return func_details

def write_functions_to_json(functions_info: Dict[str, Any], output_file: str):
    """
    Writes functions information to a JSON file.

    :param functions_info: Dictionary containing functions and their details.
    :param output_file: Path to the JSON file where the data will be written.
    """
    with open(output_file, 'w') as f:
        json.dump(functions_info, f, indent=4)

def main(module_name: str) -> None:
    """
    Main function that handles user input and writes module function information to a JSON file.

    :param module_name: Name of the module to inspect.
    """
    try:
        module = importlib.import_module(module_name)
        functions_info = get_public_functions(module)
        output_folder = "module_function_details"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        output_filename = os.path.join(output_folder, f"{module_name.replace('.', '_')}_functions.json")
        write_functions_to_json(functions_info, output_filename)
        print(f"Function information saved to {output_filename}")
    except ModuleNotFoundError:
        print(f"The specified module '{module_name}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":2w
    main('langchain')