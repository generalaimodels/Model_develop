import json
import importlib
import inspect
import contextlib
import io
import argparse
from typing import Dict, Any, Callable, List
import os
def get_public_functions(module: Any) -> Dict[str, Any]:
    """
    Retrieves a dictionary of public functions from a module along with their details.

    :param module: Module from which to retrieve functions.
    :return: Dictionary of function names and their details.
    """
    functions_info = {}
    for func_name in dir(module):
        if not func_name.startswith('_'):  # Exclude private attributes/functions
            func = getattr(module, func_name)
            if callable(func):
                functions_info[func_name] = get_function_details(func)
    return functions_info

def get_function_details(func: Callable) -> Dict[str, Any]:
    """
    Retrieves the details of a function including its name, signature, and docstring.

    :param func: Callable function to retrieve details of.
    :return: Dictionary with function details.
    """
    try:
        signature = str(inspect.signature(func))
    except ValueError:
        signature = "Signature not available"

    docstring = inspect.getdoc(func) or "No documentation available."

    # Capture the help text
    with io.StringIO() as buf, contextlib.redirect_stdout(buf):
        help(func)
        help_text = buf.getvalue()

    func_details = {
        'name': func.__name__,
        'signature': signature,
        'docstring': docstring,
        'help': help_text
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
        # Dynamically import the module provided by the user
        module = importlib.import_module(module_name)
        # Retrieve information about public functions in the module
        functions_info = get_public_functions(module)
        # Write the function information to a JSON file
        output_folder = "modules_function"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        output_filename = os.path.join(output_folder, f"{module_name.replace('.', '_')}_functions.json")
        write_functions_to_json(functions_info, output_filename)
        print(f"Function information saved to {output_filename}")
    except ModuleNotFoundError:
        print(f"The specified module '{module_name}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
    # try:
    #     # Dynamically import the module provided by the user
    #     module = importlib.import_module(module_name)
    #     # Retrieve information about public functions in the module
    #     functions_info = get_public_functions(module)
    #     # Write the function information to a JSON file
    #     output_filename = f"{module_name.replace('.', '_')}_functions.json"
    #     write_functions_to_json(functions_info, output_filename)
    #     print(f"Function information saved to {output_filename}")
    # except ModuleNotFoundError:
    #     print(f"The specified module '{module_name}' was not found.")

# import pkg_resources

# def print_installed_modules():
#     """Print names of all installed modules."""
#     installed_packages = pkg_resources.working_set
#     print("List of installed modules:")
#     for package in sorted(installed_packages, key=lambda x: str(x).lower()):
#         package_name = package.project_name
#         try:
#             main(f'{package_name}')  # Assuming main is a function you want to call for each package
#         except Exception as e:
#             print(f"Error occurred while processing {package_name}: {e}")
#         print(str(package_name))

# # Call the function to print the installed module names
# print_installed_modules()

main("langchain.vectorstores")