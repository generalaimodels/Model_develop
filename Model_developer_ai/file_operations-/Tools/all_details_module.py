# import argparse
# import importlib
# import inspect
# import json
# from typing import Any, Dict

# def get_function_details(func) -> Dict[str, Any]:
#     """
#     Extracts details from a function, including its name, signature, and docstring.

#     Args:
#         func (Function): The function to inspect.

#     Returns:
#         Dict[str, Any]: A dictionary containing the function's name, signature, and docstring.
#     """
#     return {
#         "name": func.__name__,
#         "signature": str(inspect.signature(func)),
#         "doc": func.__doc__
#     }

# def inspect_module(module_name: str) -> Dict[str, Any]:
#     """
#     Inspects the given module and returns a dictionary with detailed information about its functions.

#     Args:
#         module_name (str): The name of the module to inspect.

#     Returns:
#         Dict[str, Any]: Detailed information about the module's functions.
#     """
#     module_info = {"functions": []}
#     module = importlib.import_module(module_name)

#     for name, obj in inspect.getmembers(module, inspect.isfunction):
#         func_details = get_function_details(obj)
#         module_info["functions"].append(func_details)

#     return module_info

# def main():
#     parser = argparse.ArgumentParser(description="Generate JSON file with detailed module information.")
#     parser.add_argument('--module_name', type=str, help="Name of the module to inspect.")

#     args = parser.parse_args()

#     module_info = inspect_module(args.module_name)
#     json_filename = f"{args.module_name.replace('.', '_')}_info.json"

#     with open(json_filename, 'w') as f:
#         json.dump(module_info, f, indent=4)

#     print(f"Module information saved to {json_filename}")

# if __name__ == "__main__":
#     main()


import json
import torch
import paramiko
import random
import inspect
import contextlib
import io
import os
from typing import Dict, Any

def get_public_functions(module) -> Dict[str, Any]:
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
                functions_info[func_name] = get_function_details(func, module)
    return functions_info

def get_function_details(func: callable, module) -> Dict[str, Any]:
    """
    Retrieves the details of a function including its name, signature, and docstring.

    :param func: Callable function to retrieve details of.
    :param module: Module containing the function.
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

if __name__ == "__main__":
    # Retrieve information about public functions in the torch module
    module = paramiko
    torch_functions_info = get_public_functions(module)
    # Write the function information to a JSON file
    output_filename = "paramiko.json"
    write_functions_to_json(torch_functions_info, output_filename)