import importlib
import csv
import sys
import inspect
import csv

import torch

def analyze_module_recursively(module, *, output_file="functionality2.csv"):
    """
    Analyzes a given module recursively, extracting details for each public function using function introspection.
    Writes the results to a CSV file.

    Args:
        module: The Python module to analyze.
        output_file (str, optional): Name of the CSV file to write results to. Defaults to "functionality.csv".

    Returns:
        None
    """

    all_functions = {}  # Dictionary to store function details

    def analyze_recursively(current_module, parent_call: list = []):
        """
        Analyzes a module recursively and stores function details.

        Args:
            current_module: The module to analyze.
            parent_call (list, optional): The call chain leading to the current analysis context.

        Returns:
            None
        """

        for name, obj in inspect.getmembers(current_module, inspect.isfunction):
            if not name.startswith('_'):
                # Found a public function

                # Analyze function signature and docstring using introspection
                signature = inspect.signature(obj)
                docstring = inspect.getdoc(obj) or ''
                # Sanitize docstring by replacing newlines with spaces
                sanitized_docstring = ' '.join(docstring.splitlines())

                # Store data for the current function
                all_functions[(current_module.__name__, name)] = {
                    "call_chain": parent_call + [name],
                    "call_depth": len(parent_call),
                    "arguments": list(signature.parameters),
                    "return_type": signature.return_annotation,
                    "docstring": sanitized_docstring
                }

    # Main analysis call
    analyze_recursively(module)

    # Write data to CSV file
    with open(output_file, "w", newline="") as csvfile:
        fieldnames = ["Module", "Function", "CallChain", "CallDepth", "Arguments", "ReturnType", "Docstring"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for (module_name, function_name), details in all_functions.items():
            writer.writerow({
                "Module": module_name,
                "Function": function_name,
                "CallChain": "->".join(details["call_chain"]),
                "CallDepth": details["call_depth"],
                "Arguments": ", ".join([str(arg) for arg in details["arguments"]]),
                "ReturnType": getattr(details["return_type"], '__name__', str(details["return_type"])),
                "Docstring": details["docstring"]
            })


def get_public_functionalities(module_name):
    """
    Get a list of public functionalities (functions, classes, etc.) from a given module.

    :param module_name: The name of the module as a string.
    :return: A list of public functionalities in the module.
    """
    # Dynamically import the module.
    module = importlib.import_module(module_name)
    # Get all attributes from the module.
    functionality_dir = dir(module)
    # Filter out private functionalities (those that start with an underscore).
    public_functionalities = [fun for fun in functionality_dir if not fun.startswith('_')]
    return public_functionalities

def write_functionalities_to_csv(module_name, functionalities, filename='functionalities.csv'):
    """
    Write the list of functionalities to a CSV file.

    :param module_name: The name of the module.
    :param functionalities: The list of functionalities to write.
    :param filename: The name of the CSV file.
    """
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header row.
        writer.writerow(['Module', 'Functionality'])
        # Write each functionality.
        for functionality in functionalities:
            writer.writerow([module_name, functionality])

# The main function to tie the script together.
def main(module_name):
    """
    The main function of the script. It gets the public functionalities of the given module
    and writes them to a CSV file.

    :param module_name: The name of the module to inspect.
    """
    functionalities = get_public_functionalities(module_name)
    write_functionalities_to_csv(module_name, functionalities)

main('langchain')