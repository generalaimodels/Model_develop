import pkg_resources
import json
from typing import Dict, List, Any

def generate_requirements_file(output_filename: str = "requirements.txt") -> None:
    """
    Creates a requirements file listing the user's currently installed Python packages.
    
    Args:
        output_filename (str): The name of the file to generate.
        
    Returns:
        None
    """
    installed_packages = sorted(["%s==%s" % (i.key, i.version) for i in pkg_resources.working_set])
    
    with open(output_filename, "w", encoding="utf-8") as file:
        for package in installed_packages:
            file.write(package + "\n")
    
    print(f"Generated requirements file: {output_filename}")
def list_functions_from_module(module_name: str) -> List[str]:
    """
    Lists functions available in a given module.
    
    Args:
        module_name (str): The name of the module to inspect.
        
    Returns:
        List[str]: A list of function names found in the module.
    """
    try:
        module = __import__(module_name)
        function_list = []
        for func in dir(module):
            try:
                # Call getattr and filter out special attributes and non-functions.
                attribute = getattr(module, func)
                if callable(attribute) and not func.startswith('__'):
                    function_list.append(func)
            except AttributeError:
                # Skip attributes that raise AttributeError on access.
                pass
        return function_list
    except (ModuleNotFoundError, ImportError, RuntimeError):
        # If module can't be imported or other import-related errors occur, return an empty list
        print(f"Warning: Skipping module '{module_name}' due to import error.")
        return []
def generate_functions_json(requirements_filename: str = "requirements.txt", output_filename: str = "functions.json") -> None:
    """
    Generates a JSON file containing functions from the modules listed in the requirements file.
    
    Args:
        requirements_filename (str): The name of the requirements file to read.
        output_filename (str): The name of the JSON file to generate.
        
    Returns:
        None
    """
    functions_dict: Dict[str, Any] = {}

    with open(requirements_filename, "r", encoding="utf-8") as file:
        for line in file:
            package_name = line.split('==')[0]
            functions_list = list_functions_from_module(package_name)
            functions_dict[package_name] = functions_list

    try:
        with open(output_filename, "w", encoding="utf-8") as file:
            json.dump(functions_dict, file, indent=4)
        print(f"Generated JSON file: {output_filename}")
    except TypeError as e:
        print(f"Error writing JSON file: {e}")
if __name__ == "__main__":
    # Step 1: Generate the requirements.txt file
    generate_requirements_file()
    
    # Step 2: Generate the functions.json file from the requirements.txt
    generate_functions_json()