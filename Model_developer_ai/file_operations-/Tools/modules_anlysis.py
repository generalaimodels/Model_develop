import importlib
import csv
import sys
import inspect
import os
import json
import pkg_resources

def analyze_module_recursively(module, *, output_file="functionality2.json"):
    all_functions = {}

    def analyze_recursively(current_module, parent_call: list = []):
        for name, obj in inspect.getmembers(current_module, inspect.isfunction):
            if not name.startswith('_'):
                signature = inspect.signature(obj)
                docstring = inspect.getdoc(obj) or ''
                sanitized_docstring = ' '.join(docstring.splitlines())

                all_functions[(current_module.__name__, name)] = {
                    "call_chain": parent_call + [name],
                    "call_depth": len(parent_call),
                    "arguments": list(signature.parameters),
                    "return_type": getattr(signature.return_annotation, '__name__', str(signature.return_annotation)),
                    "docstring": sanitized_docstring
                }

    analyze_recursively(module)

    with open(output_file, "w", encoding="utf-8") as jsonfile:
        json.dump(all_functions, jsonfile, indent=4)

def get_public_functionalities(module_name):
    module = importlib.import_module(module_name)
    functionality_dir = dir(module)
    public_functionalities = [fun for fun in functionality_dir if not fun.startswith('_')]
    return public_functionalities

def write_functionalities_to_json(module_name, functionalities, filename='functionalities.json'):
    data = {module_name: functionalities}
    with open(filename, "w", encoding="utf-8") as jsonfile:
        json.dump(data, jsonfile, indent=4)

def generate_requirements_file(output_filename="requirements.txt"):
    installed_packages = sorted(["%s==%s" % (i.key, i.version) for i in pkg_resources.working_set])

    with open(output_filename, "w", encoding="utf-8") as file:
        for package in installed_packages:
            file.write(package + "\n")

    print(f"Generated requirements file: {output_filename}")

def main(module_names):
    output_folder = "json_files"
    os.makedirs(output_folder, exist_ok=True)

    for module_name in module_names:
        module_folder = os.path.join(output_folder, module_name)
        os.makedirs(module_folder, exist_ok=True)

        functionalities = get_public_functionalities(module_name)
        json_filename = os.path.join(module_folder, 'functionalities.json')
        write_functionalities_to_json(module_name, functionalities, filename=json_filename)

        module = importlib.import_module(module_name)
        json_filename = os.path.join(module_folder, 'functionality2.json')
        analyze_module_recursively(module, output_file=json_filename)

    requirements_filename = os.path.join(output_folder, 'requirements.txt')
    generate_requirements_file(output_filename=requirements_filename)

if __name__ == "__main__":
    module_names = ['transformers', 'torch', 'numpy']  # Add more module names as needed
    main(module_names)