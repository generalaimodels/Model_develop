import os
import json
from pathlib import Path
from typing import List, Dict
def get_folder_structure(root_folder: str) -> List[Dict]:

    structure = []

    for path, subdirs, files in os.walk(root_folder):
        folder_dict = {
            "folders": os.path.basename(path),
        }

        extensions = {}

        for file in files:
            # Use pathlib to handle Windows paths properly
            from pathlib import Path
            file_path = Path(path, file)
            file_path_str = str(file_path).replace('\\', '/')
            
            ext = os.path.splitext(file_path_str)[1].lower()
            if ext not in extensions:
                extensions[ext] = []
            extensions[ext].append(file_path_str)

        folder_dict["extensions"] = extensions

        structure.append(folder_dict)

    return structure



def write_structure_to_file(structure: List[Dict], output_file: str):
    """
    Write folder structure JSON to a file.
    """
    with open(output_file, "w") as f:
        json.dump(structure, f, indent=4)

def generate_directory_structure(root_folder: str, output_file: str):
    """
    Generate and write folder structure JSON.
    """
    structure = get_folder_structure(root_folder)
    write_structure_to_file(structure, output_file)




def get_folder_structure(root_folder: str) -> List[Dict]:
    """
    Generate folder structure JSON.
    """
    # Existing implementation...

def write_structure_to_file(structure: List[Dict], output_file: str):
    """
    Write folder structure JSON to a file.
    """
    with open(output_file, "w") as f:
        json.dump(structure, f, indent=4)

def generate_directory_structure(root_folder: str, output_file: str):
    """
    Generate and write folder structure JSON.
    """
    structure = get_folder_structure(root_folder)
    write_structure_to_file(structure, output_file)
    
    
def load_structure(filepath: str) -> List[Dict]:
    """Load folder structure JSON from file"""
    with open(filepath, 'r') as f:
        return json.load(f)
# if __name__ == "__main__":
#     root_folder = r"C:\path\to\folder"
#     output_file = "directory_structure.json"

#     generate_directory_structure(root_folder, output_file)




filepath = 'directory_structure.json'
structure = load_structure(filepath)


