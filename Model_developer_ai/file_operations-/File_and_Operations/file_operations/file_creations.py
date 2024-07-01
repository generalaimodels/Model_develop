import os
import json
from typing import List, Dict, Any
from pathlib import Path



def save_to_json(data: List[Dict[str, Any]], filename: str) -> None:
    """
    Saves the provided data to a JSON file with the given filename.
    
    :param data: The data to be saved in JSON format.
    :param filename: The name of the file to save the JSON data to.
    """
    with open(filename, 'w') as json_file:
        json.dump(data, json_file, indent=4)
def get_extensions_and_paths(directory: str) -> List[Dict[str, Any]]:
    """
    Traverses the directory tree starting at the given directory and compiles a list of 
    dictionaries, each containing folder names, their file extensions, and file paths.
    
    :param directory: The root directory from which to start the folder traversal.
    :return: A list of dictionaries with folder names, extensions, and file paths.
    
    
    # Example usage
    if __name__ == "__main__":
        try:
            directory_to_scan = "C:/Users/hemanthk.LAP53-FJS.000/OneDrive/Desktop/hemanth/Hemanth"
            folder_data = get_extensions_and_paths(directory_to_scan)
            json_filename = "folder_structure.json"
            save_to_json(folder_data, json_filename)
        except Exception as e:
            print(f"An error occurred: {e}")
    """
    folder_structure = []
    root_path = Path(directory)
    for root, dirs, files in os.walk(root_path):
        folder_name = os.path.basename(root)
        folder_info = {
            "folder": folder_name,
            "extensions": set(),
            "files": {}
        }
        for file in files:
            file_path = Path(root) / file
            extension = file_path.suffix
            if extension:
                folder_info["extensions"].add(extension)
                # Use as_posix() to convert the path to a string with forward slashes
                folder_info["files"].setdefault(extension, []).append(file_path.as_posix())
        
        # Convert the set of extensions to a sorted list
        folder_info["extensions"] = sorted(list(folder_info["extensions"]))
        folder_structure.append(folder_info)

    return folder_structure




# # Read the JSON data from a file
# with open("folder_structure.json") as f:
#     data = json.load(f)

# for i in range(len(data)):
    
#   print(data[i]['extensions'])






