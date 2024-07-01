import os
import zipfile
from typing import List

def extract_zip_files(directory: str) -> List[str]:
    """
    This function takes a directory path as input, traverses through the directory and its subdirectories,
    and extracts all .zip files found.

    Args:
        directory (str): The directory path.

    Returns:
        List[str]: A list of the paths of the extracted files.
    """

    extracted_files = []

    # Traverse the directory
    for foldername, subfolders, filenames in os.walk(directory):
        for filename in filenames:
            # Check if the file is a .zip file
            if filename.endswith('.zip'):
                file_path = os.path.join(foldername, filename)

                # Open the .zip file
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    # Extract all the contents of the .zip file in current directory
                    zip_ref.extractall(foldername)
                    extracted_files.append(file_path)

    return extracted_files

# Example usage:
directory = r"C:\Users\hemanthk.LAP53-FJS.000\OneDrive\Desktop\hemanth\Hemanth\Experiments"
extracted_files = extract_zip_files(directory)
print(f"Extracted files: {extracted_files}")
