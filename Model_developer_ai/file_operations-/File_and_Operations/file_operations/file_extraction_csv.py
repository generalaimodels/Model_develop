import os
import csv
from typing import List

def get_file_paths(folder: str, extensions: List[str]) -> List[List[str]]:
    """
    Recursively get lists of paths for files with given extensions.

    Args:
        folder: Root folder path to search.
        extensions: List of file extensions to search for.

    Returns: 
        A list containing a list of paths for each file extension.
    """

    path_lists = {ext: [] for ext in extensions}

    for root, dirs, files in os.walk(folder):
        for file in files:
            name, ext = os.path.splitext(file)
            if ext in extensions:
                path = os.path.join(root, file).replace('\\', '/')
                path_lists[ext].append(path)

    return [path_lists[ext] for ext in extensions]


def save_to_csv(file_paths: List[List[str]], csv_file: str) -> None:
    """
    Save lists of file paths to a CSV file.

    Args:
        file_paths: A list containing a list of paths for each file type.
        csv_file: Path to the CSV file to save.
    
    if __name__ == '__main__':
    folder = 'C:/Users/hemanthk.LAP53-FJS.000/OneDrive/Desktop/hemanth/Hemanth' 
    extensions = ['.pdf','.png',]
    csv_file = 'file_paths.csv'

    file_paths = get_file_paths(folder, extensions)
    save_to_csv(file_paths, csv_file)
    """

    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in zip(*file_paths):
            writer.writerow(row)



