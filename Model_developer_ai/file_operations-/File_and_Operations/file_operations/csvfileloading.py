import csv
from typing import List
from datasets import load_dataset, DatasetDict
import csv
from typing import List

def read_csv_file_paths(file_path: str) -> List[str]:
    """
    Reads a CSV file and returns a list of file paths.
    Automatically detects if there is a header and identifies the column with the file paths.

    Args:
        file_path (str): The path to the CSV file containing the list of file paths.

    Returns:
        List[str]: A list of file paths.
    """
    with open(file_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        rows = list(reader)
        
        # Check if the first row is a header or not
        # This can be a heuristic, such as checking for file extension in the first row
        # If the first item doesn't look like a file path, assume it's a header
        if rows and not rows[0][0].strip().lower().endswith('.csv'):
            # If there is a header, remove the first row
            header = rows.pop(0)
        else:
            # No header present
            header = None
        
        # Detect the column with the file paths
        # If we don't have a header, we can either make an assumption or prompt the user
        if header:
            # If there's a header, we could assume the first column with a string like 'path' is the right one
            # Here we're just using the first column by default
            file_path_index = 0
        else:
            # No header, so we're assuming the first column contains the file paths
            # Alternatively, you could prompt the user or use a heuristic
            file_path_index = 0
        
        # Compile the list of file paths
        file_paths = [row[file_path_index] for row in rows if len(row) > file_path_index]
        
    return file_paths

from datasets import load_dataset, DatasetDict

def load_datasets_from_csv(file_paths: List[str]) -> DatasetDict:
    """
    Loads multiple datasets from a list of CSV file paths, handling encoding errors.

    Args:
        file_paths (List[str]): The list of file paths to CSV files.

    Returns:
        DatasetDict: A dictionary of datasets indexed by file path.
    """
    datasets = {}
    for file_path in file_paths:
        try:
            # Attempt to load each dataset using UTF-8 encoding
            datasets[file_path] = load_dataset('csv', data_files=file_path)
        except UnicodeDecodeError as e:
            print(f"UnicodeDecodeError for file {file_path}: {e}")
            try:
                # Attempt to load using 'latin1' encoding
                datasets[file_path] = load_dataset('csv', data_files=file_path, encoding='latin1')
            except Exception as e:
                # If there's still an error, skip this file and continue with others
                print(f"Failed to load file {file_path} with alternate encoding: {e}")
    return DatasetDict(datasets)

# Example usage
if __name__ == "__main__":
    # Path to the CSV file containing the list of CSV file paths
    csv_list_file_path = 'C:/Users/heman/Desktop/Deep learning/file_paths.csv'
    # Read the CSV file paths
    csv_file_paths = read_csv_file_paths(csv_list_file_path)
    # Load the datasets from the CSV file paths
    all_datasets = load_datasets_from_csv(csv_file_paths)

    # Now, you can work with `all_datasets` which contains all the loaded datasets
    # Example: print the first few entries of each dataset
    for path, dataset in all_datasets.items():
        print(f"Dataset from {path}:")
        print(dataset['train'][:5])  # Adjust the slice as needed or process the data as required