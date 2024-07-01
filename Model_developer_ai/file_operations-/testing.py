import csv
import pandas as pd
from typing import List, Dict
from langchain_community.document_loaders import DirectoryLoader

NEW_COLUMN_NAMES=["id", "content", "source"]
def save_to_csv(file_path: str, data: List[Dict[str, str]], mode: str = 'a') -> None:
    """
    Save or append data to a CSV file.
    
    :param file_path: Path to the CSV file to save or append data.
    :param data: List of dictionaries containing the data to be saved.
    :param mode: File opening mode ('a' for append, 'w' for write).
    """
    with open(file_path, mode, newline='', encoding='utf-8') as csvfile:
        fieldnames = data[0].keys() if data else []
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if mode == 'w':
            writer.writeheader()
        
        for row in data:
            writer.writerow(row)
def rename_columns(csv_file: str, new_columns: List[str]) -> None:
    """
    This function renames the columns of a CSV file.

    Parameters:
    csv_file (str): The path to the CSV file.
    new_columns (List[str]): The new column names.

    Returns:
    None
    """
    # Load the CSV file
    df = pd.read_csv(csv_file)

    # Check if the number of new column names matches the number of columns in the CSV file
    if len(df.columns) != len(new_columns):
        raise ValueError("The number of new column names must match the number of columns in the CSV file.")

    # Rename the columns
    df.columns = new_columns

    # Save the dataframe to the same CSV file
    df.to_csv(csv_file, index=False)
def load_docs_from_folder(folder_path: str, file_glob: str) -> List[Dict[str, str]]:
    """
    Load documents from a folder with a specific file extension and return their content and metadata.
    
    :param folder_path: The path to the folder containing the files.
    :param file_glob: The file extension or pattern to filter files.
    :return: A list of dictionaries with file content and metadata.
    """
    directory_loader = DirectoryLoader(folder_path, glob=file_glob, show_progress=True,
                                       use_multithreading=True, silent_errors=True)
    return directory_loader.load()
def main(folder_path: str, csv_output_path: str) -> None:
   
    """
    Main function to load documents from a directory and save them into a CSV file.
    
    :param folder_path: The path to the folder containing the files.
    :param csv_output_path: The path to the output CSV file.
    if __name__ == "__main__":
    NEW_COLUMN_NAMES=["ID", "hemanth", "joking"]
    FOLDER_PATH="C:/Users/hemanthk.LAP53-FJS.000/OneDrive/Desktop/hemanth/Hemanth/file_operations-/File_and_Operations/file_operations/"
    OUTPUTFILE_PATH="file_operations.csv"
    folder_path = FOLDER_PATH
    csv_output_path = OUTPUTFILE_PATH
    main(folder_path, csv_output_path)
    """
    file_types = ['.md', '.pdf', '.py', '.csv','.txt']
    first_run = True
    doc_id = 1  # Initialize document ID

    for file_type in file_types:
        print(f"============================ *{file_type[1:]} files* ==============================")
        docs = load_docs_from_folder(folder_path, f"**/*{file_type}")
        
        if not docs:
            print(f"No {file_type[1:]} files found.")
            continue
        
        # Assuming that the Document object has 'page_content' and 'source' properties.
        data_to_save = [{'id': str(doc_id + i),  # Generate an incremental ID for each document.
                         'content': doc.page_content,
                         'source': doc.metadata['source']} for i, doc in enumerate(docs) if hasattr(doc, 'metadata')]
        
        # Update the document ID for the next batch of files
        doc_id += len(docs)
        
        # If it's the first run, write headers to the CSV, otherwise append without headers.
        save_to_csv(csv_output_path, data_to_save, 'w' if first_run else 'a')
        first_run = False
        if NEW_COLUMN_NAMES is not None:
            rename_columns(csv_output_path, NEW_COLUMN_NAMES)




