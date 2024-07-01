import csv
import os
from typing import List
from langchain_community.document_loaders import DirectoryLoader
def write_to_csv(file_path: str, data: dict, write_header: bool) -> None:
    """
    Function to append data into a CSV file.
    
    Args:
    file_path (str): The path to the CSV file.
    data (dict): The data to be appended into the CSV file.
    write_header (bool): Whether to write the header.
    """
    mode = 'a' if os.path.exists(file_path) else 'w'
    with open(file_path, mode, newline='', encoding='UTF-8', errors='ignore') as file:
        writer = csv.DictWriter(file, fieldnames=["content", "documents", "metasource"])
        if write_header:
            writer.writeheader()
        try:
            writer.writerow({k: data[k] for k in ["content", "documents", "metasource"]})
        except UnicodeEncodeError:
            print(f"Warning: UnicodeEncodeError encountered for file {data['documents']}. Skipping this file.")

def read_pdfs_from_folder(folder_path: str, csv_file_path: str) -> None:
    """
    Function to recursively read PDF files from a folder and its subfolders and extract their content.
    
    Args:
    folder_path (str): The path to the folder containing the PDF files.
    csv_file_path (str): The path to the CSV file.
    """
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            if file_name.endswith(".pdf"):
                full_file_path = os.path.join(root, file_name)
                loader = DirectoryLoader(full_file_path)
                pages = loader.load_and_split()
                for page in pages:
                    data = {
                        "content": page.page_content,
                        "documents": file_name,
                        "metasource": page.metadata['source']
                    }
                    write_to_csv(csv_file_path, data, file_name == os.listdir(root)[0])

# Usage
                
folder_path = "C:/Users/heman/Desktop/Deep learning/research_papers-/"
csv_file_path = "output.csv"
read_pdfs_from_folder(folder_path, csv_file_path)
                    

