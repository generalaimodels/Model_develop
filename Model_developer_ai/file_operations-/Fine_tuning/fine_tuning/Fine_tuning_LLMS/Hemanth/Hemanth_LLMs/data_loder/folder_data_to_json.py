import json
from typing import List, Dict
from langchain_community.document_loaders import DirectoryLoader

def create_json_file(folder_path: str, output_file: str) -> None:
    """
    Load all files from the given folder and create a JSON file.

    Args:
        folder_path (str): Path to the folder containing the files.
        output_file (str): Path to the output JSON file.
    """
    # Load all files
    loader = DirectoryLoader(folder_path, glob="**/[!.]*", show_progress=True, use_multithreading=True, silent_errors=True)
    documents = loader.load()

    # Convert documents to a list of dictionaries
    data: List[Dict[str, str]] = []
    for doc in documents:
        data.append({
            'page_content': doc.page_content,
            'metadata': str(doc.metadata)
        })

    # Write to the JSON file
    with open(output_file, 'w') as json_file:
        json.dump(data, json_file, default=str)

if __name__ == "__main__":
    folder_path = 'C:/Users/heman/Desktop/Deep learning/Hemanth_LLMs/data_loder/'
    output_file = 'C:/Users/heman/Desktop/Deep learning/Hemanth_LLMs/output.json'
    create_json_file(folder_path, output_file)