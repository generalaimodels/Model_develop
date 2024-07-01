import os
import glob
from typing import List, Tuple
from datasets import Dataset

from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_files_by_extension(folder_path: str, extensions: List[str]) -> List[str]:
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(folder_path, f"**/*.{ext}"), recursive=True))
    return files

def split_documents(chunk_size: int, documents: List[str]) -> List[Tuple[str, str]]:
    
    separators = [
        "\n#{1,6} ",
        "```\n",
        "\n\\*\\*\\*+\n",
        "\n---+\n",
        "\n___+\n",
        "\n\n",
        "\n",
        " ",
        "",
    ]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size / 10),
        add_start_index=True,
        strip_whitespace=True,
        separators=separators,
    )

    processed_docs = []
    for doc_path in documents:
        with open(doc_path, "r") as f:
            content = f.read()
            chunks = text_splitter.split_content(content)
            for chunk in chunks:
                processed_docs.append((doc_path, chunk))

    return processed_docs

def main():
    folderpath = "/content/research_papers-/papers"
    extensions = ["pdf", "py", "csv", "json", "txt", "md"]
    documents = []

    for ext in extensions:
        documents.extend(load_files_by_extension(folderpath, [ext]))

    for doc in documents:
        loader = DirectoryLoader(folderpath, glob=f"**/*.{ext}", show_progress=True, use_multithreading=True, silent_errors=True)
        docs = loader.load()

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