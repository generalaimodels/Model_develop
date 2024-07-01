
import os
import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from zipfile import ZipFile
from tarfile import TarFile
import json
from typing import Dict, List
from tqdm import tqdm  # Progress bar for better user experience

def download_large_file(url: str, output_file: str, chunk_size: int = 1024 * 1024) -> None:
    """
    Download a large file from the given URL with resuming capabilities.

    Args:
        url (str): The URL of the file to download.
        output_file (str): The file path to save the downloaded file.
        chunk_size (int, optional): The size of each chunk to read and write. Defaults to 1MB.

    Returns:
        None
    """
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=10, status_forcelist=[ 500, 502, 503, 504 ])
    session.mount('http://', HTTPAdapter(max_retries=retries))
    session.mount('https://', HTTPAdapter(max_retries=retries))

    headers = {}
    if os.path.exists(output_file):
        headers["Range"] = f"bytes={os.path.getsize(output_file)}-"

    response = session.get(url, headers=headers, stream=True)
    response.raise_for_status()

    with open(output_file, "ab") as file:
        for chunk in tqdm(response.iter_content(chunk_size=chunk_size), total=int(response.headers.get("content-length", 0)) // chunk_size):
            file.write(chunk)

def extract_large_file(input_file: str, output_dir: str) -> None:
    """
    Extract a large .zip or .tar file to the given directory.

    Args:
        input_file (str): The path to the .zip or .tar file to extract.
        output_dir (str): The directory where the extracted contents should be saved.

    Returns:
        None
    """
    if input_file.endswith(".zip"):
        with ZipFile(input_file, "r") as zip_file:
            zip_file.extractall(output_dir)
    elif input_file.endswith(".tar"):
        with TarFile.open(input_file, "r") as tar_file:
            tar_file.extractall(output_dir)
    else:
        raise ValueError("Unsupported file format. Only .zip and .tar files are supported.")

def preprocess_data(directory: str) -> List[Dict]:
    """
    Preprocess data from the given directory, assuming a specific folder structure.
    This function is a placeholder; replace it with your actual preprocessing logic.

    Args:
        directory (str): The directory containing the data to preprocess.

    Returns:
        List[Dict]: A list of dictionaries, each representing a preprocessed data item.
    """
    # Replace with your actual data preprocessing logic
    data_items = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Process each file and add it to the data_items list
            pass
    return data_items

def save_json(data: List[Dict], output_file: str) -> None:
    """
    Save the given data as a JSON file.

    Args:
        data (List[Dict]): The list of dictionaries to save as a JSON file.
        output_file (str): The file path to save the JSON data.

    Returns:
        None
    """
    with open(output_file, "w") as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    url = "https://example.com/large_file.zip"  # Replace with your URL
    output_file = "./large_file"
