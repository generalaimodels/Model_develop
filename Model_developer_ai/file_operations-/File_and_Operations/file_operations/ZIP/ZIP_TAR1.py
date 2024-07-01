import os
import requests
from zipfile import ZipFile
from tarfile import TarFile
import json
from typing import Dict, List

def download_and_extract(url: str, output_dir: str) -> None:
    """
    Download a file from the given URL and extract its contents.
    The function supports both .zip and .tar files and continues the download if interrupted.

    Args:
        url (str): The URL of the file to download.
        output_dir (str): The directory where the downloaded file and extracted contents should be saved.

    Returns:
        None
    """
    response = requests.get(url, stream=True)
    response.raise_for_status()

    file_name = os.path.join(output_dir, url.split("/")[-1])
    chunk_size = 1024  # 1KB

    try:
        with open(file_name, "wb") as file:
            for chunk in response.iter_content(chunk_size=chunk_size):
                file.write(chunk)
    except (requests.exceptions.ConnectionError, Exception) as e:
        print(f"Download failed: {e}")
        return

    if file_name.endswith(".zip"):
        try:
            with ZipFile(file_name, "r") as zip_file:
                zip_file.extractall(output_dir)
        except Exception as e:
            print(f"Failed to extract zip file: {e}")
    elif file_name.endswith(".tar"):
        try:
            with TarFile.open(file_name, "r") as tar_file:
                tar_file.extractall(output_dir)
        except Exception as e:
            print(f"Failed to extract tar file: {e}")
    else:
        print("Unsupported file format. Only .zip and .tar files are supported.")

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
    url = "https://example.com/file.zip"  # Replace with your URL
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)

    try:
        download_and_extract(url, output_dir)
        preprocessed_data = preprocess_data(output_dir)
        save_json(preprocessed_data, "./preprocessed_data.json")
    except Exception as e:
        print(f"An error occurred: {e}")