import os
import json
import requests
from typing import Dict
from tqdm import tqdm
from urllib.parse import urlparse
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Importing for handling archives
from zipfile import ZipFile
import tarfile

# Function to download a file with resuming capability
def download_file(url: str, dest_folder: str) -> str:
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)  # Create destination folder if it does not exist
    
    filename = os.path.basename(urlparse(url).path)
    file_path = os.path.join(dest_folder, filename)
    
    # Start the session
    session = requests.Session()
    retry = Retry(connect=3, backoff_factor=0.5)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)

    response = session.get(url, stream=True)
    response.raise_for_status()  # Ensure we raise an error for bad statuses

    # Stream the download
    with open(file_path, 'ab') as file, tqdm(
        desc=filename,
        total=int(response.headers.get('content-length', 0)),
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)
    
    return file_path

# Function to extract the archive
def extract_archive(file_path: str) -> str:
    if file_path.endswith('.zip'):
        with ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(path=os.path.splitext(file_path)[0])
            return os.path.splitext(file_path)[0]
    elif file_path.endswith('.tar') or file_path.endswith('.tar.gz') or file_path.endswith('.tgz'):
        with tarfile.open(file_path, 'r:*') as tar_ref:
            tar_ref.extractall(path=os.path.splitext(file_path)[0])
            return os.path.splitext(file_path)[0]
    else:
        raise ValueError("Unsupported archive format")

# Function to preprocess data within a folder
def preprocess_data(folder_path: str) -> Dict:
    # Assuming the folder contains files we want to preprocess
    # This function should be adapted to your data and preprocessing steps
    processed_data = {}
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            # Example preprocessing: just reading the file names
            processed_data[file] = {"path": file_path}
    
    return processed_data

# Function to save data to a JSON file
def save_to_json(data: Dict, file_path: str):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# Main function to orchestrate the download, extraction, and preprocessing
def from_zip_tar_main(url: str, dest_folder: str = 'downloads', json_output: str = 'data.json'):
    """
    Main function that handles downloading a .zip or .tar file from a given URL,
    extracts its contents, preprocesses the data inside, and saves the processed
    data into a JSON file.
    
    :param url: URL of the .zip or .tar file to download
    :param dest_folder: Destination folder where the downloaded file will be stored
    :param json_output: Filename for the output JSON file containing preprocessed data
    
    
         archive_url = "https://www.openslr.org/resources/12/test-other.tar.gz"  # Replace with your actual URL
         
         # Call the main function
         from_zip_tar_main(archive_url)
    
    """
    # Download the file
    print("Starting download...")
    file_path = download_file(url, dest_folder)
    print("Download finished.")
    
    # Extract the contents
    print("Extracting archive...")
    extraction_path = extract_archive(file_path)
    print(f"Extraction completed. Extracted to: {extraction_path}")
    
    # Preprocess the data
    print("Preprocessing data...")
    processed_data = preprocess_data(extraction_path)
    
    # Save to JSON
    json_path = os.path.join(dest_folder, json_output)
    save_to_json(processed_data, json_path)
    print(f"Preprocessed data saved to: {json_path}")
