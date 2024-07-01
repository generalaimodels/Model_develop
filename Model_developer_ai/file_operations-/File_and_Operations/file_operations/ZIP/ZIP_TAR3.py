import os
import json
import requests
from typing import Any, Dict
from tqdm import tqdm

# Importing for handling archives
from zipfile import ZipFile
import tarfile

class LargeFileManager:
    def __init__(self, url: str, dest_folder: str = 'downloads'):
        self.url = url
        self.dest_folder = dest_folder
        self.filename = os.path.basename(self.url)
        self.filepath = os.path.join(self.dest_folder, self.filename)

    def download(self) -> str:
        """Downloads a file from a URL in chunks, with the ability to resume."""
        if not os.path.exists(self.dest_folder):
            os.makedirs(self.dest_folder)

        headers = {}
        if os.path.exists(self.filepath):
            # Get the size of the already downloaded file for resuming
            headers = {'Range': f'bytes={os.path.getsize(self.filepath)}-'}

        # Create a stream request to download the file in chunks
        with requests.get(self.url, headers=headers, stream=True) as r:
            r.raise_for_status()

            # Get total file size for tqdm progress bar
            total_size = int(r.headers.get('content-length', 0)) + os.path.getsize(self.filepath)
            with open(self.filepath, 'ab') as f, tqdm(
                desc=self.filename, total=total_size, unit='B', unit_scale=True, unit_divisor=1024
            ) as bar:
                for chunk in r.iter_content(chunk_size=1024):
                    f.write(chunk)
                    bar.update(len(chunk))
        return self.filepath

    def extract(self) -> None:
        """Extracts downloaded .zip or .tar files."""
        if self.filepath.endswith('.zip'):
            with ZipFile(self.filepath, 'r') as zip_ref:
                zip_ref.extractall(self.dest_folder)
        elif self.filepath.endswith(('.tar', '.tar.gz', '.tgz')):
            with tarfile.open(self.filepath, 'r:*') as tar_ref:
                tar_ref.extractall(self.dest_folder)
        else:
            raise ValueError("Unsupported archive format")

    @staticmethod
    def preprocess(folder_path: str) -> Dict[str, Any]:
        """Placeholder for preprocessing the data inside the extracted folders."""
        # TODO: Implement actual preprocessing logic based on folder structure
        processed_data = {}
        # Example: Read file names and paths
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                processed_data[file] = os.path.join(root, file)
        return processed_data

    @staticmethod
    def save_to_json(data: Dict[str, Any], output_path: str) -> None:
        """Saves the processed data to a JSON file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

def main():
    # URL to the .zip or .tar file
    archive_url = "http://example.com/largefile.zip"  # Replace with the actual URL

    # Initialize the LargeFileManager
    manager = LargeFileManager(archive_url)

    # Download the file with resume capability
    print("Starting download...")
    filepath = manager.download()
    print("Download completed.")

    # Extract the contents of the archive
    print("Extracting files...")
    manager.extract()
    print("Extraction completed.")

    # Preprocess the extracted data and save to JSON
    extracted_folder_path = os.path.splitext(filepath)[0]
    print("Preprocessing data...")
    processed_data = manager.preprocess(extracted_folder_path)
    json_output_path = os.path.join(manager.dest_folder, 'processed_data.json')
    manager.save_to_json(processed_data, json_output_path)
    print(f"Processed data saved to {json_output_path}.")

if __name__ == '__main__':
    main()