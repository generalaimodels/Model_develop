import os
import shutil
from typing import List
from datetime import datetime
import logging



logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def find_files(directory: str, extensions: List[str]) -> List[str]:
    """
    Recursively finds all files in the given directory that match the given file extensions.
    """
    matches = []
    try:
        for root, _, files in os.walk(directory):
            for file in files:
                if any(file.lower().endswith(ext) for ext in extensions):
                    matches.append(os.path.join(root, file))
    except Exception as e:
        logging.error(f"Error while finding files: {e}")
    return matches

def move_file_with_timestamp(src_path: str, dest_path: str):
    """
    Moves a file to the destination path, appending a timestamp if a file with the same name exists.
    """
    try:
        if os.path.exists(dest_path):
            base_name, file_extension = os.path.splitext(dest_path)
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            dest_path = f"{base_name}_{timestamp}{file_extension}"
        shutil.move(src_path, dest_path)
    except Exception as e:
        logging.error(f"Error while moving file '{src_path}' to '{dest_path}': {e}")

def move_files(files: List[str], output_dir: str):
    """
    Moves the given files into the specified output directory.
    If a file with the same name already exists, it appends a timestamp to the filename.
    """
    for file in files:
        try:
            os.makedirs(output_dir, exist_ok=True)
            file_extension = file.split('.')[-1]
            dest_dir = os.path.join(output_dir, file_extension)
            os.makedirs(dest_dir, exist_ok=True)
            
            base_name = os.path.basename(file)
            dest_file_path = os.path.join(dest_dir, base_name)
            
            move_file_with_timestamp(file, dest_file_path)
        except Exception as e:
            logging.error(f"Error while processing file '{file}': {e}")

def organize_files(input_dir: str, output_dir: str,extensions: List[str]):
    """
    Organizes files by their extension from the input directory into the output directory.
    organize_files('/content/drive/MyDrive', '/content/hemanth')
    extensions = ['.pdf', '.txt', '.md', '.docx', '.py', '.csv', '.png']
    """
    try:
        if not os.path.isdir(input_dir):
            logging.error(f"Input directory '{input_dir}' does not exist or is not a directory.")
            return
        if not os.path.isdir(output_dir):
            logging.error(f"Output directory '{output_dir}' does not exist or is not a directory.")
        else:
            logging.info(f"Output directory '{output_dir}' Automatic created .")
            
            files_to_move = find_files(input_dir, extensions)
            move_files(files_to_move, output_dir)
            logging.info(f"completed moving files into folders {output_dir} 📁 in organized")
            return
    except Exception as e:
        logging.error(f"Error while organizing files: {e}")

