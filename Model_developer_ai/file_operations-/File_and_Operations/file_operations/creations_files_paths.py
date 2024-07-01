from typing import List, Tuple
import os
import csv

def get_all_files(input_folder: str) -> List[Tuple[str, str]]:
    """
    Get all files and directories from input folder and subfolders.
    Returns list of tuples (file_path, file_extension).
    """
    all_files = []
    for root, dirs, files in os.walk(input_folder):
        # Add directories with a trailing forward slash.
        for dir in dirs:
            dir_path = os.path.join(root, dir).replace('\\', '/') + '/'
            all_files.append((dir_path, ''))
        # Add files with forward slashes.
        for file in files:
            file_path = os.path.join(root, file).replace('\\', '/')
            file_ext = os.path.splitext(file)[1]
            all_files.append((file_path, file_ext))
    return all_files
def get_csv_files(all_files: List[Tuple[str, str]]) -> List[str]:
    """
    Get paths of all CSV files from the list of files.
    """
    csv_files = []
    for file_path, file_ext in all_files:
        if file_ext == '.csv':
            csv_files.append(file_path)
    return csv_files

def get_file_paths_by_ext(all_files: List[Tuple[str, str]], ext: str) -> List[str]:
    """
    Get paths of all files with given extension from the list of files.
    """
    matched_files = []
    for file_path, file_ext in all_files:
        if file_ext == ext:
            matched_files.append(file_path)
    return matched_files

def write_to_csv(csv_path: str, png_paths: List[str], pdf_paths: List[str], 
                 md_paths: List[str], py_paths: List[str], json_paths: List[str],
                 txt_paths: List[str], csv_paths: List[str]):
    """
    Write file paths to a CSV file with columns for each file type.
    """
    with open(csv_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['PNG', 'PDF', 'MD', 'PY', 'JSON', 'TXT', 'CSV'])
        
        # Calculate the maximum number of rows needed.
        max_rows = max(len(png_paths), len(pdf_paths), len(md_paths), len(py_paths),
                       len(json_paths), len(txt_paths), len(csv_paths))
        
        for i in range(max_rows):
            row = [
                png_paths[i] if i < len(png_paths) else '',
                pdf_paths[i] if i < len(pdf_paths) else '',
                md_paths[i] if i < len(md_paths) else '',
                py_paths[i] if i < len(py_paths) else '',
                json_paths[i] if i < len(json_paths) else '',
                txt_paths[i] if i < len(txt_paths) else '',
                csv_paths[i] if i < len(csv_paths) else ''
            ]
            writer.writerow(row)

def extract_files(input_folder: str, output_csv: str):
    """
    Main function to extract file paths by type and write to CSV.
    """
    all_files = get_all_files(input_folder)
    
    png_paths = get_file_paths_by_ext(all_files, '.png')
    pdf_paths = get_file_paths_by_ext(all_files, '.pdf')
    md_paths = get_file_paths_by_ext(all_files, '.md')
    py_paths = get_file_paths_by_ext(all_files, '.py')
    json_paths = get_file_paths_by_ext(all_files, '.json')
    txt_paths = get_file_paths_by_ext(all_files, '.txt')
    csv_paths = get_file_paths_by_ext(all_files, '.csv')

    write_to_csv(output_csv, png_paths, pdf_paths, md_paths, py_paths, json_paths, txt_paths, csv_paths)

DOCS_DIR = 'C:/Users/hemanthk.LAP53-FJS.000/OneDrive/'
extract_files(input_folder=DOCS_DIR, output_csv='docs.csv')