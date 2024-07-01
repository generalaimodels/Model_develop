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
                 txt_paths: List[str], csv_paths: List[str], jpg_paths: List[str],
                 jpeg_paths: List[str], gif_paths: List[str], bmp_paths: List[str],
                 tiff_paths: List[str], tif_paths: List[str], webp_paths: List[str],
                 svg_paths: List[str], raw_paths: List[str], nef_paths: List[str],
                 cr2_paths: List[str], ico_paths: List[str], heif_paths: List[str],
                 heic_paths: List[str], jp2_paths: List[str], dds_paths: List[str]):
    """
    Write file paths to a CSV file with columns for each file type.
    """
    with open(csv_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['PNG', 'PDF', 'MD', 'PY', 'JSON', 'TXT', 'CSV', 'JPG', 'JPEG', 'GIF', 'BMP', 'TIFF', 'TIF', 'WEBP', 'SVG', 'RAW', 'NEF', 'CR2', 'ICO', 'HEIF', 'HEIC', 'JP2', 'DDS'])

        # Calculate the maximum number of rows needed.
        max_rows = max(len(png_paths), len(pdf_paths), len(md_paths), len(py_paths),
                       len(json_paths), len(txt_paths), len(csv_paths), len(jpg_paths),
                       len(jpeg_paths), len(gif_paths), len(bmp_paths), len(tiff_paths),
                       len(tif_paths), len(webp_paths), len(svg_paths), len(raw_paths),
                       len(nef_paths), len(cr2_paths), len(ico_paths), len(heif_paths),
                       len(heic_paths), len(jp2_paths), len(dds_paths))

        for i in range(max_rows):
            row = [
                png_paths[i] if i < len(png_paths) else '',
                pdf_paths[i] if i < len(pdf_paths) else '',
                md_paths[i] if i < len(md_paths) else '',
                py_paths[i] if i < len(py_paths) else '',
                json_paths[i] if i < len(json_paths) else '',
                txt_paths[i] if i < len(txt_paths) else '',
                csv_paths[i] if i < len(csv_paths) else '',
                jpg_paths[i] if i < len(jpg_paths) else '',
                jpeg_paths[i] if i < len(jpeg_paths) else '',
                gif_paths[i] if i < len(gif_paths) else '',
                bmp_paths[i] if i < len(bmp_paths) else '',
                tiff_paths[i] if i < len(tiff_paths) else '',
                tif_paths[i] if i < len(tif_paths) else '',
                webp_paths[i] if i < len(webp_paths) else '',
                svg_paths[i] if i < len(svg_paths) else '',
                raw_paths[i] if i < len(raw_paths) else '',
                nef_paths[i] if i < len(nef_paths) else '',
                cr2_paths[i] if i < len(cr2_paths) else '',
                ico_paths[i] if i < len(ico_paths) else '',
                heif_paths[i] if i < len(heif_paths) else '',
                heic_paths[i] if i < len(heic_paths) else '',
                jp2_paths[i] if i < len(jp2_paths) else '',
                dds_paths[i] if i < len(dds_paths) else ''
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
    jpg_paths = get_file_paths_by_ext(all_files, '.jpg')
    jpeg_paths = get_file_paths_by_ext(all_files, '.jpeg')
    gif_paths = get_file_paths_by_ext(all_files, '.gif')
    bmp_paths = get_file_paths_by_ext(all_files, '.bmp')
    tiff_paths = get_file_paths_by_ext(all_files, '.tiff')
    tif_paths = get_file_paths_by_ext(all_files, '.tif')
    webp_paths = get_file_paths_by_ext(all_files, '.webp')
    svg_paths = get_file_paths_by_ext(all_files, '.svg')
    raw_paths = get_file_paths_by_ext(all_files, '.raw')
    nef_paths = get_file_paths_by_ext(all_files, '.nef')
    cr2_paths = get_file_paths_by_ext(all_files, '.cr2')
    ico_paths = get_file_paths_by_ext(all_files, '.ico')
    heif_paths = get_file_paths_by_ext(all_files, '.heif')
    heic_paths = get_file_paths_by_ext(all_files, '.heic')
    jp2_paths = get_file_paths_by_ext(all_files, '.jp2')
    dds_paths = get_file_paths_by_ext(all_files, '.dds')
    write_to_csv(output_csv, png_paths, pdf_paths, md_paths, py_paths, json_paths, txt_paths, csv_paths, jpg_paths, jpeg_paths, gif_paths, bmp_paths, tiff_paths, tif_paths, webp_paths, svg_paths, raw_paths, nef_paths, cr2_paths, ico_paths, heif_paths, heic_paths, jp2_paths, dds_paths)


            


DOCS_DIR = "C:/Users/heman/Desktop/Hemanth/Hemanth_LLMs/Deep_learning/machine-learning-book"
extract_files(input_folder=DOCS_DIR, output_csv='testing.csv')