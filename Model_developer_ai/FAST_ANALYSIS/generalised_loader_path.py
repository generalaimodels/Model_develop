import os
import logging
from typing import List
from PyPDF2 import PdfFileReader
from markdown import markdown
# from docx import Document
from pptx import Presentation
from csv import reader
from json import loads

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def convert_to_text(file_path: str, output_folder: str) -> None:
    """
    Convert the given file to text format and save it to the output folder.
    
    if __name__ == "__main__":
        files_to_convert = ['example.gif', 'example.md', 'example.docx', 'example.pdf', 'example.csv', 'example.json']
        output_directory = 'converted_files'
        os.makedirs(output_directory, exist_ok=True)
        batch_conversion(files_to_convert, output_directory)
    """
    try:
        file_name, file_extension = os.path.splitext(file_path)
        file_name = os.path.basename(file_name)
        output_path = os.path.join(output_folder, f"{file_name}.txt")

        if file_extension.lower() in ['.gif', '.ico', '.jpg', '.png']:
            logging.error(f"File format {file_extension} cannot be converted to text.")
            return

        with open(output_path, 'w', encoding='utf-8') as output_file:
            if file_extension.lower() == '.md':
                with open(file_path, 'r', encoding='utf-8') as md_file:
                    content = markdown(md_file.read())
                    output_file.write(content)
            elif file_extension.lower() == '.docx':
                doc = Document(file_path)
                for para in doc.paragraphs:
                    output_file.write(para.text + '\n')
            elif file_extension.lower() == '.pdf':
                pdf = PdfFileReader(file_path)
                for page_num in range(pdf.numPages):
                    page = pdf.getPage(page_num)
                    text = page.extractText()
                    output_file.write(text + '\n')
            elif file_extension.lower() in ['.csv', '.json',".yml",".txt"]:
                if file_extension.lower() == '.csv':
                    data_reader = reader(open(file_path))
                elif file_extension.lower() == '.json':
                    data_reader = loads(open(file_path).read())
                elif file_extension.lower() == '.yml':
                     data_reader = loads(open(file_path).read())
                elif file_extension.lower() == '.txt':
                     data_reader = loads(open(file_path).read())
                for row in data_reader:
                    output_file.write(str(row) + '\n')
            else:
                logging.warning(f"File format {file_extension} is not supported yet.")
                return

        logging.info(f"File {file_path} has been converted to text and saved to {output_path}")
    except Exception as e:
        logging.error(f"An error occurred while converting {file_path}: {e}")

def batch_conversion(file_list: List[str], output_folder: str) -> None:
    """
    Convert a batch of files to text format.
    """
    for file_path in file_list:
        if os.path.isfile(file_path):
            convert_to_text(file_path, output_folder)
        else:
            logging.error(f"The provided path {file_path} is not a file.")
