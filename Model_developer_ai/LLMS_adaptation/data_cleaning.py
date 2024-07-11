import csv
import logging
from typing import List, Dict, Union
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def read_csv(file_path: Path) -> List[Dict[str, str]]:
    """
    Read the input CSV file and return its contents as a list of dictionaries.
    
    Args:
        file_path (Path): Path to the input CSV file.
    
    Returns:
        List[Dict[str, str]]: List of dictionaries containing the CSV data.
    """
    try:
        with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            return list(reader)
    except FileNotFoundError:
        logger.error(f"Input file not found: {file_path}")
        raise
    except csv.Error as e:
        logger.error(f"Error reading CSV file: {e}")
        raise

def process_data(data: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Process the input data and generate the output data.
    
    Args:
        data (List[Dict[str, str]]): Input data as a list of dictionaries.
    
    Returns:
        List[Dict[str, str]]: Processed output data.
    """
    output_data = []
    for row in data:
        content = ','.join(row.values())
        prompt = '1' if row.get('isFraud', '0') == '1' else '0'
        output_data.append({'content': content, 'prompt': prompt})
    return output_data

def write_csv(file_path: Path, data: List[Dict[str, str]]) -> None:
    """
    Write the processed data to the output CSV file.
    
    Args:
        file_path (Path): Path to the output CSV file.
        data (List[Dict[str, str]]): Processed data to be written.
    """
    try:
        with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['content', 'prompt']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        logger.info(f"Output file created successfully: {file_path}")
    except IOError as e:
        logger.error(f"Error writing to CSV file: {e}")
        raise

def main(input_file: Path, output_file: Path) -> None:
    """
    Main function to orchestrate the CSV processing.
    
    Args:
        input_file (Path): Path to the input CSV file.
        output_file (Path): Path to the output CSV file.
    """
    try:
        logger.info("Starting CSV processing")
        input_data = read_csv(input_file)
        processed_data = process_data(input_data)
        write_csv(output_file, processed_data)
        logger.info("CSV processing completed successfully")
    except Exception as e:
        logger.exception(f"An error occurred during processing: {e}")

main(
    input_file=r"C:\Users\heman\Desktop\Coding\data\Synthetic_Financial_datasets_log.csv\Synthetic_Financial_datasets_log.csv",
    output_file='Synthetic_Financial_datasets_log.csv1'
)