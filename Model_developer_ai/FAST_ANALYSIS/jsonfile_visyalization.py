import json
import logging
import yaml
import requests
import argparse
import os
import time
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from typing import Any, Union, List, Dict
from pathlib import Path
from datasets import DatasetDict, Dataset
from plotly.subplots import make_subplots
from PIL import Image
from textwrap import wrap

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def save_plots(dataset: Dataset, folder: str, grid_size: int) -> None:
    """
    Saves image plots in a specified folder, arranged in a grid.

    This function iterates through a dataset and creates HTML plots of the 'img' 
    feature, assuming 'img' is a 3D array representing an image (channels, height, width). 
    The plots are arranged in a grid defined by 'grid_size'. Each grid is 
    saved as a separate HTML file.

    Args:
        dataset (Dataset): The dataset containing the images to plot. 
                          It's assumed the dataset has an 'img' feature containing 3D image arrays.
        folder (str): The directory where the plots will be saved.
        grid_size (int): The number of rows and columns in each plot grid. 
                         For example, a grid_size of 5 will create a 5x5 grid.

    Raises:
        TypeError: If 'dataset' is not a Datasets.Dataset object.
        TypeError: If 'folder' is not a string.
        TypeError: If 'grid_size' is not an integer.
        ValueError: If 'grid_size' is less than 1.

    Example:
        >>> from datasets import load_dataset
        >>> dataset = load_dataset("cifar10", split="train[:100]") # Load 100 samples
        >>> save_plots(dataset, "cifar10_plots", 10)  # Create plots in a 10x10 grid
    """

    # Input Validation
    if not isinstance(dataset, Dataset):
        raise TypeError("Expected 'dataset' to be a Datasets.Dataset object.")
    if not isinstance(folder, str):
        raise TypeError("Expected 'folder' to be a string.")
    if not isinstance(grid_size, int):
        raise TypeError("Expected 'grid_size' to be an integer.")
    if grid_size < 1:
        raise ValueError("Expected 'grid_size' to be greater than 0.")

    os.makedirs(folder, exist_ok=True)

    rows = dataset.num_rows
    num_plots = (rows + grid_size - 1) // grid_size  # Calculate the number of grids needed

    for plot_index in range(num_plots):
        fig = make_subplots(rows=grid_size, cols=grid_size, 
                            subplot_titles=[f"Plot {j + 1}" for j in range(grid_size**2)])

        for grid_row in range(grid_size):
            for grid_col in range(grid_size):
                data_index = plot_index * grid_size**2 + grid_row * grid_size + grid_col 
                if data_index < rows:
                    img = dataset[data_index]['img']

                    # Assuming 'img' is (C, H, W), convert to (H, W, C) for Plotly
                    img = np.transpose(img, axes=(1, 2, 0)) 

                    fig.add_trace(go.Image(z=img), row=grid_row + 1, col=grid_col + 1)

        plot_path = os.path.join(folder, f'grid_{plot_index + 1}.html')
        fig.write_html(plot_path)
        print(f"Saved {plot_path}")

def main_make_plots_datasets(dataset) -> None:


    start_time = time.time()
    try:
        grid_size = 2 
        for group, dataset in dataset.items():
            save_plots(dataset, os.path.join('plots', group), grid_size)

    except Exception as e:
        print(f"An error occurred: {e}")

    elapsed_time = time.time() - start_time
    print(f"Execution time: {elapsed_time:.8f}s")

def load_json(file_path: str) -> Union[Dict[str, Any], List[Any]]:
    """
    Load JSON data from a file.

    :param file_path: Path to the JSON file
    :return: Parsed JSON data
    """
    try:
        with open(file_path, 'r') as file:
            content = file.read().strip()
            if not content:
                logging.error(f"The JSON file {file_path} is empty.")
                raise ValueError("Empty JSON file")
            return json.loads(content)
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON in file {file_path}: {e}")
        raise
    except FileNotFoundError as e:
        logging.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred when loading JSON from file {file_path}: {e}")
        raise

def load_yaml(file_path: str) -> Union[Dict[str, Any], List[Any]]:
    """
    Load YAML data from a file.

    :param file_path: Path to the YAML file
    :return: Parsed YAML data
    """
    try:
        with open(file_path, 'r') as file:
            content = file.read().strip()
            if not content:
                logging.error(f"The YAML file {file_path} is empty.")
                raise ValueError("Empty YAML file")
            return yaml.safe_load(content)
    except yaml.YAMLError as e:
        logging.error(f"Error decoding YAML in file {file_path}: {e}")
        raise
    except FileNotFoundError as e:
        logging.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred when loading YAML from file {file_path}: {e}")
        raise

def fetch_data_from_url(url: str) -> Union[Dict[str, Any], List[Any]]:
    """
    Fetch data from a URL.

    :param url: URL to fetch data from
    :return: Parsed data from the URL
    """
    try:
        response = requests.get(url)
        response.raise_for_status()

        if not response.text.strip():
            logging.error("The URL returned an empty response.")
            raise ValueError("Empty response from URL")

        if url.endswith('.json'):
            return response.json()
        elif url.endswith('.yml') or url.endswith('.yaml'):
            return yaml.safe_load(response.text)
        else:
            logging.error("Unsupported file format.")
            raise ValueError("Unsupported file format.")
    except requests.RequestException as e:
        logging.error(f"Error fetching data from URL: {e}")
        raise
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from URL: {e}")
        raise
    except yaml.YAMLError as e:
        logging.error(f"Error decoding YAML from URL: {e}")
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        raise

def process_value(value: Any) -> str:
    """
    Process the value to ensure it is in a printable format. If the value is a long string, truncate it to 20 words.

    :param value: The value to process
    :return: Processed value as string
    """
    if isinstance(value, str):
        truncated_value = ' '.join(value.split()[:20])
        if len(value.split()) > 20:
            truncated_value += '...'
        return truncated_value
    return str(value)

def extract_data(data: Union[Dict[str, Any], List[Any]], parent_key: str = '', result: List[Dict[str, str]] = None) -> List[Dict[str, str]]:
    """
    Recursively extract keys and values from JSON/YAML data and store them in a list of dictionaries.

    :param data: JSON/YAML data to process
    :param parent_key: The parent key to help trace nested structures
    :param result: List to store the extracted data
    :return: List of dictionaries with key-value pairs
    """
    if result is None:
        result = []

    if isinstance(data, dict):
        for key, value in data.items():
            full_key = f"{parent_key}.{key}" if parent_key else key
            if isinstance(value, (dict, list)):
                extract_data(value, full_key, result)
            else:
                result.append({'Key': full_key, 'Value': process_value(value)})
    elif isinstance(data, list):
        for index, value in enumerate(data):
            full_key = f"{parent_key}[{index}]"
            if isinstance(value, (dict, list)):
                extract_data(value, full_key, result)
            else:
                result.append({'Key': full_key, 'Value': process_value(value)})

    return result

def wrap_text(text: str, width: int) -> List[str]:
    """
    Wrap the text to fit within the given width.

    :param text: The text to wrap
    :param width: The width to wrap the text within
    :return: List of wrapped text lines
    """
    words = text.split()
    lines = []
    current_line = []

    for word in words:
        if sum(len(w) for w in current_line) + len(current_line) + len(word) <= width:
            current_line.append(word)
        else:
            lines.append(' '.join(current_line))
            current_line = [word]

    if current_line:
        lines.append(' '.join(current_line))

    return lines

def format_table(data: List[Dict[str, str]]) -> str:
    """
    Format the extracted data into a professional, dynamically adjusted table.

    :param data: List of dictionaries with key-value pairs
    :return: Formatted table as a string
    """
    if not data:
        return "No data to display."
    column_widths = {}
    for row in data:
        for key, value in row.items():
            column_widths[key] = max(column_widths.get(key, 0), len(str(value)), len(key))
    for key, width in column_widths.items():
        column_widths[key] = max(width , 5)  # 4 for padding
    header = []
    separator = []
    for key, width in column_widths.items():
        header.append(f"{key.upper():^{width}}")
        separator.append("-" * width)
    table = ["|".join(header), "|".join(separator)]
    for row in data:
        row_lines = []
        for key, value in row.items():
            wrapped_value = wrap(str(value), width=column_widths[key])
            max_lines = max(len(wrapped_value), len(row_lines))
            row_lines = [line.ljust(column_widths[key]) if i < len(line) else " " * column_widths[key]
                          for i, line in enumerate(wrapped_value)]
            for i in range(max_lines - len(row_lines)):
                row_lines.append(" " * column_widths[key])
        for i in range(max_lines):
            row_data = []
            for key in row.keys():
                row_data.append(row_lines[i])
            table.append("|".join(row_data))

    return "\n".join(table)

def visualize_data(file_path_or_url: str) -> None:
    """
    Main function to handle user input and initiate JSON/YAML processing.

    :param file_path_or_url: Path to the JSON/YAML file or URL
    """
    try:
        if file_path_or_url.startswith(('http://', 'https://')):
            data = fetch_data_from_url(file_path_or_url)
        else:
            path = Path(file_path_or_url)
            if not path.is_file():
                logging.error(f"The file path provided does not exist: {file_path_or_url}")
                return

            if file_path_or_url.endswith(('.json')):
                data = load_json(file_path_or_url)
            elif file_path_or_url.endswith(('.yml', '.yaml')):
                data = load_yaml(file_path_or_url)
            else:
                logging.error("Unsupported file format.")
                return

        extracted_data = extract_data(data)
        table = format_table(extracted_data)
        print(table)
    except Exception as e:
        logging.error(f"An error occurred in the main function with input '{file_path_or_url}': {e}")
