import os
import csv
from typing import List


def get_image_paths(folder: str) -> List[str]:
    """
    Recursively get a list of paths for all .png images in the given folder.
    Handles Windows-style paths properly.

    Args:
        folder: The folder path to search.

    Returns:
        A list of paths for all .png images found.
    """
    
    image_paths = []

    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith('.png'):
                path = os.path.join(root, file)
                path = path.replace('\\', '/') # convert Windows paths
                image_paths.append(path)

    return image_paths


def save_to_csv(image_paths: List[str], csv_file: str) -> None:
    """
    Save a list of image paths to a CSV file.

    Args:
        image_paths: A list of image path strings.
        csv_file: Path to the CSV file to save.
    """
    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for path in image_paths:
            writer.writerow([path])


if __name__ == '__main__':
    folder = 'C:/Users/heman/Desktop/deeplearning/accelerate/'
    csv_file = 'image_paths.csv'

    image_paths = get_image_paths(folder)
    save_to_csv(image_paths, csv_file)
