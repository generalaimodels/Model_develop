import os
import random
import pandas as pd
import json
import librosa
import numpy as np
from pathlib import Path


import plotly.express as px
import plotly.graph_objects as go
import soundfile as sf
import numpy as np
from datasets import DatasetDict  # Assuming you have the datasets library installed
from PIL import Image

from dataclasses import dataclass, asdict
from typing import List,Any, Dict
from datasets import DatasetDict
from datasets import(Dataset, 
                     DatasetDict,
                     Features,
                     Array2D,
                     Array3D,
                     Array4D,
                     Array5D,
                     arrow_dataset,
                     arrow_reader,
                     ArrowBasedBuilder,
)
                     

def load_and_label_files(root_dir: str) -> DatasetDict:
    """
    Load and label files from a folder and its subfolders.

    Args:
        root_dir (str): The root directory to search for files.

    Returns:
        DatasetDict: A dictionary of datasets containing the labeled files.
        
    Example:
       def main():
           root_dir = "C:/Users/heman/"
           
           try:
               labeled_datasets = load_and_label_files(root_dir)
               print(labeled_datasets['audio']['path'][0] )
               print("Labeled datasets:")
               for file_type, dataset in labeled_datasets.items():
                   print(f"{file_type}: {len(dataset)} files")
           except FileNotFoundError as e:
               print(f"Error: {str(e)}")
           except Exception as e:
               print(f"An unexpected error occurred: {str(e)}")

        if __name__ == "__main__":
           main()
    """
    file_paths: Dict[str, List[str]] = {
        "audio": [],
        "video": [],
        "text": [],
        "image": [],
        "code": []
    }

    audio_extensions = [".mp3", ".wav", ".flac", ".aac", ".ogg", ".wma", ".m4a"]
    video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm"]
    text_extensions = [".txt", ".md", ".rtf", ".doc", ".docx", ".pdf", ".epub"]
    image_extensions = [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".svg", ".webp"]
    code_extensions = [
        ".py", ".java", ".c", ".cpp", ".js", ".ts", ".html", ".css", ".php", ".rb", 
        ".go", ".rs", ".swift", ".kt", ".m", ".sh", ".bat", ".pl", ".lua", ".sql", 
        ".r", ".ipynb", ".json", ".xml", ".yaml", ".yml"
    ]

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            file_path = file_path.replace("\\", "/")  # Ensure forward slashes
            extension = os.path.splitext(filename)[1].lower()

            if extension in audio_extensions:
                file_paths["audio"].append(file_path)
            elif extension in video_extensions:
                file_paths["video"].append(file_path)
            elif extension in text_extensions:
                file_paths["text"].append(file_path)
            elif extension in image_extensions:
                file_paths["image"].append(file_path)
            elif extension in code_extensions:
                file_paths["code"].append(file_path)

    datasets = {}
    for file_type, paths in file_paths.items():
        if paths:
            dataset = Dataset.from_dict({"path": paths})
            dataset = dataset.map(lambda x: {"label": file_type}, desc='Files:')
            datasets[file_type] = dataset

    return DatasetDict(datasets)



def load_and_label_files_test(root_dir: str) -> DatasetDict:
    """
    Load and label files from a folder and its subfolders using forward slashes in paths.

    Args:
        root_dir (str): The root directory to search for files.

    Returns:
        DatasetDict: A dictionary of datasets containing the labeled files.
    """
    file_paths: Dict[str, List[str]] = {
        "audio": [],
        "video": [],
        "text": [],
        "image": [],
        "code": []
    }

    audio_extensions = [".mp3", ".wav", ".flac", ".aac", ".ogg", ".wma", ".m4a"]
    video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm"]
    text_extensions = [".txt", ".md", ".rtf", ".doc", ".docx", ".pdf", ".epub"]
    image_extensions = [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".svg", ".webp"]
    code_extensions = [
        ".py", ".java", ".c", ".cpp", ".js", ".ts", ".html", ".css", ".php", ".rb", 
        ".go", ".rs", ".swift", ".kt", ".m", ".sh", ".bat", ".pl", ".lua", ".sql", 
        ".r", ".ipynb", ".json", ".xml", ".yaml", ".yml"
    ]

    root_path = Path(root_dir)
    for file_path in root_path.rglob('*'):
        if file_path.is_file():
            extension = file_path.suffix.lower()
            file_path_str = str(file_path.as_posix())  # Convert path to string with forward slashes

            if extension in audio_extensions:
                file_paths["audio"].append(file_path_str)
            elif extension in video_extensions:
                file_paths["video"].append(file_path_str)
            elif extension in text_extensions:
                file_paths["text"].append(file_path_str)
            elif extension in image_extensions:
                file_paths["image"].append(file_path_str)
            elif extension in code_extensions:
                file_paths["code"].append(file_path_str)

    datasets = {}
    for file_type, paths in file_paths.items():
        if paths:
            dataset = Dataset.from_dict({"path": paths})
            dataset = dataset.map(lambda x: {"label": file_type},desc='Files:')
            datasets[file_type] = dataset

    return DatasetDict(datasets)




@dataclass
class AudioData:
    path: str
    audio: np.ndarray
    sample_rate: int
    label: int
    transcript: str


def load_audio_files(base_path: str, transcripts: List[str]) -> List[AudioData]:
    """_summary_

    Args:
        base_path (str): _description_
        transcripts (List[str]): _description_

    Returns:
        List[AudioData]: _description_
        
        
    Example:
    

    """
    audio_files = []
    for label, subfolder in enumerate(sorted(Path(base_path).iterdir())):
        if subfolder.is_dir():
            transcript = transcripts[label]
            for audio_file in subfolder.glob('*.flac'):
                audio, sample_rate = librosa.load(audio_file, sr=48000)
                audio_files.append(AudioData(
                    path=str(audio_file),
                    audio=audio,
                    sample_rate=sample_rate,
                    label=label,
                    transcript=transcript
                ))
    return audio_files


def split_dataset(audio_files: List[AudioData], train_ratio: float, test_ratio: float, eval_ratio: float) -> DatasetDict:
    random.shuffle(audio_files)
    total_count = len(audio_files)
    train_count = int(total_count * train_ratio)
    test_count = int(total_count * test_ratio)

    train_set = audio_files[:train_count]
    test_set = audio_files[train_count:train_count + test_count]
    eval_set = audio_files[train_count + test_count:]

    return DatasetDict({
        'train': Dataset.from_pandas(pd.DataFrame([asdict(ad) for ad in train_set])),
        'test': Dataset.from_pandas(pd.DataFrame([asdict(ad) for ad in test_set])),
        'eval': Dataset.from_pandas(pd.DataFrame([asdict(ad) for ad in eval_set]))
    })


def save_splits(splits: DatasetDict, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for split_name, dataset in splits.items():
        output_path = Path(output_dir) / f'{split_name}.json'
        dataset.to_json(output_path)


def main(base_path: str, transcripts: List[str], output_dir: str) -> None:
    audio_files = load_audio_files(base_path, transcripts)
    print(f'Loaded {len(audio_files)} audio files.')
    splits = split_dataset(audio_files, train_ratio=0.8, test_ratio=0.1, eval_ratio=0.1)
    print(f'Split into train: {len(splits["train"])}, test: {len(splits["test"])}, eval: {len(splits["eval"])}')
    print(splits)
    save_splits(splits, output_dir)



def plot_images(dataset: DatasetDict, output_folder: str) -> None:
    """Plot images and save them in the respective folders."""
    image_data = dataset['image']
    for record in image_data:
        img_path = record['path']
        label = record['label']
        img = Image.open(img_path)
        fig = px.imshow(img, title=f'Label: {label}')
        plot_path = os.path.join(output_folder, 'image', f"{os.path.basename(img_path).split('.')[0]}.html")
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        fig.write_html(plot_path)

def plot_audio(dataset: DatasetDict, output_folder: str) -> None:
    """Plot audio waveforms and save them in the respective folders."""
    audio_data = dataset['audio']
    for record in audio_data:
        audio_path = record['path']
        label = record['label']
        audio, sample_rate = sf.read(audio_path)
        time = np.linspace(0, len(audio) / sample_rate, num=len(audio))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=time, y=audio, mode='lines', name='Audio Waveform'))
        fig.update_layout(title=f'Label: {label}', xaxis_title='Time (s)', yaxis_title='Amplitude')
        plot_path = os.path.join(output_folder, 'audio', f"{os.path.basename(audio_path).split('.')[0]}.html")
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        fig.write_html(plot_path)



