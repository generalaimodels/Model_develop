import os
import random
from typing import List, Dict, Union, Optional, Callable
import numpy as np
import torch
import torchaudio
import librosa
from transformers import AutoFeatureExtractor
from datasets import Dataset, Audio
import json
from datasets import Dataset, DatasetDict
from torch.utils.data import IterableDataset
class AudioDataset:
    def __init__(
        self,
        data: List[Dict[str, Dict]],
        labels: Optional[List[Union[int, str]]] = None,
        sampling_rate: int = 16000,
        duration: Optional[float] = None,
        mono: bool = True
    ):
        """
        Initialize the AudioDataset.

        Args:
            data (List[Dict[str, Dict]]): List of dictionaries containing audio data.
            labels (Optional[List[Union[int, str]]]): List of labels for the audio data.
            sampling_rate (int): Target sampling rate for the audio.
            duration (Optional[float]): Target duration for the audio in seconds.
            mono (bool): Whether to convert audio to mono channel.
        """
        self.data = data
        self.labels = labels
        self.sampling_rate = sampling_rate
        self.duration = duration
        self.mono = mono
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
        self.transforms = []
        self._validate_data()

    def _validate_data(self):
        """Validate the input data."""
        if not isinstance(self.data, list):
            raise ValueError("Data must be a list of dictionaries.")
        if self.labels is not None and len(self.data) != len(self.labels):
            raise ValueError("Number of data items and labels must match.")

    def __len__(self) -> int:
        """Return the number of items in the dataset."""
        return len(self.data)

    def __getitem__(self, index: int) -> Dict:
        """
        Get an item from the dataset.

        Args:
            index (int): Index of the item to retrieve.

        Returns:
            Dict: Dictionary containing the audio data and label (if available).
        """
        if index < 0 or index >= len(self):
            raise IndexError("Index out of range")

        item = self.data[index]
        audio = self._load_and_process_audio(item['audio']['path'])
        
        result = {'audio': audio}
        if self.labels is not None:
            result['label'] = self.labels[index]
        
        return result

    def __iter__(self):
        """Return an iterator for the dataset."""
        self._iter_index = 0
        return self

    def __next__(self):
        """Get the next item in the dataset."""
        if self._iter_index >= len(self):
            raise StopIteration
        item = self[self._iter_index]
        self._iter_index += 1
        return item

    def _load_and_process_audio(self, audio_path: str) -> torch.Tensor:
        """
        Load and process an audio file.

        Args:
            audio_path (str): Path to the audio file.

        Returns:
            torch.Tensor: Processed audio tensor.
        """
        try:
            waveform, sr = torchaudio.load(audio_path)
            
            if self.mono and waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            if sr != self.sampling_rate:
                waveform = torchaudio.transforms.Resample(sr, self.sampling_rate)(waveform)
            
            if self.duration is not None:
                target_length = int(self.duration * self.sampling_rate)
                if waveform.shape[1] > target_length:
                    waveform = waveform[:, :target_length]
                elif waveform.shape[1] < target_length:
                    pad_length = target_length - waveform.shape[1]
                    waveform = torch.nn.functional.pad(waveform, (0, pad_length))
            
            for transform in self.transforms:
                waveform = transform(waveform)
            
            return waveform
        
        except Exception as e:
            print(f"Error processing audio file {audio_path}: {str(e)}")
            return torch.zeros((1, self.sampling_rate))

    def split(self, train_ratio: float = 0.8) -> tuple:
        """
        Split the dataset into training and testing sets.

        Args:
            train_ratio (float): Ratio of training data to total data.

        Returns:
            tuple: Training and testing datasets.
        """
        if not 0 < train_ratio < 1:
            raise ValueError("Train ratio must be between 0 and 1")

        split_index = int(len(self) * train_ratio)
        train_data = self.data[:split_index]
        test_data = self.data[split_index:]

        train_labels = self.labels[:split_index] if self.labels else None
        test_labels = self.labels[split_index:] if self.labels else None

        return (
            AudioDataset(train_data, train_labels, self.sampling_rate, self.duration, self.mono),
            AudioDataset(test_data, test_labels, self.sampling_rate, self.duration, self.mono)
        )

    def shuffle(self, seed: Optional[int] = None):
        """
        Shuffle the dataset.

        Args:
            seed (Optional[int]): Random seed for reproducibility.
        """
        if seed is not None:
            random.seed(seed)
        
        combined = list(zip(self.data, self.labels)) if self.labels else list(zip(self.data, [None] * len(self)))
        random.shuffle(combined)
        self.data, self.labels = zip(*combined)
        self.data = list(self.data)
        self.labels = list(self.labels) if self.labels[0] is not None else None

    def normalize(self):
        """Normalize the audio data."""
        def normalize_transform(waveform):
            return (waveform - waveform.mean()) / waveform.std()
        self.transforms.append(normalize_transform)

    def to_tensor(self):
        """Convert the audio data to PyTorch tensors."""
        def to_tensor_transform(waveform):
            return torch.tensor(waveform)
        self.transforms.append(to_tensor_transform)

    def augment(self, augmentation_func: Callable):
        """
        Add an augmentation function to the dataset.

        Args:
            augmentation_func (Callable): Function to apply for data augmentation.
        """
        self.transforms.append(augmentation_func)

    def add_transform(self, transform_func: Callable):
        """
        Add a custom transform function to the dataset.

        Args:
            transform_func (Callable): Custom transform function to apply.
        """
        self.transforms.append(transform_func)

    def preprocess(self):
        """Apply the feature extractor preprocessing."""
        def preprocess_transform(waveform):
            return self.feature_extractor(waveform, sampling_rate=self.sampling_rate)['input_values'][0]
        self.transforms.append(preprocess_transform)

    def save(self, path: str):
        """
        Save the dataset to a file.

        Args:
            path (str): Path to save the dataset.
        """
        data_to_save = {
            'data': self.data,
            'labels': self.labels,
            'sampling_rate': self.sampling_rate,
            'duration': self.duration,
            'mono': self.mono
        }
        with open(path, 'w') as f:
            json.dump(data_to_save, f)

    @classmethod
    def load(cls, path: str):
        """
        Load a dataset from a file.

        Args:
            path (str): Path to load the dataset from.

        Returns:
            AudioDataset: Loaded dataset.
        """
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(data['data'], data['labels'], data['sampling_rate'], data['duration'], data['mono'])

    def summary(self) -> Dict[str, Union[int, float, bool]]:
        """
        Provide a summary of the dataset.

        Returns:
            Dict[str, Union[int, float, bool]]: Summary of the dataset.
        """
        return {
            'num_samples': len(self),
            'sampling_rate': self.sampling_rate,
            'duration': self.duration,
            'mono': self.mono,
            'has_labels': self.labels is not None,
            'num_transforms': len(self.transforms)
        }

    def filter(self, condition: Callable[[Dict], bool]) -> 'AudioDataset':
        """
        Filter the dataset based on a condition.

        Args:
            condition (Callable[[Dict], bool]): Function that returns True for items to keep.

        Returns:
            AudioDataset: Filtered dataset.
        """
        filtered_data = [item for item in self.data if condition(item)]
        filtered_labels = [label for item, label in zip(self.data, self.labels) if condition(item)] if self.labels else None
        return AudioDataset(filtered_data, filtered_labels, self.sampling_rate, self.duration, self.mono)

    def batch(self, batch_size: int) -> List[Dict]:
        """
        Create batches of the dataset.

        Args:
            batch_size (int): Size of each batch.

        Returns:
            List[Dict]: List of batches.
        """
        batches = []
        for i in range(0, len(self), batch_size):
            batch = [self[j] for j in range(i, min(i + batch_size, len(self)))]
            batches.append(batch)
        return batches

    def map(self, func: Callable[[Dict], Dict]) -> 'AudioDataset':
        """
        Apply a function to all items in the dataset.

        Args:
            func (Callable[[Dict], Dict]): Function to apply to each item.

        Returns:
            AudioDataset: Transformed dataset.
        """
        mapped_data = [func(item) for item in self.data]
        return AudioDataset(mapped_data, self.labels, self.sampling_rate, self.duration, self.mono)

    def cache(self, cache_dir: str):
        """
        Cache the processed audio data to disk.

        Args:
            cache_dir (str): Directory to store the cached data.
        """
        os.makedirs(cache_dir, exist_ok=True)
        for i, item in enumerate(self.data):
            cache_path = os.path.join(cache_dir, f"item_{i}.pt")
            if not os.path.exists(cache_path):
                processed_audio = self._load_and_process_audio(item['audio']['path'])
                torch.save(processed_audio, cache_path)
            item['audio']['cached_path'] = cache_path

    def export_format(self, format: str = 'huggingface') -> Union[Dataset, Dict]:
        """
        Export the dataset to a specified format.

        Args:
            format (str): Format to export to ('huggingface' or 'dict').

        Returns:
            Union[Dataset, Dict]: Exported dataset.
        """
        if format == 'huggingface':
            return Dataset.from_dict({
                'audio': [item['audio']['path'] for item in self.data],
                'label': self.labels if self.labels else [None] * len(self)
            }).cast_column('audio', Audio(sampling_rate=self.sampling_rate))
        elif format == 'dict':
            return {
                'data': self.data,
                'labels': self.labels,
                'sampling_rate': self.sampling_rate,
                'duration': self.duration,
                'mono': self.mono
            }
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def memory_footprint(self) -> int:
        """
        Calculate the approximate memory footprint of the dataset.

        Returns:
            int: Approximate memory usage in bytes.
        """
        import sys
        return sum(sys.getsizeof(item) for item in self.data) + \
               (sys.getsizeof(self.labels) if self.labels else 0) + \
               sum(sys.getsizeof(transform) for transform in self.transforms)

class AudioDataset(IterableDataset):
    def __init__(
        self,
        data: List[Dict[str, Dict]],
        labels: Optional[List] = None,
        duration: Optional[float] = None,
        sampling_rate: int = 16000,
        mono: bool = True,
        feature_extractor: Optional[str] = None,
    ):
        """
        Initialize the AudioDataset.

        Args:
            data (List[Dict[str, Dict]]): List of dictionaries containing audio data.
            labels (Optional[List]): List of labels for the audio data.
            duration (Optional[float]): Target duration for audio clips in seconds.
            sampling_rate (int): Target sampling rate for audio.
            mono (bool): Whether to convert audio to mono.
            feature_extractor (Optional[str]): Name of the pre-trained feature extractor.
        """
        self.data = data
        self.labels = labels
        self.duration = duration
        self.sampling_rate = sampling_rate
        self.mono = mono
        self.feature_extractor = (
            AutoFeatureExtractor.from_pretrained(feature_extractor)
            if feature_extractor
            else None
        )
        self._validate_data()
        self.index = 0
        self.transforms = []
        self._cache = {}

    def _validate_data(self):
        """Validate the input data."""
        if not isinstance(self.data, list) or not all(
            isinstance(item, dict) for item in self.data
        ):
            raise ValueError("Data must be a list of dictionaries.")
        if self.labels and len(self.data) != len(self.labels):
            raise ValueError("Number of data items and labels must match.")

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.data)

    def __getitem__(self, index: int) -> Dict:
        """
        Get an item from the dataset.

        Args:
            index (int): Index of the item to retrieve.

        Returns:
            Dict: Dictionary containing the audio data and label.
        """
        if index < 0 or index >= len(self):
            raise IndexError("Index out of range")

        if index in self._cache:
            return self._cache[index]

        item = self.data[index]
        audio_path = item.get("audio", {}).get("path")
        if not audio_path or not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        try:
            audio, sr = librosa.load(
                audio_path, sr=self.sampling_rate, mono=self.mono, duration=self.duration
            )
        except Exception as e:
            raise RuntimeError(f"Error loading audio file: {e}")

        if self.feature_extractor:
            features = self.feature_extractor(
                audio, sampling_rate=sr, return_tensors="pt"
            )
            processed_item = {"features": features}
        else:
            processed_item = {"audio": audio, "sampling_rate": sr}

        if self.labels:
            processed_item["label"] = self.labels[index]

        for transform in self.transforms:
            processed_item = transform(processed_item)

        self._cache[index] = processed_item
        return processed_item

    def __iter__(self):
        """Return an iterator for the dataset."""
        return self

    def __next__(self) -> Dict:
        """Get the next item in the dataset."""
        if self.index >= len(self):
            self.index = 0
            raise StopIteration
        item = self[self.index]
        self.index += 1
        return item

    def split(self, train_ratio: float = 0.8) -> DatasetDict:
        """
        Split the dataset into training and testing sets.

        Args:
            train_ratio (float): Ratio of data to use for training.

        Returns:
            DatasetDict: Dictionary containing train and test datasets.
        """
        if not 0 < train_ratio < 1:
            raise ValueError("Train ratio must be between 0 and 1")

        indices = list(range(len(self)))
        random.shuffle(indices)
        split = int(train_ratio * len(self))
        train_indices, test_indices = indices[:split], indices[split:]

        train_data = [self.data[i] for i in train_indices]
        test_data = [self.data[i] for i in test_indices]

        train_labels = (
            [self.labels[i] for i in train_indices] if self.labels else None
        )
        test_labels = [self.labels[i] for i in test_indices] if self.labels else None

        train_dataset = AudioDataset(
            train_data,
            train_labels,
            self.duration,
            self.sampling_rate,
            self.mono,
            self.feature_extractor,
        )
        test_dataset = AudioDataset(
            test_data,
            test_labels,
            self.duration,
            self.sampling_rate,
            self.mono,
            self.feature_extractor,
        )

        return DatasetDict({"train": train_dataset, "test": test_dataset})

    def shuffle(self, seed: Optional[int] = None):
        """
        Shuffle the dataset.

        Args:
            seed (Optional[int]): Random seed for shuffling.
        """
        if seed is not None:
            random.seed(seed)
        random.shuffle(self.data)
        if self.labels:
            random.shuffle(self.labels)

    def normalize(self):
        """Normalize the audio data."""
        def normalize_transform(item):
            if "audio" in item:
                item["audio"] = librosa.util.normalize(item["audio"])
            return item
        self.add_transform(normalize_transform)

    def to_tensor(self):
        """Convert audio data to PyTorch tensors."""
        def to_tensor_transform(item):
            if "audio" in item:
                item["audio"] = torch.from_numpy(item["audio"]).float()
            return item
        self.add_transform(to_tensor_transform)

    def augment(self, augmentation_func: Callable):
        """
        Add an augmentation function to the dataset.

        Args:
            augmentation_func (Callable): Function to apply for augmentation.
        """
        self.add_transform(augmentation_func)

    def add_transform(self, transform: Callable):
        """
        Add a custom transform to the dataset.

        Args:
            transform (Callable): Function to apply as a transform.
        """
        self.transforms.append(transform)

    def preprocess(self):
        """Apply all preprocessing steps."""
        self.normalize()
        self.to_tensor()

    def save(self, path: str):
        """
        Save the dataset to a file.

        Args:
            path (str): Path to save the dataset.
        """
        data_to_save = {
            "data": self.data,
            "labels": self.labels,
            "duration": self.duration,
            "sampling_rate": self.sampling_rate,
            "mono": self.mono,
        }
        with open(path, "w") as f:
            json.dump(data_to_save, f)
    @classmethod
    def load(cls, path: str):
        """
        Load a dataset from a file.

        Args:
            path (str): Path to load the dataset from.

        Returns:
            AudioDataset: Loaded dataset.
        """
        with open(path, "r") as f:
            data = json.load(f)
        return cls(
            data["data"],
            data["labels"],
            data["duration"],
            data["sampling_rate"],
            data["mono"],
        )

    def summary(self) -> Dict[str, Union[int, float, str]]:
        """
        Provide a summary of the dataset.

        Returns:
            Dict[str, Union[int, float, str]]: Summary of the dataset.
        """
        return {
            "num_samples": len(self),
            "duration": self.duration,
            "sampling_rate": self.sampling_rate,
            "mono": self.mono,
            "has_labels": self.labels is not None,
        }

    def filter(self, condition: Callable[[Dict], bool]) -> 'AudioDataset':
        """
        Filter the dataset based on a condition.

        Args:
            condition (Callable[[Dict], bool]): Function to apply as a filter.

        Returns:
            AudioDataset: Filtered dataset.
        """
        filtered_data = [item for item in self.data if condition(item)]
        filtered_labels = (
            [label for item, label in zip(self.data, self.labels) if condition(item)]
            if self.labels
            else None
        )
        return AudioDataset(
            filtered_data,
            filtered_labels,
            self.duration,
            self.sampling_rate,
            self.mono,
            self.feature_extractor,
        )

    def batch(self, batch_size: int) -> 'AudioDataset':
        """
        Create batches from the dataset.

        Args:
            batch_size (int): Size of each batch.

        Returns:
            AudioDataset: Batched dataset.
        """
        def batch_generator():
            for i in range(0, len(self), batch_size):
                yield self.data[i:i + batch_size]

        return AudioDataset(
            list(batch_generator()),
            self.labels,
            self.duration,
            self.sampling_rate,
            self.mono,
            self.feature_extractor,
        )

    def map(self, func: Callable[[Dict], Dict]) -> 'AudioDataset':
        """
        Apply a function to all items in the dataset.

        Args:
            func (Callable[[Dict], Dict]): Function to apply to each item.

        Returns:
            AudioDataset: Mapped dataset.
        """
        mapped_data = [func(item) for item in self.data]
        return AudioDataset(
            mapped_data,
            self.labels,
            self.duration,
            self.sampling_rate,
            self.mono,
            self.feature_extractor,
        )

    def cache(self):
        """Cache all items in the dataset."""
        for i in range(len(self)):
            _ = self[i]

    def export_format(self, format: str = 'wav'):
        """
        Export audio files to a specific format.

        Args:
            format (str): Audio format to export to.
        """
        def export_transform(item):
            if "audio" in item and "sampling_rate" in item:
                audio = item["audio"]
                sr = item["sampling_rate"]
                if isinstance(audio, torch.Tensor):
                    audio = audio.numpy()
                temp_path = f"temp_audio.{format}"
                librosa.output.write_wav(temp_path, audio, sr)
                item["audio"] = {"path": temp_path}
            return item
        self.add_transform(export_transform)

    def memory_footprint(self) -> int:
        """
        Calculate the memory footprint of the dataset.

        Returns:
            int: Memory usage in bytes.
        """
        import sys
        return sum(sys.getsizeof(item) for item in self.data)

    def __repr__(self) -> str:
        """Return a string representation of the dataset."""
        return f"AudioDataset(samples={len(self)}, duration={self.duration}, sampling_rate={self.sampling_rate}, mono={self.mono})"

# # Example usage:
# if __name__ == "__main__":
#     # Sample data (replace with your actual data)
#     sample_data = [
#         {"audio": {"path": "path/to/audio1.wav"}},
#         {"audio": {"path": "path/to/audio2.wav"}},
#         {"audio": {"path": "path/to/audio3.wav"}},
#     ]
#     sample_labels = [0, 1, 0]

#     # Create dataset
#     dataset = AudioDataset(sample_data, sample_labels, duration=5, sampling_rate=16000, mono=True)

#     # Preprocess
#     dataset.preprocess()

#     # Split dataset
#     train_test = dataset.split(0.8)

#     # Print summary
#     print(dataset.summary())

#     # Iterate through dataset
#     for item in dataset:
#         print(item.keys())

#     # Save and load
#     dataset.save("dataset.json")
#     loaded_dataset = AudioDataset.load("dataset.json")

#     # Apply custom transform
#     def custom_transform(item):
#         # Add your custom transformation logic here
#         return item

#     dataset.add_transform(custom_transform)

#     # Filter dataset
#     filtered_dataset = dataset.filter(lambda x: x.get("label", 0) == 1)

#     # Create batches
#     batched_dataset = dataset.batch(32)

#     # Map function to dataset
#     mapped_dataset = dataset.map(lambda x: {**x, "new_field": 1})

#     # Cache dataset
#     dataset.cache()

#     # Export to different format
#     dataset.export_format("mp3")

#     # Print memory footprint
#     print(f"Memory footprint: {dataset.memory_footprint()} bytes")
    # @classmethod
    # def load(cls, path: str):
    #     """
    #     Load a dataset from a file.

    #     Args:
    #         path (str): Path to load the dataset from.

    #     Returns:
    #         AudioDataset: Loaded dataset.
    #     """
    #     with open(path, "r
# # Example usage:
# if __name__ == "__main__":
#     # Sample data (replace with your actual data)
#     sample_data = [
#         {"audio": {"path": "path/to/audio1.wav"}},
#         {"audio": {"path": "path/to/audio2.wav"}},
#         {"audio": {"path": "path/to/audio3.wav"}}
#     ]
#     sample_labels = [0, 1, 0]

#     # Create dataset
#     dataset = AudioDataset(sample_data, sample_labels, sampling_rate=16000, duration=5, mono=True)

#     # Apply some transformations
#     dataset.normalize()
#     dataset.to_tensor()
#     dataset.preprocess()

#     # Print summary
#     print(dataset.summary())

#     # Iterate through dataset
#     for item in dataset:
#         print(item['audio'].shape, item['label'])

#     # Split dataset
#     train_dataset, test_dataset = dataset.split(0.8)

#     # Export to HuggingFace dataset
#     hf_dataset = dataset.export_format('huggingface')

#     # Print memory footprint
#     print(f"Memory footprint: {dataset.memory_footprint() / 1024 / 1024:.2f} MB")