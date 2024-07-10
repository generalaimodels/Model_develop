import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any, Optional, Callable, Union, Tuple
import numpy as np
import random
import json
import os
from transformers import PreTrainedTokenizer
from datasets import Dataset as HFDataset
import psutil
import sys

class NovelDataset(Dataset):
    def __init__(
        self,
        data: List[Dict[str, Dict[str, Any]]],
        labels: Optional[List[Any]] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None
    ) -> None:
        """
        Initialize the NovelDataset.

        Args:
            data (List[Dict[str, Dict[str, Any]]]): The input data.
            labels (Optional[List[Any]]): The corresponding labels.
            tokenizer (Optional[PreTrainedTokenizer]): Tokenizer for text processing.
        """
        self.data = data
        self.labels = labels
        self.tokenizer = tokenizer
        self._validate_data()
        self.transforms = []
        self._iterator = None

    def _validate_data(self) -> None:
        """Validate the input data."""
        if not isinstance(self.data, list):
            raise ValueError("Data must be a list of dictionaries.")
        if self.labels is not None and len(self.data) != len(self.labels):
            raise ValueError("Data and labels must have the same length.")

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.data)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Get an item from the dataset.

        Args:
            index (int): The index of the item.

        Returns:
            Dict[str, Any]: The item at the specified index.
        """
        if not 0 <= index < len(self):
            raise IndexError("Index out of range")
        item = self.data[index]
        if self.labels is not None:
            item['label'] = self.labels[index]
        return item

    def __iter__(self) -> 'NovelDataset':
        """Initialize the iterator."""
        self._iterator = iter(range(len(self)))
        return self

    def __next__(self) -> Dict[str, Any]:
        """Get the next item in the dataset."""
        try:
            index = next(self._iterator)
            return self[index]
        except StopIteration:
            raise StopIteration("End of dataset reached")

    def split(
        self,
        ratio: float = 0.8,
        shuffle: bool = True,
        seed: Optional[int] = None
    ) -> Tuple['NovelDataset', 'NovelDataset']:
        """
        Split the dataset into training and testing sets.

        Args:
            ratio (float): The ratio of the training set size to the total dataset size.
            shuffle (bool): Whether to shuffle the data before splitting.
            seed (Optional[int]): Random seed for shuffling.

        Returns:
            Tuple[NovelDataset, NovelDataset]: The training and testing datasets.
        """
        if not 0 < ratio < 1:
            raise ValueError("Split ratio must be between 0 and 1")

        indices = list(range(len(self)))
        if shuffle:
            random.seed(seed)
            random.shuffle(indices)

        split_point = int(len(self) * ratio)
        train_indices = indices[:split_point]
        test_indices = indices[split_point:]

        train_data = [self.data[i] for i in train_indices]
        test_data = [self.data[i] for i in test_indices]

        train_labels = None if self.labels is None else [self.labels[i] for i in train_indices]
        test_labels = None if self.labels is None else [self.labels[i] for i in test_indices]

        return (
            NovelDataset(train_data, train_labels, self.tokenizer),
            NovelDataset(test_data, test_labels, self.tokenizer)
        )

    def shuffle(self, seed: Optional[int] = None) -> None:
        """
        Shuffle the dataset in-place.

        Args:
            seed (Optional[int]): Random seed for shuffling.
        """
        random.seed(seed)
        if self.labels is not None:
            combined = list(zip(self.data, self.labels))
            random.shuffle(combined)
            self.data, self.labels = zip(*combined)
        else:
            random.shuffle(self.data)

    def normalize(self, field: str, method: str = 'minmax') -> None:
        """
        Normalize a numeric field in the dataset.

        Args:
            field (str): The field to normalize.
            method (str): The normalization method ('minmax' or 'zscore').
        """
        values = [item[field] for item in self.data if field in item]
        if not values:
            raise ValueError(f"Field '{field}' not found in dataset")

        if method == 'minmax':
            min_val, max_val = min(values), max(values)
            for item in self.data:
                if field in item:
                    item[field] = (item[field] - min_val) / (max_val - min_val)
        elif method == 'zscore':
            mean, std = np.mean(values), np.std(values)
            for item in self.data:
                if field in item:
                    item[field] = (item[field] - mean) / std
        else:
            raise ValueError("Unsupported normalization method")

    def to_tensor(self, field: str) -> None:
        """
        Convert a field to PyTorch tensor.

        Args:
            field (str): The field to convert.
        """
        for item in self.data:
            if field in item:
                item[field] = torch.tensor(item[field])

    def augment(self, augmentation_func: Callable[[Dict[str, Any]], Dict[str, Any]]) -> None:
        """
        Apply data augmentation to the dataset.

        Args:
            augmentation_func (Callable[[Dict[str, Any]], Dict[str, Any]]): The augmentation function.
        """
        self.data = [augmentation_func(item) for item in self.data]

    def add_transform(self, transform: Callable[[Dict[str, Any]], Dict[str, Any]]) -> None:
        """
        Add a transform to the dataset's preprocessing pipeline.

        Args:
            transform (Callable[[Dict[str, Any]], Dict[str, Any]]): The transform function.
        """
        self.transforms.append(transform)

    def preprocess(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply all transforms in the preprocessing pipeline.

        Args:
            item (Dict[str, Any]): The item to preprocess.

        Returns:
            Dict[str, Any]: The preprocessed item.
        """
        for transform in self.transforms:
            item = transform(item)
        return item

    def save(self, path: str) -> None:
        """
        Save the dataset to a file.

        Args:
            path (str): The file path to save the dataset.
        """
        with open(path, 'w') as f:
            json.dump({'data': self.data, 'labels': self.labels}, f)

    @classmethod
    def load(cls, path: str, tokenizer: Optional[PreTrainedTokenizer] = None) -> 'NovelDataset':
        """
        Load a dataset from a file.

        Args:
            path (str): The file path to load the dataset from.
            tokenizer (Optional[PreTrainedTokenizer]): Tokenizer for text processing.

        Returns:
            NovelDataset: The loaded dataset.
        """
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(data['data'], data['labels'], tokenizer)

    def summary(self) -> Dict[str, Any]:
        """
        Provide a summary of the dataset.

        Returns:
            Dict[str, Any]: A dictionary containing dataset summary information.
        """
        summary = {
            "size": len(self),
            "features": list(self.data[0].keys()) if self.data else [],
            "has_labels": self.labels is not None,
        }
        if self.labels is not None:
            summary["label_distribution"] = {label: self.labels.count(label) for label in set(self.labels)}
        return summary

    def filter(self, condition: Callable[[Dict[str, Any]], bool]) -> 'NovelDataset':
        """
        Filter the dataset based on a condition.

        Args:
            condition (Callable[[Dict[str, Any]], bool]): The filtering condition.

        Returns:
            NovelDataset: A new dataset containing only the items that satisfy the condition.
        """
        filtered_data = [item for item in self.data if condition(item)]
        filtered_labels = None if self.labels is None else [
            label for item, label in zip(self.data, self.labels) if condition(item)
        ]
        return NovelDataset(filtered_data, filtered_labels, self.tokenizer)

    def batch(self, batch_size: int) -> 'BatchIterator':
        """
        Create a batch iterator for the dataset.

        Args:
            batch_size (int): The size of each batch.

        Returns:
            BatchIterator: An iterator that yields batches of the specified size.
        """
        return BatchIterator(self, batch_size)

    def map(self, func: Callable[[Dict[str, Any]], Dict[str, Any]]) -> 'NovelDataset':
        """
        Apply a function to all items in the dataset.

        Args:
            func (Callable[[Dict[str, Any]], Dict[str, Any]]): The function to apply.

        Returns:
            NovelDataset: A new dataset with the function applied to all items.
        """
        mapped_data = [func(item) for item in self.data]
        return NovelDataset(mapped_data, self.labels, self.tokenizer)

    def cache(self) -> None:
        """Cache the dataset in memory for faster access."""
        self._cache = list(self)

    def export_format(self, format: str = 'huggingface') -> Union[HFDataset, Any]:
        """
        Export the dataset to a specified format.

        Args:
            format (str): The desired export format.

        Returns:
            Union[HFDataset, Any]: The dataset in the specified format.
        """
        if format == 'huggingface':
            return HFDataset.from_dict({
                'data': self.data,
                'labels': self.labels if self.labels is not None else [''] * len(self)
            })
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def memory_footprint(self) -> int:
        """
        Calculate the memory footprint of the dataset.

        Returns:
            int: The memory usage in bytes.
        """
        return sum(sys.getsizeof(item) for item in self.data)


class BatchIterator:
    def __init__(self, dataset: NovelDataset, batch_size: int) -> None:
        """
        Initialize the BatchIterator.

        Args:
            dataset (NovelDataset): The dataset to iterate over.
            batch_size (int): The size of each batch.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.index = 0

    def __iter__(self) -> 'BatchIterator':
        """Initialize the iterator."""
        self.index = 0
        return self

    def __next__(self) -> List[Dict[str, Any]]:
        """Get the next batch."""
        if self.index >= len(self.dataset):
            raise StopIteration
        batch = self.dataset.data[self.index:self.index + self.batch_size]
        self.index += self.batch_size
        return batch

