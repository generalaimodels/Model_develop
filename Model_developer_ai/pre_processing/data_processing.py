import torch
import os
import random
import numpy as np
import logging

from typing import Dict, List, Union, Optional, Callable
from torch.utils.data import Dataset, DataLoader, get_worker_info


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedDataset(Dataset):
    """
    A flexible and reusable dataset class that can handle various input structures.
    
    Example:
    # Create sample data
    sample_data = {
        'input_ids': [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])],
        'labels': [torch.tensor([0, 1, 0]), torch.tensor([1, 0, 1])],
        'masked_ids': [torch.tensor([1, 0, 1]), torch.tensor([0, 1, 0])]
    }

    # Create dataset
    dataset = AdvancedDataset.from_dict(sample_data)
    logger.info(f"Dataset created with {len(dataset)} samples")

    # Create dataloader
    dataloader = create_dataloader(dataset, batch_size=2, shuffle=True)
    logger.info(f"DataLoader created with batch size {dataloader.batch_size}")

    # Iterate through the dataloader
    for batch in dataloader:
        logger.info(f"Batch keys: {batch.keys()}")
        logger.info(f"Input IDs shape: {batch['input_ids'].shape}")
        logger.info(f"Labels shape: {batch['labels'].shape}")
        logger.info(f"Masked IDs shape: {batch['masked_ids'].shape}")
        break
    """

    def __init__(self, data: List[Dict[str, torch.Tensor]]):
        """
        Initialize the dataset.

        Args:
            data (List[Dict[str, torch.Tensor]]): List of dictionaries containing tensor data.
        """
        self.data = data
        self._validate_data()

    def _validate_data(self) -> None:
        """
        Validate the input data structure.
        """
        if not self.data:
            logger.warning("Empty dataset provided.")
            return

        expected_keys = set(self.data[0].keys())
        for idx, item in enumerate(self.data):
            if set(item.keys()) != expected_keys:
                logger.error(f"Inconsistent keys in data item at index {idx}")
                raise ValueError(f"Inconsistent data structure at index {idx}")

    def __len__(self) -> int:
        """
        Return the length of the dataset.

        Returns:
            int: Number of items in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get an item from the dataset.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing tensor data for the specified index.
        """
        return self.data[idx]

    @classmethod
    def from_dict(cls, data_dict: Dict[str, List[torch.Tensor]]) -> 'AdvancedDataset':
        """
        Create a dataset from a dictionary of lists of tensors.

        Args:
            data_dict (Dict[str, List[torch.Tensor]]): Dictionary with keys as field names and values as lists of tensors.

        Returns:
            AdvancedDataset: New instance of AdvancedDataset.
        """
        if not all(isinstance(v, list) for v in data_dict.values()):
            logger.error("Invalid input format for from_dict method")
            raise ValueError("All values in data_dict must be lists")

        data = [dict(zip(data_dict.keys(), values)) for values in zip(*data_dict.values())]
        return cls(data)

def create_dataloader(
    dataset: AdvancedDataset,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False
) -> DataLoader:
    """
    Create a DataLoader from the given dataset.

    Args:
        dataset (AdvancedDataset): The dataset to create a DataLoader for.
        batch_size (int, optional): How many samples per batch to load. Defaults to 1.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to False.
        num_workers (int, optional): How many subprocesses to use for data loading. Defaults to 0.
        pin_memory (bool, optional): If True, the data loader will copy Tensors into CUDA pinned memory before returning them. Defaults to False.
        drop_last (bool, optional): If True, drop the last incomplete batch. Defaults to False.

    Returns:
        DataLoader: A DataLoader instance for the given dataset.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last
    )

class AdvancedDataset_update(Dataset):
    """A flexible and reusable dataset class that can handle various input structures.
    
    # Set a base seed for reproducibility
    base_seed = 42
    torch.manual_seed(base_seed)
    random.seed(base_seed)
    np.random.seed(base_seed)

    # Create sample data
    sample_data = {
        'input_ids': [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])],
        'labels': [torch.tensor([0, 1, 0]), torch.tensor([1, 0, 1])],
        'masked_ids': [torch.tensor([1, 0, 1]), torch.tensor([0, 1, 0])]
    }

    # Create dataset
    dataset = AdvancedDataset.from_dict(sample_data)
    logger.info(f"Dataset created with {len(dataset)} samples")

    # Create dataloader with multi-processing
    num_workers = 2  # Set to 0 for single-process data loading
    dataloader = create_dataloader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        generator=torch.Generator().manual_seed(base_seed)
    )
    logger.info(f"DataLoader created with batch size {dataloader.batch_size} and {num_workers} workers")

    # Iterate through the dataloader
    for batch in dataloader:
        logger.info(f"Batch keys: {batch.batch.keys()}")
        logger.info(f"Input IDs shape: {batch.batch['input_ids'].shape}")
        logger.info(f"Labels shape: {batch.batch['labels'].shape}")
        logger.info(f"Masked IDs shape: {batch.batch['masked_ids'].shape}")
        break
    """
    def __init__(self, data: List[Dict[str, torch.Tensor]]):
        self.data = data
        self._validate_data()

    def _validate_data(self) -> None:
        if not self.data:
            logger.warning("Empty dataset provided.")
            return

        expected_keys = set(self.data[0].keys())
        for idx, item in enumerate(self.data):
            if set(item.keys()) != expected_keys:
                logger.error(f"Inconsistent keys in data item at index {idx}")
                raise ValueError(f"Inconsistent data structure at index {idx}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.data[idx]

    @classmethod
    def from_dict(cls, data_dict: Dict[str, List[torch.Tensor]]) -> 'AdvancedDataset':
        if not all(isinstance(v, list) for v in data_dict.values()):
            logger.error("Invalid input format for from_dict method")
            raise ValueError("All values in data_dict must be lists")

        data = [dict(zip(data_dict.keys(), values)) for values in zip(*data_dict.values())]
        return cls(data)

def worker_init_fn(worker_id: int) -> None:
    """
    Function to initialize each worker process.
    """
    worker_info = get_worker_info()
    if worker_info is not None:
        # Set a unique seed for each worker
        seed = worker_info.seed
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        logger.info(f"Worker {worker_id} initialized with seed {seed}")

class PinnedBatch:
    """
    Custom batch type that supports pinned memory.
    """
    def __init__(self, batch: Dict[str, torch.Tensor]):
        self.batch = batch

    def pin_memory(self):
        return {k: v.pin_memory() for k, v in self.batch.items()}

def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> PinnedBatch:
    """
    Custom collate function that returns a PinnedBatch.
    """
    return PinnedBatch({key: torch.stack([d[key] for d in batch]) for key in batch[0]})

def create_dataloader(
    dataset: AdvancedDataset,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    worker_init_fn: Optional[Callable] = worker_init_fn,
    collate_fn: Callable = collate_fn,
    generator: Optional[torch.Generator] = None
) -> DataLoader:
    """
    Create a DataLoader with support for multi-processing and memory pinning.

    Args:
        dataset (AdvancedDataset): The dataset to create a DataLoader for.
        batch_size (int, optional): How many samples per batch to load. Defaults to 1.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to False.
        num_workers (int, optional): How many subprocesses to use for data loading. 0 means single-process. Defaults to 0.
        pin_memory (bool, optional): If True, the data loader will copy Tensors into CUDA pinned memory before returning them. Defaults to False.
        drop_last (bool, optional): If True, drop the last incomplete batch. Defaults to False.
        worker_init_fn (Callable, optional): Function to initialize workers. Defaults to worker_init_fn.
        collate_fn (Callable, optional): Function to collate data samples into batches. Defaults to collate_fn.
        generator (torch.Generator, optional): Generator used for sampling. Defaults to None.

    Returns:
        DataLoader: A DataLoader instance for the given dataset.
    """
    if num_workers > 0 and os.name == 'nt':
        logger.warning("On Windows, increasing num_workers will increase memory usage. "
                       "Make sure to wrap your main script with if __name__ == '__main__':")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        worker_init_fn=worker_init_fn,
        collate_fn=collate_fn,
        generator=generator
    )



class SimpleDataset(Dataset):
    """
    if __name__ == "__main__":
    # Create sample data
    sample_data = [
        {
            'input_ids': torch.tensor([1, 2, 3]),
            'labels': torch.tensor([0, 1, 0]),
            'masked_ids': torch.tensor([1, 0, 1])
        },
        {
            'input_ids': torch.tensor([4, 5, 6]),
            'labels': torch.tensor([1, 0, 1]),
            'masked_ids': torch.tensor([0, 1, 0])
        }
    ]

    # Create dataset
    dataset = SimpleDataset(sample_data)
    logger.info(f"Dataset created with {len(dataset)} samples")

    # Create dataloader
    dataloader = create_dataloader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    logger.info(f"DataLoader created with batch size {dataloader.batch_size}")

    # Iterate through the dataloader
    for batch in dataloader:
        logger.info(f"Batch keys: {batch.keys()}")
        logger.info(f"Input IDs shape: {batch['input_ids'].shape}")
        logger.info(f"Labels shape: {batch['labels'].shape}")
        logger.info(f"Masked IDs shape: {batch['masked_ids'].shape}")
        break
    """
    def __init__(self, data: List[Dict[str, torch.Tensor]]):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.data[idx]

def create_dataloader_1(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    collate_fn: Optional[Callable] = None
) -> DataLoader:
    """
    Create a DataLoader with the given dataset and parameters.

    Args:
        dataset (Dataset): The dataset to load data from.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the data.
        num_workers (int): Number of subprocesses to use for data loading.
        pin_memory (bool): If True, the data loader will copy Tensors into CUDA pinned memory before returning them.
        drop_last (bool): If True, drop the last incomplete batch.
        collate_fn (Callable, optional): Merges a list of samples to form a mini-batch.

    Returns:
        DataLoader: A DataLoader instance for the given dataset.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_fn
    )

