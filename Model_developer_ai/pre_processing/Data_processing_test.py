import itertools
import torch
import logging
import multiprocessing as mp

from typing import Dict, List, Union, Optional, Callable, Any, Iterable
from torch.utils.data import Dataset, DataLoader, Sampler, IterableDataset
from torch.utils.data.dataloader import default_collate
from collections import defaultdict

class AdvancedDataset(Dataset):
    """
    A flexible and reusable dataset class for various NLP tasks.
    Example:
    input_ids = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])]
    labels = [torch.tensor([0, 1, 0]), torch.tensor([1, 0, 1])]
    masked_ids = [torch.tensor([1, 0, 1]), torch.tensor([0, 1, 0])]

    data_dict = {
        "input_ids": input_ids,
        "labels": labels,
        "masked_ids": masked_ids
    }

    dataset = AdvancedDataset.from_dict(data_dict)
    logger.info(f"Dataset size: {len(dataset)}")

    dataloader = dataset.get_dataloader(batch_size=2)
    
    for batch in dataloader:
        logger.info(f"Batch: {batch}")
        break

    """

    def __init__(
        self,
        data: List[Dict[str, torch.Tensor]],
        required_keys: Optional[List[str]] = None
    ):
        """
        Initialize the dataset.

        Args:
            data (List[Dict[str, torch.Tensor]]): List of dictionaries containing tensor data.
            required_keys (Optional[List[str]]): List of required keys in each data item.
        """
        self.data = data
        self.required_keys = required_keys or ["input_ids", "labels", "masked_ids"]
        self._validate_data()

    def _validate_data(self) -> None:
        """
        Validate the input data to ensure it meets the required structure.
        """
        for item in self.data:
            for key in self.required_keys:
                if key not in item:
                    logger.error(f"Missing required key '{key}' in data item")
                    raise KeyError(f"Missing required key '{key}' in data item")
                if not isinstance(item[key], torch.Tensor):
                    logger.error(f"Value for key '{key}' is not a tensor")
                    raise TypeError(f"Value for key '{key}' must be a tensor")

        logger.info("Data validation completed successfully")

    def __len__(self) -> int:
        """
        Return the number of items in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get an item from the dataset by index.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing tensor data for the item.
        """
        return self.data[idx]

    @classmethod
    def from_dict(cls, data_dict: Dict[str, List[torch.Tensor]]) -> 'AdvancedDataset':
        """
        Create a dataset from a dictionary of tensors.

        Args:
            data_dict (Dict[str, List[torch.Tensor]]): Dictionary with keys as feature names and values as lists of tensors.

        Returns:
            AdvancedDataset: A new instance of the dataset.
        """
        if not all(isinstance(v, list) for v in data_dict.values()):
            logger.error("All values in data_dict must be lists of tensors")
            raise ValueError("All values in data_dict must be lists of tensors")

        data = [dict(zip(data_dict.keys(), values)) for values in zip(*data_dict.values())]
        return cls(data)

    def get_dataloader(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        **kwargs
    ) -> DataLoader:
        """
        Create a DataLoader for this dataset.

        Args:
            batch_size (int): How many samples per batch to load.
            shuffle (bool): Set to True to have the data reshuffled at every epoch.
            num_workers (int): How many subprocesses to use for data loading.
            **kwargs: Additional arguments to pass to the DataLoader.

        Returns:
            DataLoader: A DataLoader instance for this dataset.
        """
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            **kwargs
        )

class AdvancedDataset_single_or_multiple(Dataset):
    """
    A flexible and reusable dataset class for various NLP tasks with support for
    single- and multi-process data loading.
    
    
    # Example usage
    input_ids = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])]
    labels = [torch.tensor([0, 1, 0]), torch.tensor([1, 0, 1])]
    masked_ids = [torch.tensor([1, 0, 1]), torch.tensor([0, 1, 0])]

    data_dict = {
        "input_ids": input_ids,
        "labels": labels,
        "masked_ids": masked_ids
    }

    dataset = AdvancedDataset.from_dict(data_dict)
    logger.info(f"Dataset size: {len(dataset)}")

    # Single-process data loading
    single_process_loader = dataset.get_dataloader(batch_size=2, num_workers=0)
    logger.info("Single-process data loading:")
    for batch in single_process_loader:
        logger.info(f"Batch: {batch}")
        break

    # Multi-process data loading
    multi_process_loader = dataset.get_dataloader(
        batch_size=2,
        num_workers=2,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )
    logger.info("Multi-process data loading:")
    for batch in multi_process_loader:
        logger.info(f"Batch: {batch}")
        break


    
    
    """

    def __init__(
        self,
        data: List[Dict[str, torch.Tensor]],
        required_keys: Optional[List[str]] = None,
        transform: Optional[Callable] = None
    ):
        """
        Initialize the dataset.

        Args:
            data (List[Dict[str, torch.Tensor]]): List of dictionaries containing tensor data.
            required_keys (Optional[List[str]]): List of required keys in each data item.
            transform (Optional[Callable]): A function/transform to apply to the data.
        """
        self.data = data
        self.required_keys = required_keys or ["input_ids", "labels", "masked_ids"]
        self.transform = transform
        self._validate_data()

    def _validate_data(self) -> None:
        """
        Validate the input data to ensure it meets the required structure.
        """
        for item in self.data:
            for key in self.required_keys:
                if key not in item:
                    logger.error(f"Missing required key '{key}' in data item")
                    raise KeyError(f"Missing required key '{key}' in data item")
                if not isinstance(item[key], torch.Tensor):
                    logger.error(f"Value for key '{key}' is not a tensor")
                    raise TypeError(f"Value for key '{key}' must be a tensor")

        logger.info("Data validation completed successfully")

    def __len__(self) -> int:
        """
        Return the number of items in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get an item from the dataset by index.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing tensor data for the item.
        """
        item = self.data[idx]
        if self.transform:
            item = self.transform(item)
        return item

    @classmethod
    def from_dict(cls, data_dict: Dict[str, List[torch.Tensor]]) -> 'AdvancedDataset':
        """
        Create a dataset from a dictionary of tensors.

        Args:
            data_dict (Dict[str, List[torch.Tensor]]): Dictionary with keys as feature names and values as lists of tensors.

        Returns:
            AdvancedDataset: A new instance of the dataset.
        
        
        """
        if not all(isinstance(v, list) for v in data_dict.values()):
            logger.error("All values in data_dict must be lists of tensors")
            raise ValueError("All values in data_dict must be lists of tensors")

        data = [dict(zip(data_dict.keys(), values)) for values in zip(*data_dict.values())]
        return cls(data)

    def get_dataloader(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = False,
        worker_init_fn: Optional[Callable] = None,
        **kwargs
    ) -> DataLoader:
        """
        Create a DataLoader for this dataset.

        Args:
            batch_size (int): How many samples per batch to load.
            shuffle (bool): Set to True to have the data reshuffled at every epoch.
            num_workers (int): How many subprocesses to use for data loading.
            pin_memory (bool): If True, the data loader will copy Tensors into CUDA pinned memory before returning them.
            worker_init_fn (Optional[Callable]): If not None, this will be called on each worker subprocess with the worker id as input.
            **kwargs: Additional arguments to pass to the DataLoader.

        Returns:
            DataLoader: A DataLoader instance for this dataset.
        """
        if num_workers > 0 and mp.get_start_method(allow_none=True) is None:
            mp.set_start_method('spawn')

        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            worker_init_fn=worker_init_fn,
            **kwargs
        )

def worker_init_fn(worker_id: int) -> None:
    """
    Initialize the worker process.

    Args:
        worker_id (int): The ID of the worker process.
    """
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        logger.info(f"Initializing worker {worker_id}")
        # Set a unique seed for this worker
        torch.manual_seed(torch.initial_seed() + worker_id)
        # You can perform additional initialization here if needed




logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedDataset_map(Dataset):
    def __init__(
        self,
        data: List[Dict[str, torch.Tensor]],
        transform: Optional[Callable] = None
    ):
        self.data = data
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        if self.transform:
            item = self.transform(item)
        return item

class AdvancedIterableDataset(IterableDataset):
    def __init__(
        self,
        data_source: Iterable[Dict[str, torch.Tensor]],
        transform: Optional[Callable] = None
    ):
        self.data_source = data_source
        self.transform = transform

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading
            iter_data = iter(self.data_source)
        else:  # in a worker process
            per_worker = int(len(self.data_source) / worker_info.num_workers)
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = iter_start + per_worker
            iter_data = itertools.islice(self.data_source, iter_start, iter_end)
        
        for item in iter_data:
            if self.transform:
                item = self.transform(item)
            yield item

def custom_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {key: default_collate([d[key] for d in batch]) for key in batch[0]}

class AdvancedDataLoader:
    """
    input_ids = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])]
    labels = [torch.tensor([0, 1, 0]), torch.tensor([1, 0, 1])]
    masked_ids = [torch.tensor([1, 0, 1]), torch.tensor([0, 1, 0])]

    data = [
        {"input_ids": iid, "labels": lab, "masked_ids": mid}
        for iid, lab, mid in zip(input_ids, labels, masked_ids)
    ]

    dataset = AdvancedDataset_map(data)
    
    dataloader = AdvancedDataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        persistent_workers=True
    )

    logger.info("Map-style dataset loading:")
    for batch in dataloader:
        logger.info(f"Batch: {batch}")
        break

    # Example usage with iterable-style dataset
    iterable_data = [
        {
            "input_ids": torch.tensor([i, i+1, i+2]),
            "labels": torch.tensor([0, 1, 0]),
            "masked_ids": torch.tensor([1, 0, 1])
        }
        for i in range(5)
    ]

    iterable_dataset = AdvancedIterableDataset(iterable_data)
    
    iterable_dataloader = AdvancedDataLoader(
        iterable_dataset,
        batch_size=2,
        num_workers=2,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )

    logger.info("Iterable-style dataset loading:")
    for batch in iterable_dataloader:
        logger.info(f"Batch: {batch}")
    """
    def __init__(
        self,
        dataset: Union[Dataset, IterableDataset],
        batch_size: int = 1,
        shuffle: bool = False,
        sampler: Optional[Sampler] = None,
        batch_sampler: Optional[Sampler] = None,
        num_workers: int = 0,
        collate_fn: Callable = custom_collate_fn,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: float = 0,
        worker_init_fn: Optional[Callable] = None,
        multiprocessing_context: Optional[str] = None,
        generator: Optional[torch.Generator] = None,
        prefetch_factor: Optional[int] = None,
        persistent_workers: bool = False,
        pin_memory_device: str = ""
    ):
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
            multiprocessing_context=multiprocessing_context,
            generator=generator,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            pin_memory_device=pin_memory_device
        )

    def __iter__(self):
        return iter(self.dataloader)

    def __len__(self):
        return len(self.dataloader)

def worker_init_fn(worker_id: int) -> None:
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        logger.info(f"Initializing worker {worker_id}")
        torch.manual_seed(torch.initial_seed() + worker_id)

