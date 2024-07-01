


import os
import time
from typing import Any, Dict, Optional, Sequence, Union, Callable
from pathlib import Path
import sys
import numpy as np
from dataclasses import dataclass, field
from random import randint
from typing import Optional
from datasets import DatasetDict, load_dataset
from datasets.features import Features
from datasets.splits import Split
from datasets.download.download_config import DownloadConfig
from datasets.download.download_manager import DownloadMode
from datasets.utils.info_utils import VerificationMode

from datasets import (load_dataset, 
                      DatasetDict,
                      Dataset,
                      concatenate_datasets
                      )

from typing import Dict, Any, List,Union,Optional
from pathlib import Path


SEED:int=0
#Load the datset
def load_and_prepare_dataset(
    input_source: Union[str, Path, Dict[str, List[Union[str, Path]]]],
    split_ratios: tuple = (0.8, 0.1, 0.1),
    seed: int = 42,
    streaming: bool = False
    ) -> DatasetDict:
    """
    Load a dataset from various input sources and prepare it by splitting into train, test, and eval sets.

    :param input_source: A dataset name, path to a folder, a single file, multiple files, or a dictionary specifying train, test, and eval files.
    :param split_ratios: A tuple containing the ratios for train, test, and eval splits (default is (0.8, 0.1, 0.1)).
    :param seed: A random seed for reproducibility of the split (default is 42).
    :param streaming: Whether to use streaming to handle large files (default is False).
    :return: A DatasetDict containing the split datasets.
    
    Example:
    # Example usage with streaming for large files:
    # dataset_dict = load_and_prepare_dataset({
    #     'train': ['train_file_1.csv', 'train_file_2.csv'],
    #     'test': ['test_file.csv'],
    #     'eval': ['eval_file.csv']
    # }, streaming=True)
    # print(dataset_dict)
    OUTPUT1:
    DatasetDict({
    train: DatasetDict({
        train: Dataset({
            features: ['act', 'prompt'],
            num_rows: 459
        })
    })
    test: DatasetDict({
        train: Dataset({
            features: ['act', 'prompt'],
            num_rows: 459
        })
    })
    eval: DatasetDict({
        train: Dataset({
            features: ['act', 'prompt'],
            num_rows: 153
        })
    })
    })
    EXAMPLE2:
    dataset=load_and_prepare_dataset('fka/awesome-chatgpt-prompts')
    DatasetDict({
    train: Dataset({
        features: ['act', 'prompt'],
        num_rows: 122
    })
    test: Dataset({
        features: ['act', 'prompt'],
        num_rows: 15
    })
    eval: Dataset({
        features: ['act', 'prompt'],
        num_rows: 16
    })
    })
    EXAMPLE3:
    datset_path=load_and_prepare_dataset('/content/awesome-chatgpt-prompts')
    DatasetDict({
    train: Dataset({
        features: ['act', 'prompt'],
        num_rows: 122
    })
    test: Dataset({
        features: ['act', 'prompt'],
        num_rows: 15
    })
    eval: Dataset({
        features: ['act', 'prompt'],
        num_rows: 16
    })
    })

    """
    # Load dataset from different types of input sources
    if isinstance(input_source, (str, Path)):
        # Dataset name, single file or path to folder
        dataset = load_dataset(input_source, streaming=streaming)
        dataset = DatasetDict(dataset)
    elif isinstance(input_source, dict):
        # Dictionary with specified train, test, and eval files
        formats = ['csv', 'json', 'jsonl', 'parquet', 'txt']
        datasets = {}
        for split, files in input_source.items():
            format_detected = None
            for fmt in formats:
                if any(str(file).endswith(fmt) for file in files):
                    format_detected = fmt
                    break
            if format_detected is None:
                raise ValueError(f"No supported file format detected for files: {files}")
            datasets[split] = load_dataset(format_detected, data_files=files, streaming=streaming)
        dataset = DatasetDict(datasets)
    else:
        raise ValueError("Input source should be a dataset name, path to a folder, a single file, multiple files, or a dictionary.")

    # Perform the split if needed and if not in streaming mode
    if not streaming:
        train_size, test_size, eval_size = split_ratios
        assert 0.0 < train_size < 1.0 and 0.0 < test_size < 1.0 and 0.0 < eval_size < 1.0 and (train_size + test_size + eval_size) == 1.0, \
            "Split ratios must be between 0 and 1 and sum up to 1."

        if "train" not in dataset or "test" not in dataset or "eval" not in dataset:
            # Assuming all splits are to be derived from the 'train' dataset
            full_dataset = concatenate_datasets(list(dataset.values())) if isinstance(dataset, dict) else dataset
            split_dataset = full_dataset.train_test_split(train_size=train_size, seed=seed)
            test_eval_split = split_dataset['test'].train_test_split(test_size=test_size / (test_size + eval_size), seed=seed)

            dataset = DatasetDict({
                "train": split_dataset["train"],
                "test": test_eval_split["train"],
                "eval": test_eval_split["test"]
            })

    return dataset


SEED:int = 42

def set_seed(seed: int) -> None:
    # This function should set the random seed for all libraries (numpy, random, etc.)
    import numpy as np
    import random
    np.random.seed(seed)
    random.seed(seed)

# Decorator to measure the execution time of functions
def timing_decorator(func: Callable) -> Callable:
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} executed in {end_time - start_time:.8f}s")
        return result
    return wrapper

@timing_decorator
def prepare_datasets( path: str,
                    name: Optional[str] = None,
                     data_dir: Optional[str] = None,
                     data_files: Union[str, Sequence[str], Dict[str, Union[str, Sequence[str]]], None] = None,
                     split: Optional[Split] = None,
                     cache_dir: Optional[str] = None,
                     features: Optional[Features] = None,
                     download_config: Optional[DownloadConfig] = None,
                     download_mode: Optional[Union[DownloadMode, str]] = None,
                     verification_mode: Optional[Union[VerificationMode, str]] = None,
                     keep_in_memory: Optional[bool] = None,
                     revision: Optional[Union[str, int]] = None,
                     num_proc: Optional[int] = None,
                     storage_options: Optional[Dict[str, Any]] = None,
                     **config_kwargs: Any) -> DatasetDict:
    """
    
    _summary_

    Args:
        path (str): _description_
        name (Optional[str], optional): _description_. Defaults to None.
        data_dir (Optional[str], optional): _description_. Defaults to None.
        data_files (Union[str, Sequence[str], Dict[str, Union[str, Sequence[str]]], None], optional): _description_. Defaults to None.
        split (Optional[Split], optional): _description_. Defaults to None.
        cache_dir (Optional[str], optional): _description_. Defaults to None.
        features (Optional[Features], optional): _description_. Defaults to None.
        download_config (Optional[DownloadConfig], optional): _description_. Defaults to None.
        download_mode (Optional[Union[DownloadMode, str]], optional): _description_. Defaults to None.
        verification_mode (Optional[Union[VerificationMode, str]], optional): _description_. Defaults to None.
        keep_in_memory (Optional[bool], optional): _description_. Defaults to None.
        revision (Optional[Union[str, int]], optional): _description_. Defaults to None.
        num_proc (Optional[int], optional): _description_. Defaults to None.
        storage_options (Optional[Dict[str, Any]], optional): _description_. Defaults to None.

    Returns:
        DatasetDict: _description_
    """
    set_seed(SEED)
    dataset = load_dataset(
            path=path,
            name=name,
            data_dir=data_dir,
            data_files=data_files,
            split=split,
            cache_dir=cache_dir,
            features=features,
            download_config=download_config,
            download_mode=download_mode,
            verification_mode=verification_mode,
            keep_in_memory=keep_in_memory,
            revision=revision,
            num_proc=num_proc,
            storage_options=storage_options,
            **config_kwargs
        )
    
    # Check if the dataset has a 'train' split
    if 'train' in list(dataset.keys()):
        # Split the dataset into train and holdout sets
        train_holdout_dataset = dataset['train'].train_test_split(test_size=0.2, seed=42)
    
        # Split the holdout set into test and eval sets
        test_eval_dataset = train_holdout_dataset['test'].train_test_split(test_size=0.5, seed=42)
    
        # Update the dataset dictionary with the new splits
        dataset['train'] = train_holdout_dataset['train']
        dataset['test'] = test_eval_dataset['train']
        dataset['eval'] = test_eval_dataset['test']
    else:
        print("The dataset does not have a 'train' split.")
    
        # Check if the dataset has any splits
        if len(list(dataset.keys())) > 0:
            # Get the first available split
            first_split = list(dataset.keys())[0]
            print(f"Using the '{first_split}' split as the training set.")
    
            # Split the first available split into train and holdout sets
            train_holdout_dataset = dataset[first_split].train_test_split(test_size=0.2, seed=42)
    
            # Split the holdout set into test and eval sets
            test_eval_dataset = train_holdout_dataset['test'].train_test_split(test_size=0.5, seed=42)
    
            # Update the dataset dictionary with the new splits
            dataset['train'] = train_holdout_dataset['train']
            dataset['test'] = test_eval_dataset['train']
            dataset['eval'] = test_eval_dataset['test']
        else:
            print("The dataset does not have any splits.")
    return dataset
