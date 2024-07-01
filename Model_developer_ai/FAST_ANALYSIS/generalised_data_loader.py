import os
import time
from typing import Any, Dict, Optional, Sequence, Union, Callable
from pathlib import Path
import sys
import numpy as np
from dataclasses import dataclass, field
from random import randint
from typing import Optional
from termcolor import colored
from datasets import DatasetDict, load_dataset
from datasets.features import Features
from datasets.splits import Split
from datasets.download.download_config import DownloadConfig
from datasets.download.download_manager import DownloadMode
from datasets.utils.info_utils import VerificationMode
import logging


from datasets import (load_dataset, 
                      DatasetDict,
                      Dataset,
                      concatenate_datasets
                      )

from typing import Dict, Any, List,Union,Optional
from pathlib import Path



SEED:int = 42

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def rgb_print(text: str, rgb: tuple):
    """
    Prints the given text in the specified RGB color.
    
    Args:
        text: The text to print.
        rgb: A tuple of three integers representing the RGB color.
    """
    r, g, b = rgb
    # Convert RGB to a color string that termcolor can use
    color_str = f'\033[38;2;{r};{g};{b}m{text}\033[0m'
    return color_str


def load_and_prepare_dataset_for_hemanth(
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




def set_seed(seed: int) -> None:
    import numpy as np
    import random
    np.random.seed(seed)
    random.seed(seed)

def timing_decorator(func: Callable) -> Callable:
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} executed in {end_time - start_time:.8f}s")
        return result
    return wrapper

@timing_decorator
def prepare_datasetsforhemanth( 
                     path: str,
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
    if 'train' in list(dataset.keys()):
        train_holdout_dataset = dataset['train'].train_test_split(test_size=0.2, seed=42)
        test_eval_dataset = train_holdout_dataset['test'].train_test_split(test_size=0.5, seed=42)
        dataset['train'] = train_holdout_dataset['train']
        dataset['test'] = test_eval_dataset['train']
        dataset['eval'] = test_eval_dataset['test']
    else:
        print("The dataset does not have a 'train' split.")
        if len(list(dataset.keys())) > 0:
            first_split = list(dataset.keys())[0]
            print(f"Using the '{first_split}' split as the training set.")
            train_holdout_dataset = dataset[first_split].train_test_split(test_size=0.2, seed=42)
            test_eval_dataset = train_holdout_dataset['test'].train_test_split(test_size=0.5, seed=42)
            dataset['train'] = train_holdout_dataset['train']
            dataset['test'] = test_eval_dataset['train']
            dataset['eval'] = test_eval_dataset['test']
        else:
            print("The dataset does not have any splits.")
    return dataset



def get_dataset_info_for_hemanth(dataset) -> Dict[str, Any]:
    """
    Loads a dataset from Hugging Face Hub and provides detailed information about it.

    Args:
        dataset_name: The name of the dataset on the Hugging Face Hub.
        cache_dir: The directory to cache the downloaded dataset. 

    Returns:
        A dictionary containing information about the dataset.

    Raises:
        FileNotFoundError: If the dataset is not found in the specified cache directory.
        
    
    if __name__ == "__main__":
        dataset_name = "food101"
        dataset = prepare_datasets("food101",cache_dir="E:\\LLMS\\hemanth\\dataset_huggingface" )
        dataset_info = get_dataset_info(dataset)    

        logging.info("Dataset information:")
        for split_name, split_data in dataset_info.items():
            logging.info(f"Split: {split_name}")
            for key, value in split_data.items():
                logging.info(f"  {key}:{value}")

    """
    

    try:
        dataset_info = {} 
        for split_name, dataset_split in dataset.items():
            dataset_info[split_name] = {
                "builder_name": dataset_split.builder_name,
                # "cache_files": dataset_split.cache_files,
                "citation": dataset_split.citation,
                "column_names": dataset_split.column_names,
                "config_name": dataset_split.config_name,
                # "data": dataset_split.data,
                "dataset_size": dataset_split.dataset_size,
                "features": dataset_split.features,
                # "download_checksums": dataset_split.download_checksums,
                "format": dataset_split.format,
                "homepage": dataset_split.homepage,
                # "info": dataset_split.info,
                "num_columns": dataset_split.num_columns,
                "num_rows": dataset_split.num_rows,
                "_output_all_columns": dataset_split._output_all_columns,
                "shape": dataset_split.shape,
                "size_in_bytes": dataset_split.size_in_bytes,
                "split": dataset_split.split,
                "supervised_keys": dataset_split.supervised_keys,
                "task_templates": dataset_split.task_templates,
                "version": dataset_split.version
            }

        logging.info(f"Dataset '{dataset}' successfully loaded.")
        for split_name, split_data in dataset_info.items():
            # print(rgb_print(f"\nSplit: {split_name}", (0, 255, 0)))  # Yellow text for split names
            for key, value in split_data.items():
                print(rgb_print(f"  {key} : {value}", (0, 255, 0)))
        return dataset_info

    except FileNotFoundError:
        logging.error(f"Dataset '{dataset}' not found in the specified cache directory.")
        raise
