# 🔒 Confidentiality Notice 
#
# DARFA-LLMs, is developed under the auspices of CDAC (Center for Development of Advanced Computing). 
# The project's codebase, algorithms, and any associated documentation are considered confidential 
# and proprietary to CDACB. 
#
# Copyright Statement ©️
#
# All code within this project is the intellectual property of Kandimalla Hemanth. 👨‍💻 Any reproduction, 
# distribution, or unauthorized use of the code or its components without explicit permission from the
# copyright holder is strictly prohibited. 
#
# Project Purpose 
#
# The DARFA-llms project is dedicated to advancing human development through the application of artificial 
# intelligence (AI) technologies. Its mission is to address real-world challenges and contribute 
# to the betterment of society. 
#
#  Non-Disclosure Agreement (NDA) 
#
# Access to this Python project and its associated materials is restricted to authorized personnel only.
# 🔒 By accessing or using any part of this project, you agree to maintain strict confidentiality 
# and adhere to any applicable non-disclosure agreements (NDAs) or confidentiality policies established
# by CDACB. 
#
# ⚠️ Disclaimer of Liability 📜
#
# While every effort has been made to ensure the accuracy and reliability of the code and information 
# contained within this project, CDACB and Kandimalla Hemanth shall not be held liable for any direct, 
# indirect, incidental, special, or consequential damages arising out of the use or inability to use the
# project, even if advised of the possibility of such damages. 
#
# 🚫 Usage Restrictions 📝
#
# The code and algorithms developed as part of the DARFA project are intended for research, educational,
# and non-commercial purposes only. Any commercial use, modification, or adaptation of the codebase 
# requires explicit written consent from CDACB and Kandimalla Hemanth.
#
#  Ethical Considerations 
#
# The DARFA project adheres to ethical principles and guidelines governing the responsible 
# use of AI technologies. All efforts are made to ensure that the project's objectives 
# align with ethical standards and promote positive outcomes for humanity. 
from datasets import (load_dataset, 
                      DatasetDict,
                      Dataset,
                      concatenate_datasets
                      )

from typing import Dict, Any, List,Union,Optional
from pathlib import Path

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
