import logging
import os
from functools import partial
from typing import Tuple, Dict, Any, List
import multiprocessing
import datasets
import torch
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import AutoTokenizer
from datasets import Dataset


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
def load_data(file_path: str) -> Dataset:
    """
    Load data from CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        Dataset: Loaded dataset.

    Raises:
        FileNotFoundError: If the file is not found.
        ValueError: If the file is empty or has invalid format.
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        dataset = datasets.load_dataset('csv', data_files=file_path)['train']
        
        if len(dataset) == 0:
            raise ValueError("The dataset is empty")

        if 'content' not in dataset.column_names or 'prompt' not in dataset.column_names:
            raise ValueError("The dataset must contain 'content' and 'prompt' columns")

        logger.info(f"Loaded {len(dataset)} examples from {file_path}")
        return dataset
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise
def create_data_split(dataset: Dataset, test_size: float = 0.2, seed: int = 42) -> Tuple[Dataset, Dataset]:
    """
    Create a train-test split using the datasets library.

    Args:
        dataset (Dataset): Input dataset.
        test_size (float): Proportion of the dataset to include in the test split.
        seed (int): Random seed for reproducibility.

    Returns:
        Tuple[Dataset, Dataset]: Train and test datasets.

    Raises:
        ValueError: If test_size is not between 0 and 1.
    """
    try:
        if not 0 < test_size < 1:
            raise ValueError("test_size must be between 0 and 1")

        split = dataset.train_test_split(test_size=test_size, seed=seed)
        logger.info(f"Split dataset into {len(split['train'])} training examples and {len(split['test'])} test examples")
        return split['train'].select(range(10000)), split['test'].select(range(10000))
    except Exception as e:
        logger.error(f"Error splitting dataset: {str(e)}")
        raise
class SlidingWindowDataset(TorchDataset):
    def __init__(self, tokenized_inputs: Dict[str, torch.Tensor], max_length: int, stride: int):
        self.input_ids = tokenized_inputs['input_ids']
        self.attention_mask = tokenized_inputs['attention_mask']
        self.max_length = max_length
        self.stride = stride

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.input_ids[idx],
        }

def apply_prompts(example: Dict[str, Any], system_prompt: str) -> Dict[str, Any]:
    """
    Apply chain of thought prompting to the example.
    """
    user_prompt = (f"Content: {example['content']}\n"
                   f"Prompt: {example['prompt']}\n"
                   "Is this fraudulent? Explain your reasoning step by step.")
    example['chat'] = f"{system_prompt}\n\n{user_prompt}"
    return example

def tokenize_with_sliding_window(examples: Dict[str, Any], tokenizer: AutoTokenizer, max_length: int, stride: int) -> Dict[str, torch.Tensor]:
    """
    Tokenize examples using a sliding window approach and ensure all tensors have the same size.
    """
    tokenized = tokenizer(
        examples['chat'],
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    
    input_ids_list = []
    attention_mask_list = []
    
    for i in range(len(examples['chat'])):
        input_ids = tokenized['input_ids'][i]
        attention_mask = tokenized['attention_mask'][i]
        
        # Always create at least one window
        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        
        # If the sequence is longer than max_length, create additional windows
        if len(input_ids) == max_length:
            for start in range(stride, len(input_ids) - max_length + stride, stride):
                end = start + max_length
                window_input_ids = torch.cat([input_ids[start:], input_ids[:end-len(input_ids)]])
                window_attention_mask = torch.cat([attention_mask[start:], attention_mask[:end-len(attention_mask)]])
                input_ids_list.append(window_input_ids)
                attention_mask_list.append(window_attention_mask)
    
    # Convert lists to tensors
    result = {
        'input_ids': torch.stack(input_ids_list),
        'attention_mask': torch.stack(attention_mask_list),
        'labels': torch.stack(input_ids_list)
    }
    
    return result
def preprocess_data(dataset: Dataset, tokenizer: AutoTokenizer, max_length: int, stride: int, system_prompt: str) -> SlidingWindowDataset:
    """
    Preprocess the dataset by tokenizing and applying prompts.
    """
    try:
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        if stride <= 0:
            raise ValueError("stride must be positive")

        dataset = dataset.map(
            partial(apply_prompts, system_prompt=system_prompt),
            # num_proc=os.cpu_count()
        )
        
        tokenized_dataset = dataset.map(
            partial(tokenize_with_sliding_window, tokenizer=tokenizer, max_length=max_length, stride=stride),
            batched=True,
            remove_columns=dataset.column_names,
            # num_proc=os.cpu_count()
        )
        
        sliding_window_dataset = SlidingWindowDataset(tokenized_dataset, max_length, stride)
        
        logger.info("Data preprocessing completed successfully")
        return sliding_window_dataset
    except Exception as e:
        logger.error(f"Error in preprocessing data: {str(e)}")
        raise

def create_data_loaders(train_dataset: SlidingWindowDataset, test_dataset: SlidingWindowDataset, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for train and test datasets.
    """
    try:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, )
        logger.info("Created data loaders successfully")
        return train_loader, test_loader
    except Exception as e:
        logger.error(f"Error creating data loaders: {str(e)}")
        raise




