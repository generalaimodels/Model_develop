import os
import warnings
import torch 
import argparse
import yaml

from peft import (
    LoraConfig,
    PeftConfig,
    PeftModel,
    TaskType,
    get_peft_model,
    prepare_model_for_int8_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    default_data_collator,
    get_linear_schedule_with_warmup,
    BitsAndBytesConfig
)
from tqdm import tqdm
from accelerate import Accelerator
from dataclasses import dataclass, field
from torch.utils.data import Dataset, DataLoader
from enum import Enum
from datasets import load_dataset,DatasetDict
from typing import Union , Dict,Optional,Any,List


# Define a function to read arguments from a YAML file
def load_arguments_from_yaml(yaml_file_path):
    with open(yaml_file_path, 'r') as stream:
        try:
            arguments = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            arguments = {}
    return arguments

# Create the parser
parser = argparse.ArgumentParser(description='Process training parameters and dataset configuration.')

#model and tokenizer
parser.add_argument('--model_name_or_path', type=str, default='TinyLlama/TinyLlama-1.1B-Chat-v1.0', help='Path to the model or tokenizer.')
parser.add_argument('--tokenizer_name_or_path', type=str, default='TinyLlama/TinyLlama-1.Chat-v1.0', help='Path to the tokenizer.')
# Add arguments to the parser
parser.add_argument('--config', type=str, help='Path to the YAML configuration file.')
parser.add_argument('--max_length', type=int, default=512, help='Maximum sequence length.')
parser.add_argument('--lr', type=float, default=3e-2, help='Learning rate.')
parser.add_argument('--num_epochs', type=int, default=3, help='Number of epochs.')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size.')
parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use for training.')
parser.add_argument("--output_dir_path", type=str, default='', help="Output directory path.")
# Dataset arguments
parser.add_argument('--speific_format_dir', type=str, default='',  help='user speific format of the dataset')
parser.add_argument('--dataset_local_path', type=str, default='', help='Local path to the dataset.')
parser.add_argument('--dataset_name', type=str, default='', help='Name of the dataset.')
parser.add_argument('--text_column', type=str, default='', help='Name of the text column in the dataset.')
parser.add_argument('--label_column', type=str, default='', help='Name of the label column in the dataset.')
parser.add_argument('--speific_format_extension', type=str, default='', help='dataset extension')
# Parse the arguments
args = parser.parse_args()

# If a YAML config file is provided, override the defaults/command-line arguments with values from the config
if args.config:
    config_args = load_arguments_from_yaml(args.config)
    for key, value in config_args.items():
        setattr(args, key, value)

# Define the device
DEVICE = args.device



def advanced_data_loader(input: Union[str, Dict[str, str]], format: Optional[str] = None, split_ratios: Optional[Dict[str, float]] = None) -> Optional[DatasetDict]:
    """
    Loads a dataset from a given input path or dictionary specifying file paths and splits it.

    :param input: A string representing the dataset name or directory, or a dictionary containing file paths.
    :param format: The format of the dataset if loading from a file (e.g., 'csv' or 'json').
    :param split_ratios: A dictionary with keys 'train', 'test', and 'eval' containing split ratios.
    :return: A loaded and split dataset or None in case of failure.
    """
    if split_ratios is None:
        split_ratios = {'train': 0.8, 'test': 0.1, 'eval': 0.1}

    try:
        # Load the dataset
        if isinstance(input, dict) and format in ['csv', 'json']:
            dataset = load_dataset(format, data_files=input)
        elif isinstance(input, str) and format == 'text':
            dataset = load_dataset(format, data_dir=input)
        elif isinstance(input, str) and format is None:
            dataset = load_dataset(input)
        else:
            warnings.warn("Invalid input or format. Please provide a valid dataset name, directory, or file paths.")
            return None
    except FileNotFoundError as e:
        warnings.warn(str(e))
        return None

    # Split the dataset
    if dataset:
        split_dataset = dataset['train'].train_test_split(test_size=split_ratios['test'] + split_ratios['eval'])
        test_eval_dataset = split_dataset['test'].train_test_split(test_size=split_ratios['eval'] / (split_ratios['test'] + split_ratios['eval']))
        dataset = DatasetDict({
            'train': split_dataset['train'],
            'test': test_eval_dataset['train'],
            'eval': test_eval_dataset['test']
        })

    print("Splits: ", dataset.keys())
    print("Columns: ", {split: dataset[split].column_names for split in dataset.keys()})
    return dataset


def create_tokenizer(tokenizer_name_or_path: str = 'gpt2') -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.bos_token_id is None:
        tokenizer.bos_token_id = tokenizer.pad_token_id
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token_id = tokenizer.pad_token_id
    if tokenizer.unk_token_id is None:
        tokenizer.unk_token_id = tokenizer.pad_token_id
    if tokenizer.sep_token_id is None:
        tokenizer.sep_token_id = tokenizer.pad_token_id
    if tokenizer.cls_token_id is None:
        tokenizer.cls_token_id = tokenizer.pad_token_id
    if tokenizer.mask_token_id is None:
        tokenizer.mask_token_id = tokenizer.pad_token_id
    return tokenizer


def Create_model_loader(model_name_or_path: str = 'gpt2') -> AutoModelForCausalLM:
    model=AutoModelForCausalLM.from_pretrained(model_name_or_path)
    return model



def preprocess_function(examples, text_column, label_column):
    """
    Preprocess the dataset.
    
    Args:
        examples (dict): A dictionary where keys are column names and values are lists of data.
        text_column (str): The name of the column containing text data.
        label_column (str): The name of the column containing label data.
    
    Returns:
        dict: A dictionary containing tokenized inputs and labels suitable for model training.
    """
    tokenizer=create_tokenizer(args.tokenizer_name_or_path)
    inputs = [f"{text}: {examples[text_column][i]} Label: " for i, text in enumerate(examples[text_column])]
    targets = [str(examples[label_column][i]) for i in range(len(examples[label_column]))]

    # Tokenize inputs and labels
    model_inputs = tokenizer(inputs, padding='max_length', truncation=True, max_length=args.max_length, return_tensors='pt')
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, padding='max_length', truncation=True, max_length=args.max_length, return_tensors='pt')

    # Replace padding token id's in labels with -100 so they are ignored in loss calculation
    labels["input_ids"] = [
        [(label if label != tokenizer.pad_token_id else -100) for label in label_input]
        for label_input in labels["input_ids"]
    ]

    # Ensure labels are the same size as model_inputs
    if 'attention_mask' in model_inputs:
        labels["attention_mask"] = model_inputs['attention_mask']

    model_inputs["labels"] = torch.tensor(labels["input_ids"])
    
    return model_inputs



def main():
    if args.dataset_name is None:
        dataset=advanced_data_loader(args.dataset_local_path)
    elif args.dataset_name is not None:
        dataset=load_dataset(args.dataset_name)
    elif args.speific_format is not None:
        dataset=advanced_data_loader(args.speific_formatz, format=args.speific_format)
    elif args.dataset_local_path is None and args.dataset_name is None:
        print("Please provide either a dataset name or a dataset local path")

    # tokenizer=create_tokenizer(tokenizer_name_or_path=args.tokenizer_name_or_path)

    model=Create_model_loader(
        model_name_or_path=args.model_name_or_path
    )
    
    def preprocess_wrapper(examples):
        
        return preprocess_function(examples, text_column=args.text_column, label_column=args.label_column)
    processed_datasets = dataset.map(
    preprocess_wrapper,
    batched=True,
    num_proc=1,
    remove_columns=dataset["train"].column_names,
    load_from_cache_file=True,
    desc=f"""
    This is kandimalla hemanth fine-tuning of any given model for given dataset
    
    """,
     )
        
    train_dataset=processed_datasets['train']
    test_dataset=processed_datasets['test']
    eval_dataset=processed_datasets['eval']
    train_dataloader = DataLoader(
    train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=args.batch_size, pin_memory=True
    )
    eval_dataloader = DataLoader(
        eval_dataset, shuffle=True,collate_fn=default_data_collator, batch_size=args.batch_size, pin_memory=True
    )
    test_dataloader = DataLoader(
        test_dataset, collate_fn=default_data_collator, batch_size=args.batch_size, pin_memory=True
    )
    # optimizer
    model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    # scheduler
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=args.num_epochs)
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0
        for step, batch in enumerate(tqdm(train_dataloader)):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.detach().float()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
    
        train_epoch_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} | Train Loss: {train_epoch_loss}")
    
    model.save_pretrained(f"{args.output_dir_path}")
    