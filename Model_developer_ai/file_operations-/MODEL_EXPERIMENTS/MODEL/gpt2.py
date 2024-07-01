import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer, default_data_collator,AutoModelForCausalLM
from pathlib import Path
from typing import Dict, Any, List,Union,Optional,Tuple
from datasets import (load_dataset, 
                      DatasetDict,
                      concatenate_datasets
                      )


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
        dataset = load_dataset(input_source, streaming=streaming,trust_remote_code=True)
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
            datasets[split] = load_dataset(format_detected, data_files=files, streaming=streaming,trust_remote_code=True)
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


def preprocess_function(examples: Dict[str, List[str]], text_columns: Union[str, List[str]], label_column: str, tokenizer: AutoTokenizer, max_length: int) -> Dict[str, torch.Tensor]:
    batch_size = len(examples[label_column])
    
    if isinstance(text_columns, str):
        text_columns = [text_columns]
    
    inputs = []
    for i in range(batch_size):
        input_text = " ".join([f"{col}: {examples[col][i]}" for col in text_columns if examples[col][i] is not None])
        inputs.append(f"{input_text} Label:{examples[label_column][i]}")  # Include label in the input text
    
    # Tokenize inputs only (labels are included in the inputs)
    model_inputs = tokenizer(inputs, max_length=max_length, truncation=True, padding=True)
    
    # Set 'labels' to be identical to 'input_ids'
    model_inputs["labels"] = model_inputs["input_ids"]
    
    # Convert values to PyTorch tensors, skipping None values
    return {k: torch.tensor([v for v in model_inputs[k] if v is not None]) for k in model_inputs}
def process_dataset(dataset, text_columns: Union[str, List[str]], label_column: str, tokenizer: AutoTokenizer, max_length: int, batch_size: int):
    processed_datasets = dataset.map(
        lambda examples: preprocess_function(examples, text_columns, label_column, tokenizer, max_length),
        batched=True,
        num_proc=1,
        remove_columns=dataset["train"].column_names,
        load_from_cache_file=False,
        desc=" DARFA project  by Hemanth ",
    )
    
    train_dataset = processed_datasets["train"]
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True
    )
    
    return train_dataloader

def preprocess_function(examples: Dict[str, List[str]], text_columns: Union[str, List[str]], label_column: str, tokenizer: AutoTokenizer, max_length: int) -> Dict[str, torch.Tensor]:
    batch_size = len(examples[label_column])
    
    if isinstance(text_columns, str):
        text_columns = [text_columns]
    
    for i in range(batch_size):
        input_text = " ".join([f"{col}: {examples[col][i]}" for col in text_columns if examples[col][i] is not None])
        input_text = f"{input_text} Label:{examples[label_column][i]}"  # Include label in the input text
        
        # Tokenize input (label is included in the input)
        model_inputs = tokenizer(input_text, max_length=max_length, truncation=True, padding=True, return_tensors="pt")
        
        # Set 'labels' to be identical to 'input_ids'
        model_inputs["labels"] = model_inputs["input_ids"]
        
        yield model_inputs

def process_dataset(dataset, text_columns: Union[str, List[str]], label_column: str, tokenizer: AutoTokenizer, max_length: int, batch_size: int):
    def generate_processed_examples():
        for examples in dataset["train"]:
            yield from preprocess_function(examples, text_columns, label_column, tokenizer, max_length)
    
    train_dataset = torch.utils.data.IterableDataset(generate_processed_examples)
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, collate_fn=default_data_collator, pin_memory=True
    )
    
    return train_dataloader



def calculate_perplexity(model: torch.nn.Module, dataloader: DataLoader) -> float:
    """
    Calculate the perplexity loss function for a given model and dataloader.

    Args:
        model (torch.nn.Module): The trained language model.
        dataloader (DataLoader): The dataloader containing the test data.

    Returns:
        float: The perplexity loss value.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids']
            labels = batch['labels']

            # Move input_ids and labels to the same device as the model
            input_ids = input_ids.to(model.device)
            labels = labels.to(model.device)

            # Forward pass
            outputs = model(input_ids=input_ids, labels=labels)
            logits = outputs.logits

            # Calculate cross-entropy loss
            loss_fn = CrossEntropyLoss(ignore_index=-100)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            # Accumulate loss and token count
            total_loss += loss.item() * input_ids.numel()
            total_tokens += input_ids.numel()

    # Calculate perplexity
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    return perplexity





def train_model(model: nn.Module, train_dataloader: DataLoader, learning_rate: float, weight_decay: float, num_epochs: int) -> Tuple[nn.Module, Dict[str, list]]:
    """
    Train the model using the specified hyperparameters and visualize the training progress.

    Args:
        model (nn.Module): The model to train.
        train_dataloader (DataLoader): The dataloader containing the training data.
        learning_rate (float): The learning rate for the optimizer.
        weight_decay (float): The weight decay for the optimizer.
        num_epochs (int): The number of epochs to train the model.

    Returns:
        Tuple[nn.Module, Dict[str, list]]: A tuple containing the trained model and a dictionary with the training metrics.
    """
    # Move the model to the available device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Initialize lists to store the training metrics
    train_losses = []
    train_accuracies = []

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for batch in train_dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, labels=labels)
            logits = outputs.logits

            # Calculate loss
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update metrics
            epoch_loss += loss.item() * input_ids.size(0)
            _, predicted = torch.max(logits, -1)
            epoch_correct += (predicted == labels).sum().item()
            epoch_total += labels.numel()

        # Calculate epoch metrics
        epoch_loss /= len(train_dataloader.dataset)
        epoch_accuracy = epoch_correct / epoch_total

        # Store epoch metrics
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)

        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_accuracy:.4f}")

    # Visualize training progress
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

    return model, {'train_losses': train_losses, 'train_accuracies': train_accuracies}



# Example usage
model_name_or_path = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
model = AutoModelForCausalLM.from_pretrained(
            'gpt2',
            pad_token_id=tokenizer.eos_token_id
        )
max_length = 100
text_columns =  ['prompt',]
label_column = 'act'
batch_size = 10
dataset=load_and_prepare_dataset("fka/awesome-chatgpt-prompts")
# Assuming `dataset` is already defined and contains the necessary columns
train_dataloader = process_dataset(dataset, text_columns, label_column, tokenizer, max_length, batch_size)
learning_rate = 0.001
weight_decay = 0.1
num_epochs = 2
trained_model, metrics = train_model(model, train_dataloader, learning_rate, weight_decay, num_epochs)
perplexity = calculate_perplexity(model, train_dataloader)
print(f"Perplexity: {perplexity:.2f}")























def train_epoch(model: nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
    """
    Train the model for one epoch.

    Args:
        model (nn.Module): The model to train.
        dataloader (DataLoader): The dataloader containing the training data.
        optimizer (torch.optim.Optimizer): The optimizer for updating the model parameters.
        criterion (nn.Module): The loss function.
        device (torch.device): The device to use for training.

    Returns:
        Tuple[float, float]: The epoch loss and accuracy.
    """
    model.train()
    epoch_loss = 0.0
    epoch_correct = 0
    epoch_total = 0

    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        # Forward pass
        outputs = model(input_ids=input_ids, labels=labels)
        logits = outputs.logits

        # Calculate loss
        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update metrics
        epoch_loss += loss.item() * input_ids.size(0)
        _, predicted = torch.max(logits, -1)
        epoch_correct += (predicted == labels).sum().item()
        epoch_total += labels.numel()

    # Calculate epoch metrics
    epoch_loss /= len(dataloader.dataset)
    epoch_accuracy = epoch_correct / epoch_total

    return epoch_loss, epoch_accuracy

def train_model(model: nn.Module, train_dataloader: DataLoader, learning_rate: float, weight_decay: float, num_epochs: int, checkpoint_dir: str, plot_dir: str) -> Tuple[nn.Module, Dict[str, list]]:
    """
    Train the model using the specified hyperparameters and visualize the training progress.

    Args:
        model (nn.Module): The model to train.
        train_dataloader (DataLoader): The dataloader containing the training data.
        learning_rate (float): The learning rate for the optimizer.
        weight_decay (float): The weight decay for the optimizer.
        num_epochs (int): The number of epochs to train the model.
        checkpoint_dir (str): The directory to save model checkpoints.
        plot_dir (str): The directory to save training plots.

    Returns:
        Tuple[nn.Module, Dict[str, list]]: A tuple containing the trained model and a dictionary with the training metrics.
    """
    # Move the model to the available device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Create the checkpoint and plot directories if they don't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Initialize lists to store the training metrics
    train_losses = []
    train_accuracies = []

    # Training loop
    for epoch in range(num_epochs):
        epoch_loss, epoch_accuracy = train_epoch(model, train_dataloader, optimizer, criterion, device)

        # Store epoch metrics
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)

        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_accuracy:.4f}")

        # Save model checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)

    # Visualize training progress
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plot_path = os.path.join(plot_dir, "training_plot.png")
    plt.savefig(plot_path)

    return model, {'train_losses': train_losses, 'train_accuracies': train_accuracies}




def train_model(model: nn.Module, train_dataloader: DataLoader, learning_rate: float,
                weight_decay: float, num_epochs: int, checkpoint_dir: str, plot_dir: str) -> Tuple[nn.Module, Dict[str, list]]:
    """
    Train the model using the specified hyperparameters and visualize the training progress.

    Args:
        model (nn.Module): The model to train.
        train_dataloader (DataLoader): The dataloader containing the training data.
        learning_rate (float): The learning rate for the optimizer.
        weight_decay (float): The weight decay for the optimizer.
        num_epochs (int): The number of epochs to train the model.
        checkpoint_dir (str): The directory to save the model checkpoints.
        plot_dir (str): The directory to save the training progress plots.

    Returns:
        Tuple[nn.Module, Dict[str, list]]: A tuple containing the trained model and a dictionary with the training metrics.
    """
    # Move the model to the available device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Initialize lists to store the training metrics
    train_losses = []
    train_accuracies = []

    # Create checkpoint and plot directories if they don't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for batch in train_dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, labels=labels)
            logits = outputs.logits

            # Calculate loss
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update metrics
            epoch_loss += loss.item() * input_ids.size(0)
            _, predicted = torch.max(logits, -1)
            epoch_correct += (predicted == labels).sum().item()
            epoch_total += labels.numel()

        # Calculate epoch metrics
        epoch_loss /= len(train_dataloader.dataset)
        epoch_accuracy = epoch_correct / epoch_total

        # Store epoch metrics
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)

        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_accuracy:.4f}")

        # Save model checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), checkpoint_path)

    # Visualize training progress
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plot_path = os.path.join(plot_dir, "training_progress.png")
    plt.savefig(plot_path)
    plt.close()

    return model, {'train_losses': train_losses, 'train_accuracies': train_accuracies}
