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





import os
from typing import Tuple, Dict
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

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
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
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

def train_model_test(model: nn.Module, train_dataloader: DataLoader, learning_rate: float,
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
