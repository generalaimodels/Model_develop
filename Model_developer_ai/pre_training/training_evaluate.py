
import os
from typing import Tuple, Dict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW, get_scheduler
import plotly.graph_objects as go

def train_epoch(model: nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler._LRScheduler, criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
    """
    Train the model for one epoch.

    Args:
        model (nn.Module): The model to train.
        dataloader (DataLoader): The dataloader containing the training data.
        optimizer (torch.optim.Optimizer): The optimizer for updating the model parameters.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler.
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
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # Update metrics
        epoch_loss += loss.item() * input_ids.size(0)
        _, predicted = torch.max(logits, -1)
        epoch_correct += (predicted == labels).sum().item()
        epoch_total += labels.numel()

    # Calculate epoch metrics
    epoch_loss /= len(dataloader.dataset)
    epoch_accuracy = epoch_correct / epoch_total

    return epoch_loss, epoch_accuracy

def evaluate_model(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
    """
    Evaluate the model on the given dataloader.

    Args:
        model (nn.Module): The model to evaluate.
        dataloader (DataLoader): The dataloader containing the evaluation data.
        criterion (nn.Module): The loss function.
        device (torch.device): The device to use for evaluation.

    Returns:
        Tuple[float, float]: The evaluation loss and accuracy.
    """
    model.eval()
    eval_loss = 0.0
    eval_correct = 0
    eval_total = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, labels=labels)
            logits = outputs.logits

            # Calculate loss
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

            # Update metrics
            eval_loss += loss.item() * input_ids.size(0)
            _, predicted = torch.max(logits, -1)
            eval_correct += (predicted == labels).sum().item()
            eval_total += labels.numel()

    # Calculate evaluation metrics
    eval_loss /= len(dataloader.dataset)
    eval_accuracy = eval_correct / eval_total

    return eval_loss, eval_accuracy

def train_model(model: nn.Module, train_dataloader: DataLoader, eval_dataloader: DataLoader, num_epochs: int, checkpoint_dir: str, plot_dir: str) -> Tuple[nn.Module, Dict[str, list]]:
    """
    Train the model using the specified hyperparameters and visualize the training progress.

    Args:
        model (nn.Module): The model to train.
        train_dataloader (DataLoader): The dataloader containing the training data.
        eval_dataloader (DataLoader): The dataloader containing the evaluation data.
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

    # Define the loss function, optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    # Initialize lists to store the training metrics
    train_losses = []
    train_accuracies = []
    eval_losses = []
    eval_accuracies = []

    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch: {epoch+1}/{num_epochs}")
        print("-" * 20)

        # Train the model for one epoch
        train_loss, train_accuracy = train_epoch(model, train_dataloader, optimizer, lr_scheduler, criterion, device)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # Evaluate the model on the evaluation set
        eval_loss, eval_accuracy = evaluate_model(model, eval_dataloader, criterion, device)
        eval_losses.append(eval_loss)
        eval_accuracies.append(eval_accuracy)

        print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.4f}")
        print(f"Eval Loss: {eval_loss:.4f} | Eval Accuracy: {eval_accuracy:.4f}")
        print()

        # Save the model checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), checkpoint_path)

    # Visualize training progress using plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=train_losses, mode='lines', name='Training Loss'))
    fig.add_trace(go.Scatter(y=train_accuracies, mode='lines', name='Training Accuracy'))
    fig.add_trace(go.Scatter(y=eval_losses, mode='lines', name='Evaluation Loss'))
    fig.add_trace(go.Scatter(y=eval_accuracies, mode='lines', name='Evaluation Accuracy'))
    fig.update_layout(title='Training Progress', xaxis_title='Epoch', yaxis_title='Metric')
    plot_path = os.path.join(plot_dir, "training_plot.html")
    fig.write_html(plot_path)

    return model, {'train_losses': train_losses, 'train_accuracies': train_accuracies, 'eval_losses': eval_losses, 'eval_accuracies': eval_accuracies}

def inference(model: nn.Module, input_text: str, tokenizer) -> str:
    """
    Perform inference on a single input text.

    Args:
        model (nn.Module): The trained model.
        input_text (str): The input text to generate a response for.
        tokenizer: The tokenizer used for encoding and decoding.

    Returns:
        str: The generated response.
    """
    # Encode the input text
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    # Set the model to evaluation mode
    model.eval()

    # Generate the response
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=100,  # Adjust the maximum length as needed
            num_return_sequences=1,
            temperature=0.7,  # Adjust the temperature for controlling randomness
            top_k=50,  # Adjust the top-k value for selecting top tokens
            top_p=0.95,  # Adjust the top-p value for nucleus sampling
            do_sample=True,  # Enable sampling for more diverse responses
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode the generated response
    response = tokenizer.decode(output[0], skip_special_tokens=True)

    return response