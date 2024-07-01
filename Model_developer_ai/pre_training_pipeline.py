import logging
import os
from typing import Tuple, Dict
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image, ImageDraw, ImageFont



logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def create_logo(text: str, output_path: str = "logo.png"):
    """
    Creates a simple logo with the given text and saves it as an image.

    Args:
        text: The text to display on the logo.
        output_path: The path to save the logo image.
    """
    # Create a new image
    width, height = 400, 100
    background_color = (255, 255, 255)  # White background
    image = Image.new("RGB", (width, height), background_color)

    # Load a font
    font_path = "arial.ttf"  # Use a standard font or provide the path to a custom one
    font_size = 40
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        logging.warning(f"Font not found at '{font_path}', using default font.")
        font = ImageFont.load_default()

    # Get a drawing context
    draw = ImageDraw.Draw(image)

    # Calculate text size and position
    text_width, text_height = draw.textsize(text, font=font)
    text_position = ((width - text_width) // 2, (height - text_height) // 2)

    # Draw the text on the image
    text_color = (0, 0, 0)  # Black text
    draw.text(text_position, text, fill=text_color, font=font)

    # Save the image
    image.save(output_path)
    logging.info(f"Logo image saved to '{output_path}'")


def save_sharded_model(model: nn.Module, save_dir: str, shard_size_mb: int = 1024):
    """
    Saves a model's state dictionary in shards.

    Args:
        model: The model whose state_dict needs to be saved.
        save_dir: The directory to save the shards.
        shard_size_mb: The maximum size (in MB) of each shard file.
    """
    os.makedirs(save_dir, exist_ok=True)
    state_dict = model.state_dict()

    shard_count = 1
    current_shard_size = 0
    shard = {}

    for key, value in state_dict.items():
        tensor_size_mb = value.numel() * value.element_size() / (1024 * 1024)

        if current_shard_size + tensor_size_mb > shard_size_mb:
            # Save the current shard and start a new one
            shard_path = os.path.join(save_dir, f"model_shard_{shard_count:03d}.pt")
            torch.save(shard, shard_path)
            logging.info(f"Saved model shard to {shard_path}")

            shard_count += 1
            current_shard_size = 0
            shard = {}

        shard[key] = value
        current_shard_size += tensor_size_mb

    # Save the last shard
    if shard:
        shard_path = os.path.join(save_dir, f"model_shard_{shard_count:03d}.pt")
        torch.save(shard, shard_path)
        logging.info(f"Saved model shard to {shard_path}")


def load_sharded_model(model: nn.Module, save_dir: str):
    """
    Loads a model's state dictionary from saved shards.

    Args:
        model: The model whose state_dict needs to be loaded.
        save_dir: The directory from which to load the shards.
    """
    state_dict = {}
    for filename in sorted(os.listdir(save_dir)):
        if filename.startswith("model_shard_") and filename.endswith(".pt"):
            shard_path = os.path.join(save_dir, filename)
            shard = torch.load(shard_path)
            state_dict.update(shard)
    model.load_state_dict(state_dict)

def train(model: nn.Module,
          train_loader: DataLoader,
          optimizer: torch.optim.Optimizer,
          scheduler: torch.optim.lr_scheduler._LRScheduler,
          epochs: int,
          device: str,
          save_dir: str = "model_checkpoints"
          ) -> Dict:
    """
    Trains the model, logs the training metrics, and saves the best model checkpoints.

    Args:
        model: The model to be trained.
        train_loader: The DataLoader for the training data.
        optimizer: The optimizer to use for training.
        scheduler: The learning rate scheduler.
        epochs: The number of epochs to train for.
        device: The device to train on (e.g., 'cuda' or 'cpu').
        save_dir: The directory to save model checkpoints.

    Returns:
        A dictionary containing the training history
        (loss, accuracy, perplexity for each epoch).
    """

    history = {'train_loss': [],
               'train_accuracy': [],
               'train_perplexity': []
              }

    best_accuracy = 0.0

    # Create the save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        progress_bar = tqdm(train_loader,
                            desc=f"Epoch {epoch + 1}/{epochs}",
                            leave=False
                           )
        for batch in progress_bar:
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            try:
                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                labels=labels
                               )
            except RuntimeError as e:
                logging.error(f"RuntimeError during training: {e}")
                continue  # Skip to the next batch

            loss = outputs.loss

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Update metrics
            total_loss += loss.item() * input_ids.size(0)
            total_samples += input_ids.size(0)
            _, predicted = torch.max(outputs.logits, dim=2)
            correct_predictions += (predicted == labels).sum().item()
            progress_bar.set_postfix({'loss': loss.item()})

        # Calculate epoch metrics
        epoch_loss = total_loss / total_samples
        epoch_accuracy = correct_predictions / total_samples
        epoch_perplexity = torch.exp(torch.tensor(epoch_loss))

        # Log and store metrics
        history['train_loss'].append(epoch_loss)
        history['train_accuracy'].append(epoch_accuracy)
        history['train_perplexity'].append(epoch_perplexity.item())
        logging.info(
            f"Epoch {epoch + 1}/{epochs} - Train Loss: {epoch_loss:.4f} - "
            f"Train Accuracy: {epoch_accuracy:.4f} - "
            f"Train Perplexity: {epoch_perplexity:.4f}"
        )

        # Save the model if it achieves the best accuracy so far
        if epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
            best_model_dir = os.path.join(save_dir, f"best_model_epoch_{epoch + 1}")
            save_sharded_model(model, best_model_dir, shard_size_mb=1024)  # Adjust shard size as needed

            # Save model architecture in JSON format
            model_architecture_path = os.path.join(best_model_dir, "model_architecture.json")
            with open(model_architecture_path, "w") as f:
                f.write(model.config.to_json_string())
            logging.info(f"Model architecture saved to {model_architecture_path}")

    return history


def evaluate(model: nn.Module,
             data_loader: DataLoader,
             device: str
            ) -> Dict:
    """
    Evaluates the model and logs the evaluation metrics.

    Args:
        model: The model to be evaluated.
        data_loader: The DataLoader for the evaluation data.
        device: The device to evaluate on (e.g., 'cuda' or 'cpu').

    Returns:
        A dictionary containing the evaluation loss, accuracy,
        and perplexity.
    """

    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluation", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            try:
                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                labels=labels
                               )
            except RuntimeError as e:
                logging.error(f"RuntimeError during evaluation: {e}")
                continue  # Skip to the next batch

            loss = outputs.loss

            total_loss += loss.item() * input_ids.size(0)
            total_samples += input_ids.size(0)
            _, predicted = torch.max(outputs.logits, 2)
            correct_predictions += (predicted == labels).sum().item()

    epoch_loss = total_loss / total_samples
    epoch_accuracy = correct_predictions / total_samples
    epoch_perplexity = torch.exp(torch.tensor(epoch_loss))

    logging.info(
        f"Evaluation Loss: {epoch_loss:.4f} - "
        f"Evaluation Accuracy: {epoch_accuracy:.4f} - "
        f"Evaluation Perplexity: {epoch_perplexity:.4f}"
    )

    return {'eval_loss': epoch_loss,
            'eval_accuracy': epoch_accuracy,
            'eval_perplexity': epoch_perplexity.item()
           }


def plot_training_metrics(history: Dict, output_path: str = "training_metrics.html"):
    """
    Plots the training curves for loss, accuracy, and perplexity using Plotly,
    combining them into a single HTML file with subplots.

    Args:
        history: The training history dictionary containing
                 loss, accuracy, and perplexity.
        output_path: The path to save the generated plot.
    """

    epochs = list(range(1, len(history['train_loss']) + 1))

    fig = go.Figure()

    # Loss Plot
    fig.add_trace(go.Scatter(
        x=epochs,
        y=history['train_loss'],
        mode='lines',
        name='Training Loss'
    ))

    # Accuracy Plot
    fig.add_trace(go.Scatter(
        x=epochs,
        y=history['train_accuracy'],
        mode='lines',
        name='Training Accuracy'
    ))

    # Perplexity Plot
    fig.add_trace(go.Scatter(
        x=epochs,
        y=history['train_perplexity'],
        mode='lines',
        name='Training Perplexity'
    ))

    # Update layout for subplots
    fig.update_layout(
        title='Training Metrics',
        xaxis=dict(title='Epoch'),
        yaxis=dict(title='Value'),
        legend_title_text='Metric'
    )

    # Save the plot to an HTML file
    fig.write_html(output_path)

    logging.info(f"Training metrics plot saved to {output_path}")

def load_model_for_inference(model: nn.Module, model_dir: str, device: str):
    """Loads a sharded model for inference.

    Args:
        model: An instance of the model architecture.
        model_dir: The directory containing the sharded model weights.
        device: The device to load the model onto ('cuda' or 'cpu').
    """
    load_sharded_model(model, model_dir) 
    model.to(device)
    model.eval()  # Set the model to evaluation mode

    logging.info(f"Model loaded from {model_dir} for inference.")
    return model