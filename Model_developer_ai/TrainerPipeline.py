import logging
import os
from typing import Dict, Optional
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
import plotly.graph_objects as go
from PIL import Image, ImageDraw, ImageFont

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AdvancedModelTrainer:
    def __init__(self, model: nn.Module, device: str):
        """
        Initialize the AdvancedModelTrainer.

        Args:
            model: The neural network model to train.
            device: The device to use for training ('cuda' or 'cpu').
        """
        self.model = model
        self.device = device
        self.model.to(self.device)

    def create_logo(self, text: str, output_path: str = "logo.png"):
        """
        Creates a simple logo with the given text and saves it as an image.

        Args:
            text: The text to display on the logo.
            output_path: The path to save the logo image.
        """
        try:
            width, height = 400, 100
            background_color = (255, 255, 255)  # White background
            image = Image.new("RGB", (width, height), background_color)

            font_path = "arial.ttf"
            font_size = 40
            try:
                font = ImageFont.truetype(font_path, font_size)
            except IOError:
                logging.warning(f"Font not found at '{font_path}', using default font.")
                font = ImageFont.load_default()

            draw = ImageDraw.Draw(image)
            text_width, text_height = draw.textsize(text, font=font)
            text_position = ((width - text_width) // 2, (height - text_height) // 2)
            text_color = (0, 0, 0)  # Black text
            draw.text(text_position, text, fill=text_color, font=font)

            image.save(output_path)
            logging.info(f"Logo image saved to '{output_path}'")
        except Exception as e:
            logging.error(f"Error creating logo: {str(e)}")

    def save_sharded_model(self, save_dir: str, shard_size_mb: int = 10*1024):
        """
        Saves the model's state dictionary in shards.

        Args:
            save_dir: The directory to save the shards.
            shard_size_mb: The maximum size (in MB) of each shard file.
        """
        try:
            os.makedirs(save_dir, exist_ok=True)
            state_dict = self.model.state_dict()

            shard_count = 1
            current_shard_size = 0
            shard = {}

            for key, value in state_dict.items():
                tensor_size_mb = value.numel() * value.element_size() / (1024 * 1024)

                if current_shard_size + tensor_size_mb > shard_size_mb:
                    shard_path = os.path.join(save_dir, f"model_shard_{shard_count:03d}.pt")
                    torch.save(shard, shard_path)
                    logging.info(f"Saved model shard to {shard_path}")

                    shard_count += 1
                    current_shard_size = 0
                    shard = {}

                shard[key] = value
                current_shard_size += tensor_size_mb

            if shard:
                shard_path = os.path.join(save_dir, f"model_shard_{shard_count:03d}.pt")
                torch.save(shard, shard_path)
                logging.info(f"Saved model shard to {shard_path}")
        except Exception as e:
            logging.error(f"Error saving sharded model: {str(e)}")

    def load_sharded_model(self, save_dir: str):
        """
        Loads the model's state dictionary from saved shards.

        Args:
            save_dir: The directory from which to load the shards.
        """
        try:
            state_dict = {}
            for filename in sorted(os.listdir(save_dir)):
                if filename.startswith("model_shard_") and filename.endswith(".pt"):
                    shard_path = os.path.join(save_dir, filename)
                    shard = torch.load(shard_path)
                    state_dict.update(shard)
            self.model.load_state_dict(state_dict)
            logging.info(f"Loaded sharded model from {save_dir}")
        except Exception as e:
            logging.error(f"Error loading sharded model: {str(e)}")

    def train(self, train_loader: DataLoader, optimizer: torch.optim.Optimizer,
              scheduler: torch.optim.lr_scheduler._LRScheduler, epochs: int,
              save_dir: str = "model_checkpoints") -> Dict:
        """
        Trains the model, logs the training metrics, and saves the best model checkpoints.

        Args:
            train_loader: The DataLoader for the training data.
            optimizer: The optimizer to use for training.
            scheduler: The learning rate scheduler.
            epochs: The number of epochs to train for.
            save_dir: The directory to save model checkpoints.

        Returns:
            A dictionary containing the training history.
        """
        history = {'train_loss': [], 'train_accuracy': [], 'train_perplexity': []}
        best_accuracy = 0.0

        os.makedirs(save_dir, exist_ok=True)

        try:
            for epoch in range(epochs):
                self.model.train()
                total_loss = 0.0
                correct_predictions = 0
                total_samples = 0

                progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
                for batch in progress_bar:
                    optimizer.zero_grad()

                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)

                    try:
                        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    except RuntimeError as e:
                        logging.error(f"RuntimeError during training: {e}")
                        continue

                    loss = outputs.loss
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    total_loss += loss.item() * input_ids.size(0)
                    total_samples += input_ids.size(0)
                    _, predicted = torch.max(outputs.logits, dim=2)
                    correct_predictions += (predicted == labels).sum().item()
                    progress_bar.set_postfix({'loss': loss.item()})

                epoch_loss = total_loss / total_samples
                epoch_accuracy = correct_predictions / total_samples
                epoch_perplexity = torch.exp(torch.tensor(epoch_loss))

                history['train_loss'].append(epoch_loss)
                history['train_accuracy'].append(epoch_accuracy)
                history['train_perplexity'].append(epoch_perplexity.item())
                logging.info(
                    f"Epoch {epoch + 1}/{epochs} - Train Loss: {epoch_loss:.4f} - "
                    f"Train Accuracy: {epoch_accuracy:.4f} - "
                    f"Train Perplexity: {epoch_perplexity:.4f}"
                )

                if epoch_accuracy > best_accuracy:
                    best_accuracy = epoch_accuracy
                    best_model_dir = os.path.join(save_dir, f"best_model_epoch_{epoch + 1}")
                    self.save_sharded_model(best_model_dir, shard_size_mb=1024)

                    model_architecture_path = os.path.join(best_model_dir, "model_architecture.json")
                    with open(model_architecture_path, "w") as f:
                        f.write(self.model.config.to_json_string())
                    logging.info(f"Model architecture saved to {model_architecture_path}")

        except Exception as e:
            logging.error(f"Error during training: {str(e)}")

        return history

    def evaluate(self, data_loader: DataLoader) -> Dict:
        """
        Evaluates the model and logs the evaluation metrics.

        Args:
            data_loader: The DataLoader for the evaluation data.

        Returns:
            A dictionary containing the evaluation loss, accuracy, and perplexity.
        """
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        try:
            with torch.no_grad():
                for batch in tqdm(data_loader, desc="Evaluation", leave=False):
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)

                    try:
                        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    except RuntimeError as e:
                        logging.error(f"RuntimeError during evaluation: {e}")
                        continue

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

        except Exception as e:
            logging.error(f"Error during evaluation: {str(e)}")
            return {}

    def plot_training_metrics(self, history: Dict, output_path: str = "training_metrics.html"):
        """
        Plots the training curves for loss, accuracy, and perplexity using Plotly.

        Args:
            history: The training history dictionary containing loss, accuracy, and perplexity.
            output_path: The path to save the generated plot.
        """
        try:
            epochs = list(range(1, len(history['train_loss']) + 1))

            fig = go.Figure()

            fig.add_trace(go.Scatter(x=epochs, y=history['train_loss'], mode='lines', name='Training Loss'))
            fig.add_trace(go.Scatter(x=epochs, y=history['train_accuracy'], mode='lines', name='Training Accuracy'))
            fig.add_trace(go.Scatter(x=epochs, y=history['train_perplexity'], mode='lines', name='Training Perplexity'))

            fig.update_layout(
                title='Training Metrics',
                xaxis=dict(title='Epoch'),
                yaxis=dict(title='Value'),
                legend_title_text='Metric'
            )

            fig.write_html(output_path)
            logging.info(f"Training metrics plot saved to {output_path}")

        except Exception as e:
            logging.error(f"Error plotting training metrics: {str(e)}")

    def load_model_for_inference(self, model_dir: str) -> Optional[nn.Module]:
        """
        Loads a sharded model for inference.

        Args:
            model_dir: The directory containing the sharded model weights.

        Returns:
            The loaded model, or None if loading fails.
        """
        try:
            self.load_sharded_model(model_dir)
            self.model.to(self.device)
            self.model.eval()
            logging.info(f"Model loaded from {model_dir} for inference.")
            return self.model
        except Exception as e:
            logging.error(f"Error loading model for inference: {str(e)}")
            return None

    def __str__(self):
        return f"AdvancedModelTrainer(model={self.model.__class__.__name__}, device={self.device})"

    def __repr__(self):
        return self.__str__()
    


def create_logo(text: str, output_path: str = "logo.png") -> None:
    """
    Creates a simple logo with the given text and saves it as an image.

    Args:
        text: The text to display on the logo.
        output_path: The path to save the logo image.
    """
    width, height = 400, 100
    background_color = (255, 255, 255)
    image = Image.new("RGB", (width, height), background_color)

    font_path = "arial.ttf"
    font_size = 40

    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        logging.warning(f"Font not found at '{font_path}', using default font.")
        font = ImageFont.load_default()

    draw = ImageDraw.Draw(image)
    text_width, text_height = draw.textsize(text, font=font)
    text_position = ((width - text_width) // 2, (height - text_height) // 2)
    
    text_color = (0, 0, 0)
    draw.text(text_position, text, fill=text_color, font=font)
    
    image.save(output_path)
    logging.info(f"Logo image saved to '{output_path}'")

def save_sharded_model(model: nn.Module, save_dir: str, shard_size_mb: int = 1024) -> None:
    """
    Saves a model's state dictionary in shards.

    Args:
        model: The model whose state_dict needs to be saved.
        save_dir: The directory to save the shards.
        shard_size_mb: The maximum size (in MB) of each shard file.
    """
    os.makedirs(save_dir, exist_ok=True)
    state_dict = model.state_dict()

    # Determine shard size and initialize variables
    shard = {}
    current_shard_size = 0
    shard_count = 1

    for key, value in state_dict.items():
        tensor_size_mb = value.numel() * value.element_size() / (1024 * 1024)
        
        if current_shard_size + tensor_size_mb > shard_size_mb:
            shard_path = os.path.join(save_dir, f"model_shard_{shard_count:03d}.pt")
            torch.save(shard, shard_path)
            logging.info(f"Saved model shard to {shard_path}")

            shard_count += 1
            current_shard_size = 0
            shard = {}

        shard[key] = value
        current_shard_size += tensor_size_mb

    if shard:
        shard_path = os.path.join(save_dir, f"model_shard_{shard_count:03d}.pt")
        torch.save(shard, shard_path)
        logging.info(f"Saved model shard to {shard_path}")

def load_sharded_model(model: nn.Module, save_dir: str) -> None:
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
          ) -> Dict[str, list]:
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
    history = {'train_loss': [], 'train_accuracy': [], 'train_perplexity': []}

    best_accuracy = 0.0
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_loss, correct_predictions, total_samples = 0.0, 0, 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
        for batch in progress_bar:
            optimizer.zero_grad()
            input_ids, attention_mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)

            try:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            except RuntimeError as e:
                logging.error(f"RuntimeError during training: {e}")
                continue
            
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item() * input_ids.size(0)
            total_samples += input_ids.size(0)
            _, predicted = torch.max(outputs.logits, dim=2)
            correct_predictions += (predicted == labels).sum().item()
            progress_bar.set_postfix({'loss': loss.item()})

        epoch_loss = total_loss / total_samples
        epoch_accuracy = correct_predictions / total_samples
        epoch_perplexity = torch.exp(torch.tensor(epoch_loss))

        history['train_loss'].append(epoch_loss)
        history['train_accuracy'].append(epoch_accuracy)
        history['train_perplexity'].append(epoch_perplexity.item())

        logging.info(
            f"Epoch {epoch + 1}/{epochs} - Train Loss: {epoch_loss:.4f} - "
            f"Train Accuracy: {epoch_accuracy:.4f} - "
            f"Train Perplexity: {epoch_perplexity:.4f}"
        )

        if epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
            best_model_dir = os.path.join(save_dir, f"best_model_epoch_{epoch + 1}")
            save_sharded_model(model, best_model_dir, shard_size_mb=1024)
            model_architecture_path = os.path.join(best_model_dir, "model_architecture.json")
            with open(model_architecture_path, "w") as f:
                f.write(model.config.to_json_string())
            logging.info(f"Model architecture saved to {model_architecture_path}")

    return history

def evaluate(model: nn.Module, data_loader: DataLoader, device: str) -> Dict[str, float]:
    """
    Evaluates the model and logs the evaluation metrics.

    Args:
        model: The model to be evaluated.
        data_loader: The DataLoader for the evaluation data.
        device: The device to evaluate on (e.g., 'cuda' or 'cpu').

    Returns:
        A dictionary containing the evaluation loss, accuracy, and perplexity.
    """
    model.eval()
    total_loss, correct_predictions, total_samples = 0.0, 0, 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluation", leave=False):
            input_ids, attention_mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)

            try:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            except RuntimeError as e:
                logging.error(f"RuntimeError during evaluation: {e}")
                continue
            
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

    return {'eval_loss': epoch_loss, 'eval_accuracy': epoch_accuracy, 'eval_perplexity': epoch_perplexity.item()}

def plot_training_metrics(history: Dict[str, list], output_path: str = "training_metrics.html") -> None:
    """
    Plots the training curves for loss, accuracy, and perplexity using Plotly.

    Args:
        history: The training history dictionary.
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

    fig.update_layout(
        title='Training Metrics',
        xaxis=dict(title='Epoch'),
        yaxis=dict(title='Value'),
        legend_title_text='Metric'
    )

    # Save the plot to an HTML file
    fig.write_html(output_path)
    logging.info(f"Training metrics plot saved to {output_path}")

def load_model_for_inference(model: nn.Module, model_dir: str, device: str) -> nn.Module:
    """
    Loads a sharded model for inference.

    Args:
        model: An instance of the model architecture.
        model_dir: The directory containing the sharded model weights.
        device: The device to load the model onto ('cuda' or 'cpu').

    Returns:
        The model loaded with its state dictionary and moved to the specified device.
    """
    load_sharded_model(model, model_dir)
    model.to(device)
    model.eval()
    logging.info(f"Model loaded from {model_dir} for inference.")
    return model
