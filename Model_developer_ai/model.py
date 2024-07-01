from pre_processing.text import (
DataHemanthGPT2,
DataTokenization,
create_data_loaders,
data_tokenization,
)
from FAST_ANALYSIS import ( 
AiModelForHemanth , 
AdvancedPreProcessForHemanth,
prepare_datasetsforhemanth,
summarizemodelforhemanth,
printmodelsummaryforhemanth,
)
import logging
from typing import Dict
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
import plotly.express as px
import plotly.graph_objects as go

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train(model: nn.Module,
          train_loader: DataLoader,
          optimizer: torch.optim.Optimizer,
          scheduler: torch.optim.lr_scheduler._LRScheduler,
          epochs: int,
          device: str) -> Dict:
    """
    Trains the model and logs the training metrics.

    Args:
        model: The model to be trained.
        train_loader: The DataLoader for the training data.
        optimizer: The optimizer to use for training.
        scheduler: The learning rate scheduler.
        epochs: The number of epochs to train for.
        device: The device to train on (e.g., 'cuda' or 'cpu').

    Returns:
        A dictionary containing the training history 
        (loss, accuracy, perplexity for each epoch).
    """

    history = {'train_loss': [], 
               'train_accuracy': [], 
               'train_perplexity': []
              }

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        progress_bar = tqdm(train_loader, 
                            desc=f"Epoch {epoch+1}/{epochs}", 
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
            f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_loss:.4f} - "
            f"Train Accuracy: {epoch_accuracy:.4f} - "
            f"Train Perplexity: {epoch_perplexity:.4f}"
        )

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


def plot_training_metrics(history: Dict):
    """
    Plots the training curves for loss, accuracy, and perplexity using Plotly,
    combining them into a single HTML file with subplots.

    Args:
        history: The training history dictionary containing 
                 loss, accuracy, and perplexity.
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
    fig.write_html("training_metrics_subplots.html")

    logging.info("Training metrics plot saved to training_metrics_subplots.html")



tokenizer=AdvancedPreProcessForHemanth(
    model_type="text",
    pretrained_model_name_or_path="gpt2",
    revision="main",
    cache_dir=r"E:\LLMS\Fine-tuning\data"
)
tokenizer=tokenizer.process_data()
tokenizer.pad_token = tokenizer.eos_token
model=AiModelForHemanth.load_model(
    model_type="causal_lm",
    model_name_or_path="gpt2",
    cache_dir=r"E:\LLMS\Fine-tuning\data",
)

dataset=prepare_datasetsforhemanth(
    "fka/awesome-chatgpt-prompts",
    cache_dir=r"E:\LLMS\Fine-tuning\data",
 
)
dataset=data_tokenization(
    dataset=dataset,
    tokenizer=tokenizer,
    seq_max_length=50,
    pad_to_max_length=True,
    return_tensors='pt'
)
train_loader,test_loader,eval_loader=create_data_loaders(dataset=dataset,batch_size=10)

output=model(**next(iter(train_loader)))
print(output.logits.shape)


if __name__ == "__main__":
    epochs = 2
    learning_rate = 2e-5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    history = train(model, train_loader, optimizer, scheduler, epochs, device)
    plot_training_metrics(history)
    eval_results = evaluate(model, eval_loader, device)
    print("Evaluation Results:", eval_results)
    test_results = evaluate(model, test_loader, device) 
    print("Testing Results:", test_results) 