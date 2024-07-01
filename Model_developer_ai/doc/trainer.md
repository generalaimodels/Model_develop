```python
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


```

















```python 
import time
from datasets import Dataset
from tqdm import tqdm
from typing import Optional, Dict, Any, Union, Callable

def train_model(
    model: Any,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    image_processor: Any,
    data_collator: Callable,
    training_args: Any,
    metrics: Dict[str, Callable],
    last_checkpoint: Optional[str] = None,
) -> None:
    """
    Train and evaluate the model with visualizations and progress tracking.

    Args:
        model: The model to train.
        train_dataset: The training dataset.
        eval_dataset: The evaluation dataset.
        image_processor: The image processor for tokenization.
        data_collator: The data collator function.
        training_args: The training arguments.
        metrics: A dictionary of metric functions.
        last_checkpoint: The path to the last checkpoint to resume training from.

    Returns:
        None
    """
    try:
        trainer = CustomTrainerForHemanth(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=image_processor,
            data_collator=data_collator,
            compute_metrics=metrics,
        )
        if True:
            checkpoint = None
            if training_args.resume_from_checkpoint is not None:
                checkpoint = training_args.resume_from_checkpoint
            elif last_checkpoint is not None:
                checkpoint = last_checkpoint
            print("\033[92mStarting training...\033[0m")
            start_time = time.time()
            train_progress_bar = tqdm(total=training_args.num_train_epochs, unit="epoch")
            for epoch in range(training_args.num_train_epochs):
                print("\n \n \n ")
                train_result = trainer.train(resume_from_checkpoint=checkpoint)
                checkpoint = None
                train_progress_bar.update(1)
                train_progress_bar.set_postfix(loss=train_result.metrics["train_loss"])
                trainer.log_metrics(f"train_epoch_{epoch}", train_result.metrics)
                trainer.save_metrics(f"train_epoch_{epoch}", train_result.metrics)
                trainer.save_model(f"model_epoch_{epoch}")

            train_progress_bar.close()
            end_time = time.time()
            print(f"\033[92mTraining completed in {end_time - start_time:.2f} seconds.\033[0m")
            trainer.save_state()
        if training_args.do_eval:
            print("\033[93mStarting evaluation...\033[0m")
            eval_metrics = trainer.evaluate()
            print("\033[93mEvaluation completed.\033[0m")
            print("\033[93mEvaluation Metrics:\033[0m")
            for metric, value in eval_metrics.items():
                print(f"\033[93m{metric}: {value}\033[0m")
            trainer.log_metrics("eval", eval_metrics)
            trainer.save_metrics("eval", eval_metrics)
        kwargs = {
            "tasks": "masked-auto-encoding",
            "dataset": training_args.dataset_name,
            "tags": ["masked-auto-encoding"],
        }
        if training_args.push_to_hub:
            print("\033[94mPushing model to hub...\033[0m")
            trainer.push_to_hub(**kwargs)
            print("\033[94mModel pushed to hub.\033[0m")
        else:
            print("\033[94mCreating model card...\033[0m")
            trainer.create_model_card(**kwargs)
            print("\033[94mModel card created.\033[0m")
    except Exception as e:
        print(f"\033[91mError occurred during training/evaluation: {str(e)}\033[0m")
        raise e


```


```python
import plotly.graph_objs as go
from typing import Dict

def plot_metrics(metrics: Dict[str, float]) -> None:
    """
    Plots the metrics using Plotly.

    Args:
        metrics (Dict[str, float]): A dictionary containing metric names and their values.
    """
    try:
        if not metrics:
            raise ValueError("The metrics dictionary is empty.")
        
        # Creating a list of metric names and their corresponding values
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        # Create the plot
        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=metric_names,
            y=metric_values,
            name='Metrics',
            marker_color='indianred'
        ))

        # Adding title and labels
        fig.update_layout(
            title='Metrics Plot',
            xaxis_title='Metrics',
            yaxis_title='Values',
            legend_title='Legend'
        )

        # Show the plot
        fig.show()

    except ValueError as ve:
        print(f"ValueError: {ve}")
    except TypeError as te:
        print(f"TypeError: {te}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

import plotly.graph_objects as go
from typing import Dict, Any


def plot_metrics(metrics: Dict[str, float]) -> None:
    """
    Plots the given metrics using Plotly.

    Parameters:
    metrics (Dict[str, float]): Dictionary containing metric names and their values.

    Returns:
    None
    """
    try:
        # Ensure metrics dictionary is not empty
        if not metrics:
            raise ValueError("The metrics dictionary is empty.")

        # Initialize lists for metric names and their corresponding values
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())

        # Create the bar plot
        fig = go.Figure(
            data=[
                go.Bar(
                    x=metric_names,
                    y=metric_values,
                    text=metric_values,
                    textposition='auto',
                    marker=dict(color='rgb(55, 83, 109)'),
                    name='Metrics'
                )
            ]
        )

        # Update layout for better visuals
        fig.update_layout(
            title='Metrics Overview',
            xaxis=dict(
                title='Metric Names',
                titlefont=dict(size=18),
                tickfont=dict(size=14)
            ),
            yaxis=dict(
                title='Values',
                titlefont=dict(size=18),
                tickfont=dict(size=14)
            ),
            legend=dict(
                x=0.5,
                y=-0.2,
                xanchor='center',
                yanchor='top',
                orientation='h',
                font=dict(size=12)
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            bargap=0.3
        )

        # Show the plot
        fig.show()

    except ValueError as e:
        print(f"ValueError: {str(e)}")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")


if __name__ == "__main__":
    # Example dictionary with metrics
    metrics_dict = {
        'loss': 4.7403,
        'grad_norm': 1.2657,
        'learning_rate': 0.004985,
        'epoch': 0.01
    }

    # Plot the metrics
    plot_metrics(metrics_dict)




import plotly.graph_objects as go
from typing import Dict, List

def create_training_plots(training_data: List[Dict[str, float]]) -> None:
    """
    Generates and displays interactive plots of training metrics using Plotly.

    Args:
        training_data: A list of dictionaries, where each dictionary represents
                       the metrics for a single epoch. Each dictionary should
                       contain keys like 'loss', 'grad_norm', 'learning_rate',
                       and 'epoch'.

    Raises:
        ValueError: If the input data is empty or has an invalid format.
        KeyError: If the expected keys are missing in the input dictionaries.
    """

    if not training_data:
        raise ValueError("Training data cannot be empty.")

    epochs = []
    losses = []
    grad_norms = []
    learning_rates = []

    for data_point in training_data:
        try:
            epochs.append(data_point['epoch'])
            losses.append(data_point['loss'])
            grad_norms.append(data_point['grad_norm'])
            learning_rates.append(data_point['learning_rate'])
        except KeyError as e:
            raise KeyError(f"Missing key in training data: {e}")

    # Create Plotly figure
    fig = go.Figure()

    # Add traces for each metric
    fig.add_trace(go.Scatter(x=epochs, y=losses, mode='lines', name='Loss'))
    fig.add_trace(go.Scatter(x=epochs, y=grad_norms, mode='lines', name='Gradient Norm'))
    fig.add_trace(go.Scatter(x=epochs, y=learning_rates, mode='lines', name='Learning Rate'))

    # Update layout for better presentation
    fig.update_layout(
        title="Training Metrics",
        xaxis_title="Epoch",
        yaxis_title="Value",
        legend_title="Metrics",
        font=dict(family="Arial", size=14),
        hovermode="x unified"  # Show data on hover for all traces at the same x-coordinate
    )

    fig.show()


# Example usage:
example_data = [
    {'loss': 4.7403, 'grad_norm': 1.2657, 'learning_rate': 0.0049, 'epoch': 0.01},
    {'loss': 3.8521, 'grad_norm': 1.1023, 'learning_rate': 0.0048, 'epoch': 0.02},
    {'loss': 3.2195, 'grad_norm': 0.9876, 'learning_rate': 0.0047, 'epoch': 0.03}
    # Add more data points here...
]

create_training_plots(training_data=example_data)
```