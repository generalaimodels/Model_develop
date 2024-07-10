import logging
import os
from typing import Dict, Any,Tuple

from typing import Optional, Tuple

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed
)


import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm
from evaluate import load
from data_processing import (
    load_data,
    create_data_split,
    preprocess_data,
    create_data_loaders
)
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_and_evaluate(
    train_loader: DataLoader,
    test_loader: DataLoader,
    model,
    tokenizer,
    output_dir: str,
    num_epochs: int = 3,
    learning_rate: float = 2e-5,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Dict[str, Any]:
    """
    Train and evaluate a causal language model.

    Args:
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        base_model_path: Path to the pre-trained model
        output_dir: Directory to save model checkpoints
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        device: Device to run the model on

    Returns:
        Dictionary containing training and evaluation metrics
    """
    try:

        model.to(device)

        # Setup optimizer and scheduler
        optimizer = AdamW(model.parameters(), lr=learning_rate)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=(len(train_loader) * num_epochs),
        )

        # Initialize metrics
        best_eval_loss = float('inf')
        metrics = {"train_loss": [], "train_ppl": [], "eval_loss": [], "eval_ppl": []}

        # Training loop
        for epoch in range(num_epochs):
            logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")
            model.train()
            total_loss = 0

            for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                total_loss += loss.item()

                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # Evaluation
            model.eval()
            eval_loss = 0
            eval_preds = []

            with torch.no_grad():
                for batch in tqdm(test_loader, desc="Evaluating"):
                    # batch = {k: v.to(device) for k, v in batch.items()}
                    outputs = model(**batch)
                    loss = outputs.loss
                    eval_loss += loss.item()
                    eval_preds.extend(
                        tokenizer.batch_decode(torch.argmax(outputs.logits, -1).cpu().numpy(), skip_special_tokens=True)
                    )

            # Calculate metrics
            train_epoch_loss = total_loss / len(train_loader)
            eval_epoch_loss = eval_loss / len(test_loader)
            train_ppl = torch.exp(torch.tensor(train_epoch_loss)).item()
            eval_ppl = torch.exp(torch.tensor(eval_epoch_loss)).item()

            metrics["train_loss"].append(train_epoch_loss)
            metrics["train_ppl"].append(train_ppl)
            metrics["eval_loss"].append(eval_epoch_loss)
            metrics["eval_ppl"].append(eval_ppl)

            logger.info(f"Epoch {epoch + 1}: Train PPL: {train_ppl:.2f}, Eval PPL: {eval_ppl:.2f}")

            # Save the best model
            if eval_epoch_loss < best_eval_loss:
                best_eval_loss = eval_epoch_loss
                model_save_path = os.path.join(output_dir, f"best_model_epoch_{epoch + 1}")
                model.save_pretrained(model_save_path)
                logger.info(f"Best model saved to {model_save_path}")

        # Calculate final perplexity using the evaluate module
        perplexity = load("perplexity", module_type="metric")
        results = perplexity.compute(predictions=eval_preds,)
        metrics["final_perplexity"] = results["mean_perplexity"]

        logger.info(f"Final Perplexity: {results['mean_perplexity']:.2f}")

        return metrics

    except Exception as e:
        logger.exception(f"An error occurred during training: {str(e)}")
        raise

def pre_processing_pipeline(file_path: str, model_name: str, max_length: int = 10, stride: int = 2, batch_size: int = 5) -> Tuple[DataLoader, DataLoader, AutoTokenizer]:
    """
    Main function to process the data and create DataLoaders.
    """
    
    if max_length <= 0:
        raise ValueError("max_length must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    dataset = load_data(file_path)
    train_dataset, test_dataset = create_data_split(dataset)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    # special_tokens = {'bos_token': '<bos>', 'eos_token': '<eos>', 'unk_token': '<unk>', 'pad_token': '<pad>'}
    # tokenizer.add_special_tokens(special_tokens)
    system_prompt = "You are an AI assistant tasked with detecting fraud. Analyze the following content carefully."
    train_dataset = preprocess_data(train_dataset, tokenizer, max_length, stride, system_prompt)
    test_dataset = preprocess_data(test_dataset, tokenizer, max_length, stride, system_prompt)
    train_loader, test_loader = create_data_loaders(train_dataset, test_dataset, batch_size)
    logger.info("Data processing completed successfully")
    return train_loader, test_loader



def train_model(
    train_dataset,
    test_dataset,
    model,
    tokenizer,
    output_dir: str,
    num_epochs: int = 3,
    seed: int = 42
) -> None:
    """
    Train a transformer model using the provided datasets and save the weights.

    Args:
        train_dataset (SlidingWindowDataset): Training dataset.
        test_dataset (SlidingWindowDataset): Testing dataset.
        base_model_path (str): Path to the base model.
        output_dir (str): Directory to save the model checkpoints.
        num_epochs (int, optional): Number of training epochs. Defaults to 3.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.
    """
    set_seed(seed)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=5,
        per_device_eval_batch_size=5,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
    )

    try:
        trainer.train()
    except Exception as e:
        logger.exception("An error occurred during training: %s", str(e))
        raise

    trainer.save_model(output_dir)
    logger.info("Model training completed. Weights saved in %s", output_dir)
