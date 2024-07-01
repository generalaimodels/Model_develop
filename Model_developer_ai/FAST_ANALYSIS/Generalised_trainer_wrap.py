import os
from typing import Dict, Any, Callable, Optional, Union
from transformers import Trainer, TrainingArguments, PreTrainedModel, PreTrainedTokenizer
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from dataclasses import dataclass, field
import evaluate
import torch
import logging
from torch.utils.data import DataLoader


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelMetrics:
    """Dataclass to store model evaluation metrics."""
    accuracy: float
    f1: float
    precision: float
    recall: float


class MetricsCalculator:
    """Class to calculate various evaluation metrics."""

    @staticmethod
    def compute_metrics(pred) -> Dict[str, float]:
        """Compute and return evaluation metrics."""
        labels = pred.label_ids
        preds = pred.predictions
        loss = pred.predictions[1] if isinstance(pred.predictions, tuple) else None

        # Check if predictions are logits and convert to class predictions if necessary
        if isinstance(preds, tuple):
            preds = preds[0]
        if len(preds.shape) > 1 and preds.shape[-1] > 1:
            preds = np.argmax(preds, axis=-1)

        accuracy_metric = evaluate.load("accuracy")
        f1_metric = evaluate.load("f1")
        precision_metric = evaluate.load("precision")
        recall_metric = evaluate.load("recall")

        # Ensure labels and predictions are 1D arrays
        labels = labels.ravel()
        preds = preds.ravel()

        accuracy = accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]
        f1 = f1_metric.compute(predictions=preds, references=labels, average="weighted")["f1"]
        precision = precision_metric.compute(
            predictions=preds, references=labels, average="weighted"
        )["precision"]
        recall = recall_metric.compute(predictions=preds, references=labels, average="weighted")["recall"]

        metrics = {
            "accuracy": accuracy,
            "f1": f1,
            "precision": precision,
            "recall": recall,
        }

        # Calculate perplexity if loss is available
        if loss is not None:
            perplexity = torch.exp(torch.tensor(loss.mean())).item()
            metrics["perplexity"] = perplexity

        return metrics


@dataclass
class AdvancedModelTrainerConfig:
    """Configuration class for AdvancedModelTrainer."""
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizer
    train_loader: DataLoader
    eval_loader: DataLoader
    test_loader: DataLoader
    trainer_args: TrainingArguments
    optimizer: Optional[Optimizer] = None
    scheduler: Optional[_LRScheduler] = None
    compute_metrics: Callable = field(default_factory=MetricsCalculator.compute_metrics)
    output_dir: str = "model_outputs"


class AdvancedModelTrainer:
    """Advanced model trainer class with various functionalities."""

    def __init__(self, config: AdvancedModelTrainerConfig, eval_steps: int = 100):
        self.config = config
        self.eval_steps = eval_steps 
        self.trainer = self._create_trainer()

    def _create_trainer(self) -> Trainer:
        """Create and return a Trainer instance."""
        return Trainer(
            model=self.config.model,
            args=self.config.trainer_args,
            train_dataset=self.config.train_loader.dataset,
            eval_dataset=self.config.eval_loader.dataset,
            tokenizer=self.config.tokenizer,
            compute_metrics=self.config.compute_metrics,
            optimizers=(self.config.optimizer, self.config.scheduler)
            if self.config.optimizer and self.config.scheduler
            else None,
        )


    def train(self) -> Dict[str, float]:
        """Train the model and return the training metrics."""
        logger.info("Starting model training...")
        train_result = self.trainer.train()
        logger.info("Model training completed.")

        # Store training losses from each epoch
        training_losses = self.trainer.state.log_history  

        # Save the model, tokenizer, and training history
        self.save_model_and_tokenizer()
        self.save_training_history(training_losses)  # Pass losses here

        return train_result.metrics

    def evaluate(self, eval_dataset: Optional[DataLoader] = None) -> Dict[str, float]:
        """Evaluate the model and return the evaluation metrics."""
        logger.info("Starting model evaluation...")
        dataset = eval_dataset.dataset if eval_dataset else self.config.eval_loader.dataset
        eval_results = self.trainer.evaluate(eval_dataset=dataset)
        logger.info("Model evaluation completed.")
        return eval_results

    def test(self) -> Dict[str, float]:
        """Test the model and return the test metrics."""
        logger.info("Starting model testing...")
        test_results = self.trainer.evaluate(eval_dataset=self.config.test_loader.dataset)
        logger.info("Model testing completed.")
        return test_results

    def run_pipeline(self) -> Dict[str, Any]:
        """Run the complete pipeline: training, evaluation, and testing."""
        logger.info("Running complete model pipeline...")
        train_metrics = self.train()
        eval_metrics = self.evaluate()
        test_metrics = self.test()

        pipeline_results = {
            "train_metrics": train_metrics,
            "eval_metrics": eval_metrics,
            "test_metrics": test_metrics,
        }
        logger.info("Model pipeline completed.")
        return pipeline_results

    def predict(self, input_data: Union[str, Dict[str, Any]], batch_size: int = 32) -> Any:
        """Make predictions using the trained model."""
        logger.info("Making predictions...")
        if isinstance(input_data, str):
            # Tokenize the input string
            inputs = self.config.tokenizer(
                input_data, return_tensors="pt", padding=True, truncation=True
            )
            inputs = {k: v.to(self.trainer.args.device) for k, v in inputs.items()}
        elif isinstance(input_data, dict):
            inputs = input_data
        else:
            raise ValueError("Input must be either a string or a dictionary of tensors.")

        # Make sure we're in evaluation mode
        self.trainer.model.eval()

        with torch.no_grad():
            outputs = self.trainer.model(**inputs)

        # Process the outputs based on your model's architecture
        # This example assumes a language model output
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)

        # Convert to numpy for consistency with the Trainer's predict method
        predictions = predictions.cpu().numpy()

        logger.info("Predictions completed.")
        return predictions

    def save_model_and_tokenizer(self):
        """Save the model and tokenizer to the specified output directory."""
        output_dir = os.path.join(self.config.output_dir, "model")
        logger.info(f"Saving model and tokenizer to: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        self.config.model.save_pretrained(output_dir)
        self.config.tokenizer.save_pretrained(output_dir)

    def save_training_history(self, training_losses):  # Accept losses here
        """Save the training history (metrics) to a file."""
        output_dir = os.path.join(self.config.output_dir, "logs")
        os.makedirs(output_dir, exist_ok=True)

        # Save training metrics
        with open(os.path.join(output_dir, "train_metrics.txt"), "w") as f:
            f.write(f"Epoch\tLoss\tAccuracy\tF1\tPrecision\tRecall\n")

            # Extract and format training losses and metrics
            for epoch_data in training_losses:
                epoch = epoch_data.get('epoch', '-')
                loss = epoch_data.get('loss', '-')
                accuracy = epoch_data.get('eval_accuracy', '-')
                f1 = epoch_data.get('eval_f1', '-')
                precision = epoch_data.get('eval_precision', '-')
                recall = epoch_data.get('eval_recall', '-')
                
                f.write(f"{epoch}\t{loss}\t{accuracy}\t{f1}\t{precision}\t{recall}\n")


def create_advanced_model_trainer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    test_loader: DataLoader,
    trainer_args: TrainingArguments,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[_LRScheduler] = None,
    compute_metrics: Optional[Callable] = None,
    output_dir: str = "model_outputs",
    eval_steps: int = 100,
) -> AdvancedModelTrainer:
    """Factory function to create an AdvancedModelTrainer instance."""
    config = AdvancedModelTrainerConfig(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        eval_loader=eval_loader,
        test_loader=test_loader,
        trainer_args=trainer_args,
        optimizer=optimizer,
        scheduler=scheduler,
        compute_metrics=compute_metrics or MetricsCalculator.compute_metrics,
        output_dir=output_dir,
    )
    return AdvancedModelTrainer(config, eval_steps=eval_steps)
