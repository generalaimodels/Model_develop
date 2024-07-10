import os
import shutil
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from tqdm import tqdm
from transformers import PreTrainedTokenizer
from datasets import load_metric
import logging

class AdvancedTrainer:
    def __init__(self, args, accelerator, model, tokenizer: PreTrainedTokenizer, 
                 train_dataloader, optimizer, lr_scheduler, eval_dataloader=None):
        self.args = args
        self.accelerator = accelerator
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.eval_dataloader = eval_dataloader
        
        self.logger = logging.getLogger(__name__)
        self.setup_logging()

    def setup_logging(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def train(self) -> Tuple[int, float]:
        try:
            total_batch_size = (self.args.per_device_train_batch_size * 
                                self.accelerator.num_processes * 
                                self.args.gradient_accumulation_steps)

            self.logger.info("***** Running training *****")
            self.logger.info(f"  Num examples = {self.args.num_examples['train']}")
            self.logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size}")
            self.logger.info(f"  Total train batch size = {total_batch_size}")
            self.logger.info(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
            self.logger.info(f"  Total optimization steps = {self.args.max_steps}")

            progress_bar = tqdm(range(self.args.max_steps), disable=not self.accelerator.is_local_main_process)

            checkpoints: Optional[np.ndarray] = None
            eval_results: Optional[np.ndarray] = None
            best_checkpoint: Optional[str] = None
            best_eval_result: Optional[float] = None
            early_stopping_patience_counter = 0
            should_training_stop = False
            epoch = 0
            completed_steps = 0
            train_loss = 0.0
            self.model.zero_grad()

            for _ in range(self.args.num_train_epochs):
                epoch += 1
                self.model.train()
                for step, batch in enumerate(self.train_dataloader):
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    loss = loss / self.args.gradient_accumulation_steps
                    self.accelerator.backward(loss)
                    train_loss += loss.item()

                    if step % self.args.gradient_accumulation_steps == 0 or step == len(self.train_dataloader) - 1:
                        self.optimizer.step()
                        self.lr_scheduler.step()
                        self.optimizer.zero_grad()
                        progress_bar.update(1)
                        completed_steps += 1

                        if (self.eval_dataloader is not None and 
                            self.args.eval_strategy == "steps" and 
                            self.args.eval_steps > 0 and 
                            completed_steps % self.args.eval_steps == 0):
                            
                            self.accelerator.wait_for_everyone()
                            new_checkpoint = f"checkpoint-steps-{completed_steps}"
                            new_eval_result = self.evaluate(new_checkpoint)
                            self.logger.info(f"Evaluation result at step {completed_steps}: "
                                             f"{self.args.eval_metric} = {new_eval_result}")

                            checkpoints, eval_results, best_checkpoint, best_eval_result, early_stopping_patience_counter = \
                                self.update_checkpoints(new_checkpoint, new_eval_result, checkpoints, eval_results, 
                                                        best_checkpoint, best_eval_result, early_stopping_patience_counter)

                            if early_stopping_patience_counter >= self.args.early_stopping_patience:
                                should_training_stop = True

                    if completed_steps >= self.args.max_steps or should_training_stop:
                        break

                if completed_steps >= self.args.max_steps or should_training_stop:
                    break

                if self.eval_dataloader is not None and self.args.eval_strategy == "epoch":
                    self.accelerator.wait_for_everyone()
                    new_checkpoint = f"checkpoint-epoch-{epoch}"
                    new_eval_result = self.evaluate(new_checkpoint)
                    self.logger.info(f"Evaluation result at epoch {epoch}: "
                                     f"{self.args.eval_metric} = {new_eval_result}")

                    checkpoints, eval_results, best_checkpoint, best_eval_result, early_stopping_patience_counter = \
                        self.update_checkpoints(new_checkpoint, new_eval_result, checkpoints, eval_results, 
                                                best_checkpoint, best_eval_result, early_stopping_patience_counter)

                    if early_stopping_patience_counter >= self.args.early_stopping_patience:
                        should_training_stop = True

            self.save_best_checkpoint(best_checkpoint)

            return completed_steps, train_loss / completed_steps

        except Exception as e:
            self.logger.error(f"An error occurred during training: {str(e)}")
            raise

    def evaluate(self, checkpoint: str) -> float:
        try:
            eval_metric = load_metric(self.args.eval_metric)
            eval_loss = 0.0
            all_predictions = None
            all_references = None

            self.model.eval()
            for _, batch in enumerate(self.eval_dataloader):
                with torch.no_grad():
                    outputs = self.model(**batch)

                eval_loss += outputs.loss.item()
                predictions = outputs.logits.argmax(dim=-1)
                predictions = self.accelerator.gather(predictions)

                if all_predictions is None:
                    all_predictions = predictions.detach().cpu().numpy()
                else:
                    all_predictions = np.append(all_predictions, predictions.detach().cpu().numpy(), axis=0)

                references = batch["labels"]
                references = self.accelerator.gather(references)
                if all_references is None:
                    all_references = references.detach().cpu().numpy()
                else:
                    all_references = np.append(all_references, references.detach().cpu().numpy(), axis=0)

            eval_metric.add_batch(predictions=all_predictions, references=all_references)
            eval_result = eval_metric.compute()[self.args.eval_metric]

            return eval_result

        except Exception as e:
            self.logger.error(f"An error occurred during evaluation: {str(e)}")
            raise

    def update_checkpoints(self, new_checkpoint: str, new_eval_result: float, 
                           checkpoints: Optional[np.ndarray], eval_results: Optional[np.ndarray], 
                           best_checkpoint: Optional[str], best_eval_result: Optional[float], 
                           early_stopping_patience_counter: int) -> Tuple[np.ndarray, np.ndarray, str, float, int]:
        if checkpoints is None:
            checkpoints = np.array([new_checkpoint])
            eval_results = np.array([new_eval_result])
            best_checkpoint = new_checkpoint
            best_eval_result = new_eval_result
        else:
            if new_eval_result - best_eval_result > self.args.early_stopping_threshold:
                best_checkpoint = new_checkpoint
                best_eval_result = new_eval_result
                early_stopping_patience_counter = 0
            else:
                if new_eval_result == best_eval_result:
                    best_checkpoint = new_checkpoint
                    best_eval_result = new_eval_result
                early_stopping_patience_counter += 1

            checkpoints = np.append(checkpoints, [new_checkpoint])
            eval_results = np.append(eval_results, [new_eval_result])
            sorted_ids = np.argsort(eval_results)
            eval_results = eval_results[sorted_ids]
            checkpoints = checkpoints[sorted_ids]

        if len(checkpoints) > self.args.keep_checkpoint_max:
            checkpoint_to_remove, *checkpoints = checkpoints
            eval_results = eval_results[1:]
            if checkpoint_to_remove != new_checkpoint:
                if self.accelerator.is_main_process:
                    shutil.rmtree(os.path.join(self.args.output_dir, checkpoint_to_remove), ignore_errors=True)
                self.accelerator.wait_for_everyone()

        if new_checkpoint in checkpoints:
            self.save_checkpoint(new_checkpoint)

        return checkpoints, eval_results, best_checkpoint, best_eval_result, early_stopping_patience_counter

    def save_checkpoint(self, checkpoint: str):
        checkpoint_output_dir = os.path.join(self.args.output_dir, checkpoint)
        if self.accelerator.is_main_process:
            if not os.path.exists(checkpoint_output_dir):
                os.makedirs(checkpoint_output_dir)
        self.accelerator.wait_for_everyone()
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.save_pretrained(checkpoint_output_dir, save_function=self.accelerator.save)
        if self.accelerator.is_main_process:
            self.tokenizer.save_pretrained(checkpoint_output_dir)
            self.logger.info(f"Saving model checkpoint to {checkpoint_output_dir}")

    def save_best_checkpoint(self, best_checkpoint: Optional[str]):
        if best_checkpoint is not None:
            self.logger.info(f"Best checkpoint: {best_checkpoint}")
            best_checkpoint_output_dir = os.path.join(self.args.output_dir, best_checkpoint)
            if self.accelerator.is_main_process:
                shutil.move(best_checkpoint_output_dir, os.path.join(self.args.output_dir, "best-checkpoint"))
                shutil.rmtree(best_checkpoint_output_dir, ignore_errors=True)
            self.accelerator.wait_for_everyone()
        else:
            checkpoint_output_dir = os.path.join(self.args.output_dir, "best-checkpoint")
            if not os.path.exists(checkpoint_output_dir):
                os.makedirs(checkpoint_output_dir)

            self.accelerator.wait_for_everyone()
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            unwrapped_model.save_pretrained(checkpoint_output_dir, save_function=self.accelerator.save)
            if self.accelerator.is_main_process:
                self.tokenizer.save_pretrained(checkpoint_output_dir)
                self.logger.info(f"Saving model checkpoint to {checkpoint_output_dir}")

    def run(self) -> Dict[str, float]:
        try:
            completed_steps, average_train_loss = self.train()
            eval_result = self.evaluate("final")
            
            results = {
                "completed_steps": completed_steps,
                "average_train_loss": average_train_loss,
                f"final_{self.args.eval_metric}": eval_result
            }
            
            return results
        
        except Exception as e:
            self.logger.error(f"An error occurred during the training process: {str(e)}")
            raise

import os
import numpy as np
import shutil
import logging
from tqdm import tqdm
from typing import Optional, Tuple
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from accelerate import Accelerator

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Trainer:
    def __init__(self, args, model: PreTrainedModel, tokenizer: PreTrainedTokenizer,
                 train_dataloader: DataLoader, optimizer: Optimizer, lr_scheduler: _LRScheduler,
                 accelerator: Accelerator, eval_dataloader: Optional[DataLoader] = None):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator
        self.eval_dataloader = eval_dataloader

        self.total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
        self.checkpoints = None
        self.eval_results = None
        self.best_checkpoint = None
        self.best_eval_result = None
        self.early_stopping_patience_counter = 0
        self.should_training_stop = False
        self.epoch = 0
        self.completed_steps = 0
        self.train_loss = 0.0

    def run_training(self):
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", self.args.num_examples.get('train', 0))
        logger.info("  Instantaneous batch size per device = %d", self.args.per_device_train_batch_size)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", self.total_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", self.args.max_steps)

        progress_bar = tqdm(range(self.args.max_steps), disable=not self.accelerator.is_local_main_process)

        self.model.zero_grad()

        try:
            for _ in range(self.args.num_train_epochs):
                if self.train_one_epoch(progress_bar):
                    break  # Early stopping or max steps reached
        except Exception as e:
            logger.error("An error occurred during training: %s", str(e))
            raise
        finally:
            progress_bar.close()

        return self.save_best_checkpoint()

    def train_one_epoch(self, progress_bar) -> bool:
        self.epoch += 1
        self.model.train()

        for step, batch in enumerate(self.train_dataloader):
            outputs = self.model(**batch)
            loss = outputs.loss / self.args.gradient_accumulation_steps
            self.accelerator.backward(loss)
            self.train_loss += loss.item()

            if step % self.args.gradient_accumulation_steps == 0 or step == len(self.train_dataloader) - 1:
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                progress_bar.update(1)
                self.completed_steps += 1

                if self.eval_during_training():
                    self.accelerator.wait_for_everyone()
                    if self.check_early_stopping():
                        return True  # Early stopping

            if self.completed_steps >= self.args.max_steps:
                return True  # Max steps reached

        return self.eval_end_of_epoch()

    def eval_during_training(self) -> bool:
        if self.eval_dataloader is not None and self.args.eval_strategy == "steps":
            if self.completed_steps % self.args.eval_steps == 0:
                self.evaluate_and_save_checkpoint("steps", self.completed_steps)
                return True
        return False

    def eval_end_of_epoch(self) -> bool:
        if self.eval_dataloader is not None and self.args.eval_strategy == "epoch":
            self.evaluate_and_save_checkpoint("epoch", self.epoch)
            return True
        return False

    def evaluate_and_save_checkpoint(self, interval: str, checkpoint_id: int):
        new_checkpoint = f"checkpoint-{interval}-{checkpoint_id}"
        new_eval_result = self.evaluate(self.eval_dataloader, new_checkpoint)

        if self.checkpoints is None:
            self.checkpoints = np.array([new_checkpoint])
            self.eval_results = np.array([new_eval_result])
            self.best_checkpoint = new_checkpoint
            self.best_eval_result = new_eval_result
        else:
            self.update_best_checkpoint(new_checkpoint, new_eval_result)
            self.manage_checkpoints(new_checkpoint)

        self.save_checkpoint(new_checkpoint)

    def evaluate(self, dataloader: DataLoader, checkpoint: str) -> float:
        self.accelerator.wait_for_everyone()
        self.model.eval()

        eval_loss = 0.0
        all_predictions = None
        if self.args.has_labels:
            eval_metric = load_metric(self.args.eval_metric)
            all_references = None

        for batch in dataloader:
            with torch.no_grad():
                outputs = self.model(**batch)

            eval_loss += outputs.loss.item()
            predictions = outputs.logits.argmax(dim=-1) if not self.args.is_regression else outputs.logits.squeeze()
            predictions = self.accelerator.gather(predictions)

            if all_predictions is None:
                all_predictions = predictions.detach().cpu().numpy()
            else:
                all_predictions = np.append(all_predictions, predictions.detach().cpu().numpy(), axis=0)

            if self.args.has_labels:
                references = batch["labels"]
                references = self.accelerator.gather(references)
                if all_references is None:
                    all_references = references.detach().cpu().numpy()
                else:
                    all_references = np.append(all_references, references.detach().cpu().numpy(), axis=0)

        eval_result = self.compute_eval_metric(eval_metric, all_predictions, all_references)
        # logger.info("Evaluation result at %s %d: %s = %f", checkpoint, checkpoint_id, self.args.eval_metric, eval_result)

        return eval_result

    def compute_eval_metric(self, eval_metric, predictions, references):
        if self.args.has_labels and eval_metric is not None:
            eval_metric.add_batch(predictions=predictions, references=references)
            return eval_metric.compute()[self.args.eval_metric]
        return 0.0

    def update_best_checkpoint(self, new_checkpoint: str, new_eval_result: float):
        if new_eval_result - self.best_eval_result > self.args.early_stopping_threshold:
            self.best_checkpoint = new_checkpoint
            self.best_eval_result = new_eval_result
            self.early_stopping_patience_counter = 0
        else:
            if new_eval_result == self.best_eval_result:
                self.best_checkpoint = new_checkpoint
                self.best_eval_result = new_eval_result
            self.early_stopping_patience_counter += 1

    def check_early_stopping(self) -> bool:
        if self.early_stopping_patience_counter >= self.args.early_stopping_patience:
            self.should_training_stop = True
        return self.should_training_stop

    def manage_checkpoints(self, new_checkpoint: str):
        sorted_ids = np.argsort(self.eval_results)
        self.eval_results = self.eval_results[sorted_ids]
        self.checkpoints = self.checkpoints[sorted_ids]
        
        if len(self.checkpoints) > self.args.keep_checkpoint_max:
            checkpoint_to_remove, *self.checkpoints = self.checkpoints
            self.eval_results = self.eval_results[1:]
            if checkpoint_to_remove != new_checkpoint:
                if self.accelerator.is_main_process:
                    shutil.rmtree(os.path.join(self.args.output_dir, checkpoint_to_remove), ignore_errors=True)
                self.accelerator.wait_for_everyone()

    def save_checkpoint(self, new_checkpoint: str):
        checkpoint_output_dir = os.path.join(self.args.output_dir, new_checkpoint)
        if self.accelerator.is_main_process and not os.path.exists(checkpoint_output_dir):
            os.makedirs(checkpoint_output_dir)
        
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.save_pretrained(checkpoint_output_dir, save_function=self.accelerator.save)
        
        if self.accelerator.is_main_process:
            self.tokenizer.save_pretrained(checkpoint_output_dir)
            logger.info("Saving model checkpoint to %s", checkpoint_output_dir)

    def save_best_checkpoint(self):
        if self.best_checkpoint is not None:
            logger.info("Best checkpoint: %s", self.best_checkpoint)
            logger.info("Best evaluation result: %s = %f", self.args.eval_metric, self.best_eval_result)
            best_checkpoint_output_dir = os.path.join(self.args.output_dir, self.best_checkpoint)
            if self.accelerator.is_main_process:
                shutil.move(best_checkpoint_output_dir, os.path.join(self.args.output_dir, "best-checkpoint"))
                shutil.rmtree(best_checkpoint_output_dir, ignore_errors=True)
            self.accelerator.wait_for_everyone()
        else:
            checkpoint_output_dir = os.path.join(self.args.output_dir, "best-checkpoint")
            if not os.path.exists(checkpoint_output_dir):
                os.makedirs(checkpoint_output_dir)

            self.accelerator.wait_for_everyone()
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            unwrapped_model.save_pretrained(checkpoint_output_dir, save_function=self.accelerator.save)
            if self.accelerator.is_main_process:
                self.tokenizer.save_pretrained(checkpoint_output_dir)
                logger.info("Saving best model checkpoint to %s", checkpoint_output_dir)
        return self.completed_steps, self.train_loss / self.completed_steps


# Example of how to use the Trainer class
if __name__ == "__main__":
    from transformers import BertTokenizer, BertForSequenceClassification
    from torch.optim import AdamW
    from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
    from datasets import load_metric, load_dataset

    class Args:
        per_device_train_batch_size = 8
        gradient_accumulation_steps = 2
        num_train_epochs = 3
        max_steps = 1000
        eval_strategy = "epoch"  # Can be "steps" or "epoch"
        eval_steps = 10
        eval_metric = "accuracy"
        early_stopping_threshold = 0.01
        early_stopping_patience = 3
        keep_checkpoint_max = 5
        output_dir = "./output"
        is_regression = False
        has_labels = True
        num_examples = {"train": 1000, "eval": 200}

    args = Args()

    # Load dataset
    datasets = load_dataset("glue", "mrpc")
    tokenized_datasets = datasets.map(
        lambda examples: tokenizer(examples['sentence1'], examples['sentence2'], truncation=True, padding="max_length"),
        batched=True
    )

    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation"]

    # DataLoader
    train_dataloader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=SequentialSampler(eval_dataset),
        batch_size=args.per_device_train_batch_size
    )

    # Model and Tokenizer
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Optimizer and Scheduler
    optimizer = AdamW(model.parameters(), lr=5e-5)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

    # Accelerator
    accelerator = Accelerator()

    # Instantiate Trainer and Start Training
    trainer = Trainer(
        args=args,
        model=model,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
        eval_dataloader=eval_dataloader
    )

    completed_steps, avg_train_loss = trainer.run_training()
    logger.info("Training completed. Total steps: %d, Average training loss: %f", completed_steps, avg_train_loss)