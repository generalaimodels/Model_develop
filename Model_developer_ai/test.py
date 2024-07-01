import logging
from FAST_ANALYSIS import (
AiModelForHemanth,
AdvancedPreProcessForHemanth,
prepare_datasetsforhemanth,
get_dataset_info_for_hemanth,
DatasetDictForHemanth,
printmodelsummaryforhemanth,
summarizemodelforhemanth,
CustomTrainerForHemanth,
extract_model_details,
write_to_json_file,
save_model_weights,
 load_model_weights,
create_directory_structure,
)

from transformers import (
    TrainingArguments,
    Trainer,)
dataset=prepare_datasetsforhemanth(
    "food101",
    cache_dir=r"C:\Users\heman\Desktop\Coding\data",
)
get_dataset_info_for_hemanth(dataset)
dataset=DatasetDictForHemanth(dataset)
model_checkpoint = "google/vit-base-patch16-224-in21k" 
image_processor=AdvancedPreProcessForHemanth(
    model_type="image",
    pretrained_model_name_or_path=model_checkpoint,
    cache_dir=r"C:\Users\heman\Desktop\Coding\data",
)
image_processor=image_processor.process_data()
print(image_processor)
labels = dataset["train"].features["label"].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = i
    id2label[i] = label

from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
train_transforms = Compose(
    [
        RandomResizedCrop(image_processor.size["height"]),
        RandomHorizontalFlip(),
        ToTensor(),
        normalize,
    ]
)

val_transforms = Compose(
    [
        Resize(image_processor.size["height"]),
        CenterCrop(image_processor.size["height"]),
        ToTensor(),
        normalize,
    ]
)
def preprocess_train(example_batch):
    """Apply train_transforms across a batch."""
    example_batch["pixel_values"] = [train_transforms(image.convert("RGB")) for image in example_batch["image"]]
    return example_batch


def preprocess_val(example_batch):
    """Apply val_transforms across a batch."""
    example_batch["pixel_values"] = [val_transforms(image.convert("RGB")) for image in example_batch["image"]]
    return example_batch


Model=AiModelForHemanth.load_model(
    model_type="image_classification",
    model_name_or_path=model_checkpoint,
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes=True,  
    cache_dir=r"C:\Users\heman\Desktop\Coding\data",
)


result=summarizemodelforhemanth(Model)
printmodelsummaryforhemanth(result)

train_ds=dataset["train"].shuffle(seed=10).select(range(100))
val_ds=dataset["test"].shuffle(seed=10).select(range(100))
eval_ds=dataset["eval"].shuffle(seed=10).select(range(100))

train_ds.set_transform(preprocess_train)
val_ds.set_transform(preprocess_val)
eval_ds.set_transform(preprocess_val)

model_name = model_checkpoint.split("/")[-1]
batch_size = 1

args = TrainingArguments(
    output_dir=r"C:\Users\heman\Desktop\Coding\data",
    remove_unused_columns=False,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-3,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=batch_size,
    # max_eval_samples=2,
    # fp16=True,
    num_train_epochs=2,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    # push_to_hub=True,
    label_names=["labels"],
)
import numpy as np
import evaluate

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)

import torch


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

import time
import os
import numpy as np
from datasets import Dataset
from tqdm import tqdm
from typing import Optional, Dict, Any, Union, Callable
import logging

# Assuming this logger is defined somewhere in your project
logger = logging.getLogger(__name__)

def train_model(
    model: Any,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    predict_dataset: Optional[Dataset],
    image_processor: Any,
    data_collator: Callable,
    training_args: Any,
    metrics: Dict[str, Callable],
    last_checkpoint: Optional[str] = None
) -> None:
    """
    Train, evaluate, and predict using the given model with visualizations and progress tracking.

    Args:
        model: The model to train.
        train_dataset: The training dataset.
        eval_dataset: The evaluation dataset.
        predict_dataset: The prediction dataset (optional).
        image_processor: The image processor for tokenization.
        data_collator: The data collator function.
        training_args: The training arguments.
        metrics: A dictionary of metric functions.
        last_checkpoint: The path to the last checkpoint to resume training from.

    Returns:
        None
    """
    print(len(eval_dataset))
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
        
        # Training
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

        # Evaluation
        if training_args.do_eval:
            print("\033[93mStarting evaluation...\033[0m")
            eval_metrics = trainer.evaluate()
            max_eval_samples = (
                training_args.max_eval_samples if training_args.max_eval_samples is not None else len(eval_dataset)
            )
            eval_metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
            print("\033[93mEvaluation completed.\033[0m")
            print("\033[93mEvaluation Metrics:\033[0m")
            for metric, value in eval_metrics.items():
                print(f"\033[93m{metric}: {value}\033[0m")
            trainer.log_metrics("eval", eval_metrics)
            trainer.save_metrics("eval", eval_metrics)

        # Prediction
        if training_args.do_predict and predict_dataset is not None:
            print("\033[94mStarting prediction...\033[0m")
            predictions, labels, predict_metrics = trainer.predict(predict_dataset, metric_key_prefix="predict")
            predictions = np.argmax(predictions, axis=-1)

            # Assuming labels are not index -100
            true_predictions = [
                [label for label in prediction if label != -100] for prediction in predictions
            ]

            trainer.log_metrics("predict", predict_metrics)
            trainer.save_metrics("predict", predict_metrics)

            # Save predictions
            output_predictions_file = os.path.join(training_args.output_dir, "predictions.txt")
            if trainer.is_world_process_zero():
                with open(output_predictions_file, "w") as writer:
                    for prediction in true_predictions:
                        writer.write(" ".join(str(pred) for pred in prediction) + "\n")
            print("\033[94mPrediction completed.\033[0m")

        # Model Card and Hub
        kwargs = {
            "finetuned_from": training_args.model_name_or_path,
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
        logger.error(f"Error occurred during training/evaluation: {str(e)}")
        print(f"\033[91mError occurred during training/evaluation: {str(e)}\033[0m")
        raise e

print(len(train_ds))

train_model(
    model=Model,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    predict_dataset=eval_ds,
    image_processor=image_processor,
    data_collator=collate_fn,
    training_args=args,
    metrics=compute_metrics,
    last_checkpoint=None
)



