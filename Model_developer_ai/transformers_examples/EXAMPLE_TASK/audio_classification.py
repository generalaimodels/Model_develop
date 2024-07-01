#coding=utf-8

import logging
import sys
import warnings
import os
import datasets
import numpy as np
import evaluate
from random import randint
from FAST_ANALYSIS import (
 prepare_datasetsforhemanth,
 get_dataset_info_for_hemanth,
 AiModelForHemanth,
AdvancedPreProcessForHemanth,
AdvancedPipelineForhemanth,
DatasetDictForHemanth,
CustomTrainerForHemanth
)
import time
from datasets import Dataset
from tqdm import tqdm
from typing import Optional, Dict, Any, Union, Callable

from transformers import TrainingArguments
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
def random_subsample(wav: np.ndarray, max_length: float, sample_rate: int = 16000):
    """Randomly sample chunks of `max_length` seconds from the input audio"""
    sample_length = int(round(sample_rate * max_length))
    if len(wav) <= sample_length:
        return wav
    random_offset = randint(0, len(wav) - sample_length - 1)
    return wav[random_offset : random_offset + sample_length]

args={}
dataset=prepare_datasetsforhemanth(
  args.dataset_name
)
dataset=DatasetDictForHemanth(dataset)
raw_dataset_train=dataset['train'].shuffle(seed=args.seed).select(range(args.max_length_train))
raw_dataset_test=dataset['test'].shuffle(seed=args.seed).select(range(args.max_length_test))
raw_dataset_eval=dataset['eval'].shuffle(seed=args.seed).select(range(args.max_length_eval))

if args.audio_column_name not in raw_dataset_train.column_names:
        raise ValueError(
            f"--audio_column_name {args.audio_column_name} not found in dataset '{args.dataset_name}'. "
            "Make sure to set `--audio_column_name` to the correct audio column - one of "
            f"{', '.join(raw_dataset_train.column_names)}."
        )

if args.label_column_name not in raw_dataset_train.column_names:
        raise ValueError(
            f"--label_column_name {args.label_column_name} not found in dataset '{args.dataset_name}'. "
            "Make sure to set `--label_column_name` to the correct text column - one of "
            f"{', '.join(raw_dataset_train.column_names)}."
        )

feature_extractor = AdvancedPreProcessForHemanth(
    model_type="audio",
    pretrained_model_name_or_path=args.feature_extractor_name or args.model_name_or_path,
    return_attention_mask=args.attention_mask,
    cache_dir=args.cache_dir,
    revision=args.model_revision,
    token=args.token,
    trust_remote_code=args.trust_remote_code,
)
feature_extractor=feature_extractor.process_data()

raw_datasets = raw_dataset_train.cast_column(
        args.audio_column_name, datasets.features.Audio(sampling_rate=feature_extractor.sampling_rate)
    )
model_input_name = feature_extractor.model_input_names[0]
def train_transforms(batch):
       """Apply train_transforms across a batch."""
       subsampled_wavs = []
       for audio in batch[args.audio_column_name]:
           wav = random_subsample(
               audio["array"], max_length=args.max_length_seconds, sample_rate=feature_extractor.sampling_rate
           )
           subsampled_wavs.append(wav)
       inputs = feature_extractor(subsampled_wavs, sampling_rate=feature_extractor.sampling_rate)
       output_batch = {model_input_name: inputs.get(model_input_name)}
       output_batch["labels"] = list(batch[args.label_column_name])
       return output_batch

def val_transforms(batch):
    """Apply val_transforms across a batch."""
    wavs = [audio["array"] for audio in batch[args.audio_column_name]]
    inputs = feature_extractor(wavs, sampling_rate=feature_extractor.sampling_rate)
    output_batch = {model_input_name: inputs.get(model_input_name)}
    output_batch["labels"] = list(batch[args.label_column_name])
    return output_batch

labels = raw_dataset_train.features[args.label_column_name].names
label2id, id2label = {}, {}
for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label

metric = evaluate.load("accuracy", cache_dir=args.cache_dir)
def compute_metrics(eval_pred):
       """Computes accuracy on a batch of predictions"""
       predictions = np.argmax(eval_pred.predictions, axis=1)
       return metric.compute(predictions=predictions, references=eval_pred.label_ids)

model=AiModelForHemanth.load_model(
     model_type="audio_classification",
     model_name_or_path=args.model_name_or_path,
     cache_dir=args.cache_dir,
     revision=args.model_revision,
     token=args.token,
     trust_remote_code=args.trust_remote_code,
     ignore_mismatched_sizes=args.ignore_mismatched_sizes,
    )

  # freeze the convolutional waveform encoder
if args.freeze_feature_encoder:
    model.freeze_feature_encoder()



training_args={}


if training_args.do_train:
   raw_dataset_train.set_transform(train_transforms, output_all_columns=False)
if training_args.do_eval:
    raw_dataset_eval.set_transform(val_transforms, output_all_columns=False)

train_model(
     model=model,
     args=training_args,
     train_dataset=raw_dataset_train,
     eval_dataset=raw_dataset_test,
     compute_metrics=compute_metrics,
     image_processor=feature_extractor,

)