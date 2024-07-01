""" Pretraining the llms on the user dataset """

import argparse
import yaml
import time
import logging
import math
import json
import os
import torch
import transformers
import datasets
import evaluate
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from pathlib import Path
from plotly.offline import plot
from typing import List, Optional
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from huggingface_hub import HfApi
from torch.utils.data import DataLoader
from transformers import (TrainingArguments, 
                          default_data_collator, 
                          Trainer,
                           SchedulerType, 
                           get_scheduler
                        )

from  transformers.utils import(
    check_min_version,
    send_example_telemetry,
)
from tqdm import tqdm
from transformers.utils.versions import require_version
from dataset_collection import (
    get_files_with_extensions,
    reformat_txt_files,
    write_to_csv,
    process_pdfs_from_csv,
    process_files_txtfile,
    loading_folder_using_datasets
)
from  model_loader_llm import (
    load_model_TEST,
    create_tokenizer,
    calculate_model_parameters,
                               
                               )

from  pre_processing_data import (
    ConstantLengthDataset
)

logger = get_logger(__name__)
def load_arguments_from_yaml(yaml_file_path:str):
    with open(yaml_file_path, 'r') as stream:
        try:
            arguments = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            arguments = {}
    return arguments

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a Transformers model on an image classification dataset")
     # Data folder arguments
    parser.add_argument("--dir_path", type=str, default="E:/LLMS/hemanth/fine_tuning/Fine_tuning_LLMS/Hemanth/Hemanth_LLMs/model_loder/pre_training/")
    parser.add_argument("--dir_output", type=str, default="E:/LLMS/hemanth/fine_tuning/Fine_tuning_LLMS/Hemanth/Hemanth_LLMs/model_loder/output/")
    parser.add_argument("--csv_file_path", type=str, default="E:/LLMS/hemanth/fine_tuning/Fine_tuning_LLMS/Hemanth/Hemanth_LLMs/model_loder/csvfile.csv")
      # Model and tokenizer arguments
    parser.add_argument("--model_name_path", type=str, default="gpt2")
    parser.add_argument("--model_parameters_print", action="store_true", default=False)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--chars_per_token", type=float, default=3.5)
    parser.add_argument("--fim_rate", type=float, default=0.15)
    parser.add_argument("--fim_spm_rate", type=float, default=0.2)
    # parser.add_argument("--seed", type=int, default=10)
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--trust_remote_code",
        type=bool,
        default=False,
        help=(
            "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option "
            "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
            "execute code present on the Hub on your local machine."
        ),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations. '
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--ignore_mismatched_sizes",
        action="store_true",
        help="Whether or not to enable to load a pretrained model whose head dimensions are different.",
    )
    

    args = parser.parse_args()

   

    # if args.push_to_hub or args.with_tracking:
    #     if args.output_dir is None:
    #         raise ValueError(
    #             "Need an `output_dir` to create a repo when `--push_to_hub` or `with_tracking` is specified."
    #         )

    # if args.output_dir is not None:
    #     os.makedirs(args.output_dir, exist_ok=True)

    return args


def main():
    args = parse_args()
    config_args = load_arguments_from_yaml(yaml_file_path='E:/LLMS/hemanth/Model_developer_ai/pre_training/llms_config.yml')
    for key, value in config_args.items():
        setattr(args, key, value)
    send_example_telemetry("pre_training_on_given_LLM_folder", args)
    accelerator_log_kwargs = {}
    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir
    
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)
    logger.info(accelerator.state, main_process_only=True)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
    if args.seed is not None:
        set_seed(args.seed)
    if accelerator.is_main_process:
        if args.push_to_hub:
            repo_name = args.hub_model_id
            if repo_name is None:
                repo_name = Path(args.output_dir).absolute().name
            api = HfApi()
            repo_id = api.create_repo(repo_name, exist_ok=True, token=args.hub_token).repo_id
            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()
    ext_files = get_files_with_extensions(args.dir_path)
    reformat_txt_files([args.dir_output])
    write_to_csv(args.csv_file_path, ext_files)
    process_pdfs_from_csv(csv_path=args.csv_file_path, output_folder=args.dir_output)
    process_files_txtfile(args.dir_path, args.dir_output)
    dataset = loading_folder_using_datasets(folder_path=f'{args.dir_output}/reformatted')
    dataset=dataset['train'].train_test_split(0.15)
    train_dataset=dataset['train']
    test_dataset=dataset['test']
     # Model and tokenizer loading
    model = load_model_TEST(model_name_or_path=args.model_name_path)
    tokenizer = create_tokenizer(args.model_name_path)
    if args.model_parameters_print:
        calculate_model_parameters(model)
    
    train_dataset= ConstantLengthDataset(
        tokenizer,
        train_dataset,
        infinite=True,
        seq_length=args.max_seq_length,
        chars_per_token=args.chars_per_token,
        content_field="text",
        fim_rate=args.fim_rate,
        fim_spm_rate=args.fim_spm_rate,
        seed=args.seed,
    )
    test_dataset = ConstantLengthDataset(
        tokenizer,
        test_dataset,
        infinite=True,
        seq_length=args.max_seq_length,
        chars_per_token=args.chars_per_token,
        content_field="text",
        fim_rate=args.fim_rate,
        fim_spm_rate=args.fim_spm_rate,
        seed=args.seed,
    )
    train_dataloader = DataLoader(
        train_dataset, shuffle=False, collate_fn=default_data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(test_dataset, shuffle=False,collate_fn=default_data_collator, batch_size=args.per_device_eval_batch_size)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=float(args.learning_rate))
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps
        if overrode_max_train_steps
        else args.max_train_steps * accelerator.num_processes,
    )
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)
    if args.with_tracking:
        experiment_config = vars(args)
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("pre_training_on_given_LLM_folder", experiment_config)
    
    metric = evaluate.load("accuracy")
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("┌───────────────────────────────────────────────────────────────────────────┐")
    logger.info("│                          Running training                                 │")
    logger.info("├───────────────────────────────────────────────────────────────────────────┤")
    logger.info("│  %-50s = %12d  │" % ("Num examples", len(train_dataset)))
    logger.info("│  %-50s = %12d  │" % ("Num Epochs", args.num_train_epochs))
    logger.info("│  %-50s = %12d  │" % ("Instantaneous batch size per device", args.per_device_train_batch_size))
    logger.info("│  %-50s = %10d  │" % ("Total train batch size (w. parallel, distributed & accumulation)",total_batch_size))
    logger.info("│  %-50s = %12d  │" % ("Gradient Accumulation steps", args.gradient_accumulation_steps))
    logger.info("│  %-50s = %12d  │" % ("Total optimization steps", args.max_train_steps))
    logger.info("└───────────────────────────────────────────────────────────────────────────┘")
    progress_bar = tqdm(range(args.max_train_steps),desc="Cdacb LLMs",  disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            checkpoint_path = args.resume_from_checkpoint
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)

        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(checkpoint_path)
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)
    progress_bar.update(completed_steps)

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        if args.with_tracking:
            total_loss = 0
        if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader
        for step, batch in enumerate(active_dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                # We keep track of the loss at each epoch
                if args.with_tracking:
                    total_loss += loss.detach().float()
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)

                    if args.push_to_hub and epoch < args.num_train_epochs - 1:
                        accelerator.wait_for_everyone()
                        unwrapped_model = accelerator.unwrap_model(model)
                        unwrapped_model.save_pretrained(
                            args.output_dir,
                            is_main_process=accelerator.is_main_process,
                            save_function=accelerator.save,
                        )
                        if accelerator.is_main_process:
                            model.save_pretrained(args.output_dir)
                            api.upload_folder(
                                commit_message=f"Training in progress epoch {epoch}",
                                folder_path=args.output_dir,
                                repo_id=repo_id,
                                repo_type="model",
                                token=args.hub_token,
                            )

            if completed_steps >= args.max_train_steps:
                break
            
        model.eval()
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            predictions, references = accelerator.gather_for_metrics((predictions, batch["labels"]))
            
            
            # Convert predictions and references to the expected format
            predictions = predictions.cpu().numpy().astype(np.int32)
            references = references.cpu().numpy().astype(np.int32)
            
            # Flatten the arrays into 1-dimensional arrays
            predictions = predictions.flatten()
            references = references.flatten()
            
            metric.add_batch(
                predictions=predictions,
                references=references,
            )

        eval_metric = metric.compute()
        logger.info(f"epoch {epoch}: {eval_metric}")
        # model.eval()
        # for step, batch in enumerate(eval_dataloader):
        #     with torch.no_grad():
        #         outputs = model(**batch)
        #     predictions = outputs.logits.argmax(dim=-1)
        #     predictions, references = accelerator.gather_for_metrics((predictions, batch["labels"]))
        #     print(predictions.shape, references.shape)
            
        #     # Convert predictions and references to the expected format
        #     predictions = predictions.cpu().numpy().astype(np.int32)
        #     references = references.cpu().numpy().astype(np.int32)
            
        #     metric.add_batch(
        #         predictions=list(predictions),
        #         references=list(references),
        #     )

        # eval_metric = metric.compute()
        # logger.info(f"epoch {epoch}: {eval_metric}")

        # model.eval()
        # for step, batch in enumerate(eval_dataloader):
        #     with torch.no_grad():
        #         outputs = model(**batch)
        #     predictions = outputs.logits.argmax(dim=-1)
        #     predictions, references = accelerator.gather_for_metrics((predictions, batch["labels"]))
        #     metric.add_batch(
        #         predictions=predictions,
        #         references=references,
        #     )

        # eval_metric = metric.compute()
        # logger.info(f"epoch {epoch}: {eval_metric}")

        if args.with_tracking:
            accelerator.log(
                {
                    "accuracy": eval_metric,
                    "train_loss": total_loss.item() / len(train_dataloader),
                    "epoch": epoch,
                    "step": completed_steps,
                },
                step=completed_steps,
            )

        if args.push_to_hub and epoch < args.num_train_epochs - 1:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
            )
            if accelerator.is_main_process:
                model.save_pretrained(args.output_dir)
                api.upload_folder(
                    commit_message=f"Training in progress epoch {epoch}",
                    folder_path=args.output_dir,
                    repo_id=repo_id,
                    repo_type="model",
                    token=args.hub_token,
                )

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)

    if args.with_tracking:
        accelerator.end_training()

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
        if accelerator.is_main_process:
            model.save_pretrained(args.output_dir)
            if args.push_to_hub:
                api.upload_folder(
                    commit_message="End of training",
                    folder_path=args.output_dir,
                    repo_id=repo_id,
                    repo_type="model",
                    token=args.hub_token,
                )
            all_results = {f"eval_{k}": v for k, v in eval_metric.items()}
            with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
                json.dump(all_results, f)
    

if __name__ == "__main__":
    main()
    
