
import os
import argparse
import yaml
from pathlib import Path
from typing import Dict, Any, List,Union,Optional
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch
import functools

import numpy as np

from torch.utils.data import IterableDataset
from torch.utils.data.dataloader import DataLoader
import random

from datasets import (load_dataset, 
                      DatasetDict,
                      Dataset,
                      concatenate_datasets
                      )

from transformers import (PreTrainedTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    BitsAndBytesConfig,
    SchedulerType,
    TrainingArguments,
    default_data_collator,
    get_scheduler,      
    set_seed,
                          )
from peft import (
    LoraConfig,
    PeftConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_int8_training,
)
from torch.utils.data import  DataLoader

import os
import torch

# Set the max_split_size_mb environment variable
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:50'

# Clear CUDA cache
torch.cuda.empty_cache()

# Check for reserved memory (optional)
print(f"Reserved memory before training: {torch.cuda.memory_reserved(0) / 1e9} GB")

# ... (rest of your setup and training code) ...

# from model_arguments import training_args
# Define a function to read arguments from a YAML file
def load_arguments_from_yaml(yaml_file_path:str):
    with open(yaml_file_path, 'r') as stream:
        try:
            arguments = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            arguments = {}
    return arguments

#Load the datset
def load_and_prepare_dataset(
    input_source: Union[str, Path, Dict[str, List[Union[str, Path]]]],
    split_ratios: tuple = (0.8, 0.1, 0.1),
    seed: int = 42,
    streaming: bool = False
    ) -> DatasetDict:
    """
    Load a dataset from various input sources and prepare it by splitting into train, test, and eval sets.

    :param input_source: A dataset name, path to a folder, a single file, multiple files, or a dictionary specifying train, test, and eval files.
    :param split_ratios: A tuple containing the ratios for train, test, and eval splits (default is (0.8, 0.1, 0.1)).
    :param seed: A random seed for reproducibility of the split (default is 42).
    :param streaming: Whether to use streaming to handle large files (default is False).
    :return: A DatasetDict containing the split datasets.
    
    Example:
    # Example usage with streaming for large files:
    # dataset_dict = load_and_prepare_dataset({
    #     'train': ['train_file_1.csv', 'train_file_2.csv'],
    #     'test': ['test_file.csv'],
    #     'eval': ['eval_file.csv']
    # }, streaming=True)
    # print(dataset_dict)
    OUTPUT1:
    DatasetDict({
    train: DatasetDict({
        train: Dataset({
            features: ['act', 'prompt'],
            num_rows: 459
        })
    })
    test: DatasetDict({
        train: Dataset({
            features: ['act', 'prompt'],
            num_rows: 459
        })
    })
    eval: DatasetDict({
        train: Dataset({
            features: ['act', 'prompt'],
            num_rows: 153
        })
    })
    })
    EXAMPLE2:
    dataset=load_and_prepare_dataset('fka/awesome-chatgpt-prompts')
    DatasetDict({
    train: Dataset({
        features: ['act', 'prompt'],
        num_rows: 122
    })
    test: Dataset({
        features: ['act', 'prompt'],
        num_rows: 15
    })
    eval: Dataset({
        features: ['act', 'prompt'],
        num_rows: 16
    })
    })
    EXAMPLE3:
    datset_path=load_and_prepare_dataset('/content/awesome-chatgpt-prompts')
    DatasetDict({
    train: Dataset({
        features: ['act', 'prompt'],
        num_rows: 122
    })
    test: Dataset({
        features: ['act', 'prompt'],
        num_rows: 15
    })
    eval: Dataset({
        features: ['act', 'prompt'],
        num_rows: 16
    })
    })

    """
    # Load dataset from different types of input sources
    if isinstance(input_source, (str, Path)):
        # Dataset name, single file or path to folder
        dataset = load_dataset(input_source, streaming=streaming)
        dataset = DatasetDict(dataset)
    elif isinstance(input_source, dict):
        # Dictionary with specified train, test, and eval files
        formats = ['csv', 'json', 'jsonl', 'parquet', 'txt']
        datasets = {}
        for split, files in input_source.items():
            format_detected = None
            for fmt in formats:
                if any(str(file).endswith(fmt) for file in files):
                    format_detected = fmt
                    break
            if format_detected is None:
                raise ValueError(f"No supported file format detected for files: {files}")
            datasets[split] = load_dataset(format_detected, data_files=files, streaming=streaming)
        dataset = DatasetDict(datasets)
    else:
        raise ValueError("Input source should be a dataset name, path to a folder, a single file, multiple files, or a dictionary.")

    # Perform the split if needed and if not in streaming mode
    if not streaming:
        train_size, test_size, eval_size = split_ratios
        assert 0.0 < train_size < 1.0 and 0.0 < test_size < 1.0 and 0.0 < eval_size < 1.0 and (train_size + test_size + eval_size) == 1.0, \
            "Split ratios must be between 0 and 1 and sum up to 1."

        if "train" not in dataset or "test" not in dataset or "eval" not in dataset:
            # Assuming all splits are to be derived from the 'train' dataset
            full_dataset = concatenate_datasets(list(dataset.values())) if isinstance(dataset, dict) else dataset
            split_dataset = full_dataset.train_test_split(train_size=train_size, seed=seed)
            test_eval_split = split_dataset['test'].train_test_split(test_size=test_size / (test_size + eval_size), seed=seed)

            dataset = DatasetDict({
                "train": split_dataset["train"],
                "test": test_eval_split["train"],
                "eval": test_eval_split["test"]
            })

    return dataset

def load_model(model_name_or_path: Union[str,List]) -> AutoModelForCausalLM:
    """
    Function to load a transformers model.
    
    Args:
      model_name_or_path (Union[str, Path]): The name or path of the model.

    Returns:
        model (AutoModelForCausalLM): The loaded model.
    """
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path,token = "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")
    return model


def get_number_of_trainable_parameters(model):
    r"""
    Returns the number of trainable parameters and number of all parameters in the model.
    """
    # note: same as PeftModel.get_nb_trainable_parameters
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        # Due to the design of 4bit linear layers from bitsandbytes
        # one needs to multiply the number of parameters by 2 to get
        # the correct number of parameters
        if param.__class__.__name__ == "Params4bit":
            num_params = num_params * 2

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    print("Total no of training_parameters:",trainable_params)
    print("Total no of parameters is :",all_param)
    print("percantage of trainable parameters is",100*((trainable_params)/(all_param)))
    return trainable_params, all_param


def create_tokenizer(
    tokenizer_name_or_path: Union[str,List] ) -> AutoTokenizer:
    """
    Initializes and returns a tokenizer based on the specified pretrained model or path.

    Args:
        tokenizer_name_or_path (str): The name or path of the tokenizer's pretrained model.

    Returns:
        AutoTokenizer: The initialized tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path,token = "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")
    
    # Set special tokens if they are not already set
    special_tokens = {
        'pad_token': tokenizer.eos_token,
        'bos_token': tokenizer.eos_token,
        'eos_token': tokenizer.eos_token,
        'unk_token': tokenizer.eos_token,
        'sep_token': tokenizer.eos_token,
        'cls_token': tokenizer.eos_token,
        'mask_token':tokenizer.eos_token
    }
    for token_name, token_value in special_tokens.items():
        if getattr(tokenizer, f"{token_name}_id") is None:
            setattr(tokenizer, token_name, token_value)
    
    return tokenizer

# Helper function to get token ids of the special tokens for prefix, suffix and middle for FIM transformations.
@functools.lru_cache(maxsize=None)
def get_fim_token_ids(tokenizer):
    try:
        FIM_PREFIX, FIM_MIDDLE, FIM_SUFFIX, FIM_PAD = tokenizer.special_tokens_map["additional_special_tokens"][1:5]
        suffix_tok_id, prefix_tok_id, middle_tok_id, pad_tok_id = (
            tokenizer.vocab[tok] for tok in [FIM_SUFFIX, FIM_PREFIX, FIM_MIDDLE, FIM_PAD]
        )
    except KeyError:
        suffix_tok_id, prefix_tok_id, middle_tok_id, pad_tok_id = None, None, None, None
    return suffix_tok_id, prefix_tok_id, middle_tok_id, pad_tok_id


## Adapted from https://github.com/bigcode-project/Megatron-LM/blob/6c4bf908df8fd86b4977f54bf5b8bd4b521003d1/megatron/data/gpt_dataset.py
def permute(
    sample,
    np_rng,
    suffix_tok_id,
    prefix_tok_id,
    middle_tok_id,
    pad_tok_id,
    fim_rate=0.5,
    fim_spm_rate=0.5,
    truncate_or_pad=False,
    ):
    """
    Take in a sample (list of tokens) and perform a FIM transformation on it with a probability of fim_rate, using two FIM modes:
    PSM and SPM (with a probability of fim_spm_rate).
    """

    # The if condition will trigger with the probability of fim_rate
    # This means FIM transformations will apply to samples with a probability of fim_rate
    if np_rng.binomial(1, fim_rate):

        # Split the sample into prefix, middle, and suffix, based on randomly generated indices stored in the boundaries list.
        boundaries = list(np_rng.randint(low=0, high=len(sample) + 1, size=2))
        boundaries.sort()

        prefix = np.array(sample[: boundaries[0]], dtype=np.int64)
        middle = np.array(sample[boundaries[0] : boundaries[1]], dtype=np.int64)
        suffix = np.array(sample[boundaries[1] :], dtype=np.int64)

        if truncate_or_pad:
            # calculate the new total length of the sample, taking into account tokens indicating prefix, middle, and suffix
            new_length = suffix.shape[0] + prefix.shape[0] + middle.shape[0] + 3
            diff = new_length - len(sample)

            # trancate or pad if there's a difference in length between the new length and the original
            if diff > 0:
                if suffix.shape[0] <= diff:
                    return sample, np_rng
                suffix = suffix[: suffix.shape[0] - diff]
            elif diff < 0:
                suffix = np.concatenate([suffix, np.full((-1 * diff), pad_tok_id)])

        # With the probability of fim_spm_rateapply SPM variant of FIM transformations
        # SPM: suffix, prefix, middle
        if np_rng.binomial(1, fim_spm_rate):
            new_sample = np.concatenate(
                [
                    [prefix_tok_id, suffix_tok_id],
                    suffix,
                    [middle_tok_id],
                    prefix,
                    middle,
                ]
            )
        # Otherwise, apply the PSM variant of FIM transformations
        # PSM: prefix, suffix, middle
        else:

            new_sample = np.concatenate(
                [
                    [prefix_tok_id],
                    prefix,
                    [suffix_tok_id],
                    suffix,
                    [middle_tok_id],
                    middle,
                ]
            )
    else:
        # don't apply FIM transformations
        new_sample = sample

    return list(new_sample), np_rng

# Create an Iterable dataset that returns constant-length chunks of tokens from a stream of text files.
class ConstantLengthDataset(IterableDataset):
    """
    Iterable dataset that returns constant length chunks of tokens from stream of text files.
        Args:
            tokenizer (Tokenizer): The processor used for proccessing the data.
            dataset (dataset.Dataset): Dataset with text files.
            infinite (bool): If True the iterator is reset after dataset reaches end else stops.
            seq_length (int): Length of token sequences to return.
            num_of_sequences (int): Number of token sequences to keep in buffer.
            chars_per_token (int): Number of characters per token used to estimate number of tokens in text buffer.
            fim_rate (float): Rate (0.0 to 1.0) that sample will be permuted with FIM.
            fim_spm_rate (float): Rate (0.0 to 1.0) of FIM permuations that will use SPM.
            seed (int): Seed for random number generator.
            
     Returns:
     # # Create datasets for each column
     # train_datasets = {}
     # eval_datasets = {}
     # for column in columns:
     #     train_datasets[column] = ConstantLengthDataset(
     #         tokenizer,
     #         train_data,
     #         infinite=True,
     #         seq_length=SEQ_LENGTH,
     #         chars_per_token=chars_per_token_dict[column],
     #         content_field=column,
     #         fim_rate=FIM_RATE,
     #         fim_spm_rate=FIM_SPM_RATE,
     #         seed=SEED,
     #     )
     #     eval_datasets[column] = ConstantLengthDataset(
     #         tokenizer,
     #         valid_data,
     #         infinite=False,
     #         seq_length=SEQ_LENGTH,
     #         chars_per_token=chars_per_token_dict[column],
     #         content_field=column,
     #         fim_rate=FIM_RATE,
     #         fim_spm_rate=FIM_SPM_RATE,
     #         seed=SEED,
     #     )
    """

    def __init__(
        self,
        tokenizer,
        dataset,
        infinite=False,
        seq_length=1024,
        num_of_sequences=1024,
        chars_per_token=3.6,
        content_field="content",
        fim_rate=0.5,
        fim_spm_rate=0.5,
        seed=0,
    ):
        self.tokenizer = tokenizer
        self.concat_token_id = tokenizer.eos_token_id
        self.dataset = dataset
        self.seq_length = seq_length
        self.infinite = infinite
        self.current_size = 0
        self.max_buffer_size = seq_length * chars_per_token * num_of_sequences
        self.content_field = content_field
        self.fim_rate = fim_rate
        self.fim_spm_rate = fim_spm_rate
        self.seed = seed

        (
            self.suffix_tok_id,
            self.prefix_tok_id,
            self.middle_tok_id,
            self.pad_tok_id,
        ) = get_fim_token_ids(self.tokenizer)
        if not self.suffix_tok_id and self.fim_rate > 0:
            print("FIM is not supported by tokenizer, disabling FIM")
            self.fim_rate = 0

   
    def __iter__(self):
       
       iterator = iter(self.dataset)
       more_examples = True
       np_rng = np.random.RandomState(seed=self.seed)
       while more_examples:
        buffer, buffer_len = [], 0
        while True:
            if buffer_len >= self.max_buffer_size:
                break
            try:
                # Concatenate all columns into a single text
                example = next(iterator)
                text = ' '.join([str(example[col]) for col in self.dataset.column_names])
                buffer.append(text)
                buffer_len += len(buffer[-1])
            except StopIteration:
                if self.infinite:
                    iterator = iter(self.dataset)
                else:
                    more_examples = False
                    break
            tokenized_inputs = self.tokenizer(buffer, truncation=False)["input_ids"]
            all_token_ids = []
            for tokenized_input in tokenized_inputs:
                # optionally do FIM permutations
                if self.fim_rate > 0:
                    tokenized_input, np_rng = permute(
                        tokenized_input,
                        np_rng,
                        self.suffix_tok_id,
                        self.prefix_tok_id,
                        self.middle_tok_id,
                        self.pad_tok_id,
                        fim_rate=self.fim_rate,
                        fim_spm_rate=self.fim_spm_rate,
                        truncate_or_pad=False,
                    )
                all_token_ids.extend(tokenized_input + [self.concat_token_id])
            examples = []
            for i in range(0, len(all_token_ids), self.seq_length):
                input_ids = all_token_ids[i : i + self.seq_length]
                if len(input_ids) == self.seq_length:
                    examples.append(input_ids)
            random.shuffle(examples)
            for example in examples:
                self.current_size += 1
                yield {
                    "input_ids": torch.LongTensor(example),
                    "labels": torch.LongTensor(example),
                }

def Slip_dataset(dataset):
    return dataset['train'],dataset['test'],dataset['eval']

def Pre_Training_Aguments():
    parser = argparse.ArgumentParser(description="Pre-training on the custom dataset custom model.") 
    parser.add_argument("--model_name_or_path", default=None, type=str,  help="Model name or path of the model or name_of_the_model on Hugging face🙋🏼‍♂️🙋🏼‍♂️")
    parser.add_argument("--output_dir", default=None, type=str,  help="output directory")
    parser.add_argument("--dataset_name_or_path", default=None, type=str,  help="Dataset name or path of the dataset")
    parser.add_argument("--tokenizer_name_or_path", default=None, type=str,  help="The model checkpoint for weights initialization.")
    parser.add_argument("--config_name", default="", type=str, help="Tokenizer_name or path of the tokenizer ")
    parser.add_argument("--max_seq_length", default=1024, type=int, help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--tokenizer_name", default="", type=str, help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--chars_per_token", default=3.5, type=float, help="Generally speaking, a number between 2.0 and 3.5 can be considered good enough.")
    parser.add_argument('--fim_rate', type=float, default=0.5, help='Rate of FIM to apply.')
    parser.add_argument('--fim_spm_rate', type=float, default=0.5, help='Rate of fim_spm_rate to apply.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed. any number')
    # parser.add_argument("--cache_dir", default="", type=str, help="Optional directory to store the pre-
    args = parser.parse_args()
    return args
args=Pre_Training_Aguments()
config_args = load_arguments_from_yaml(yaml_file_path='/content/drive/MyDrive/Fine_tuning_LLMS/training.yml')
for key, value in config_args.items():
    setattr(args, key, value)
dataset=load_and_prepare_dataset(args.dataset_name_or_path)
tokenizer=create_tokenizer(tokenizer_name_or_path=args.tokenizer_name_or_path)
model=load_model(model_name_or_path=args.model_name_or_path)
get_number_of_trainable_parameters(model=model)
train_dataset, test_dataset,eval_dataset=Slip_dataset(dataset=dataset)
train_datasets = ConstantLengthDataset(
    tokenizer,
    train_dataset,
    infinite=True,
    seq_length=args.max_seq_length,
    chars_per_token=args.chars_per_token,
    content_field="content",
    fim_rate=args.fim_rate,
    fim_spm_rate=args.fim_spm_rate,
    seed=args.seed,
)
eval_datasets = ConstantLengthDataset(
    tokenizer,
    eval_dataset,
    infinite=False,
    seq_length=args.max_seq_length,
    chars_per_token=args.chars_per_token,
    content_field="content",
    fim_rate=args.fim_rate,
    fim_spm_rate=args.fim_spm_rate,
    seed=args.seed,
)


training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=250,
    save_total_limit=5,
    per_device_train_batch_size=1,
    warmup_steps=100,
    weight_decay=0.01,
    dataloader_drop_last=True,
    bf16=True,
    logging_steps=100,
    learning_rate=0.001,
    gradient_checkpointing=False,
    push_to_hub=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    remove_unused_columns=True,
    max_steps=10000

)



# def model_parse_arguments():
#     parser = argparse.ArgumentParser(description="Argument Parser for Model Arguments")
    
#     parser.add_argument("--output_dir", type=str, default='', help="Output directory")
#     parser.add_argument("--overwrite_output_dir", action="store_true", help="Overwrite output directory if it exists")
#     parser.add_argument("--do_train", action="store_true", help="Whether to run training")
#     parser.add_argument("--do_eval", action="store_true", help="Whether to run evaluation")
#     parser.add_argument("--do_predict", action="store_true", help="Whether to run prediction")
#     parser.add_argument("--evaluation_strategy", type=str, choices=["no", "steps", "epoch"], default="no", help="Evaluation strategy")
#     parser.add_argument("--prediction_loss_only", action="store_true", help="Compute and log only the prediction loss without training loss")
#     parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Batch size per GPU/TPU core/CPU for training")
#     parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="Batch size per GPU/TPU core/CPU for evaluation")
#     parser.add_argument("--per_gpu_train_batch_size", type=int, help="Deprecated, use per_device_train_batch_size instead")
#     parser.add_argument("--per_gpu_eval_batch_size", type=int, help="Deprecated, use per_device_eval_batch_size instead")
#     parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass")
#     parser.add_argument("--eval_accumulation_steps", type=int, help="Number of steps to accumulate for evaluation")
#     parser.add_argument("--eval_delay", type=float, help="Delay evaluation until after a certain number of steps")
#     parser.add_argument("--learning_rate", type=float, default=0.00005, help="Initial learning rate for training")
#     parser.add_argument("--weight_decay", type=float, default=0, help="Weight decay to apply")
#     parser.add_argument("--adam_beta1", type=float, default=0.9, help="Beta1 hyperparameter for Adam optimizer")
#     parser.add_argument("--adam_beta2", type=float, default=0.999, help="Beta2 hyperparameter for Adam optimizer")
#     parser.add_argument("--adam_epsilon", type=float, default=1e-8, help="Epsilon hyperparameter for Adam optimizer")
#     parser.add_argument("--max_grad_norm", type=float, default=1, help="Max gradient norm")
#     parser.add_argument("--num_train_epochs", type=float, default=3, help="Total number of training epochs to perform")
#     parser.add_argument("--max_steps", type=int, default=-1, help="Total number of training steps to perform")
#     parser.add_argument("--lr_scheduler_type", type=str, choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"], default="linear", help="Learning rate scheduler type")
#     parser.add_argument("--lr_scheduler_kwargs", type=str, help="Additional arguments for learning rate scheduler")
#     parser.add_argument("--warmup_ratio", type=float, default=0, help="Warmup ratio for learning rate scheduler")
#     parser.add_argument("--warmup_steps", type=int, default=0, help="Warmup steps for learning rate scheduler")
#     parser.add_argument("--log_level", type=str, choices=["passive", "active", "debug"], default="passive", help="Log level")
#     parser.add_argument("--log_level_replica", type=str, choices=["warning", "info", "error"], default="warning", help="Log level for replica")
#     parser.add_argument("--log_on_each_node", action="store_true", help="Whether to log on each node")
#     parser.add_argument("--logging_dir", type=str, help="Logging directory")
#     parser.add_argument("--logging_strategy", type=str, choices=["no", "steps", "epoch"], default="steps", help="Logging strategy")
#     parser.add_argument("--logging_first_step", action="store_true", help="Log on the first step")
#     parser.add_argument("--logging_steps", type=float, default=500, help="Log every N steps")
#     parser.add_argument("--logging_nan_inf_filter", action="store_true", help="Filter out NaN and Inf in logs")
#     parser.add_argument("--save_strategy", type=str, choices=["no", "steps", "epoch"], default="steps", help="Saving strategy")
#     parser.add_argument("--save_steps", type=float, default=500, help="Save every N steps")
#     parser.add_argument("--save_total_limit", type=int, help="Limit total amount of saved checkpoints")
#     parser.add_argument("--save_safetensors", type=bool, help="Whether to save safe tensors")
#     parser.add_argument("--save_on_each_node", action="store_true", help="Whether to save on each node")
#     parser.add_argument("--save_only_model", action="store_true", help="Whether to save only model")
#     parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA")
#     parser.add_argument("--use_cpu", action="store_true", help="Use CPU")
#     parser.add_argument("--use_mps_device", action="store_true", help="Use MPS device")
#     parser.add_argument("--seed", type=int, default=42, help="Seed for random number generators")
#     parser.add_argument("--data_seed", type=int, help="Seed for data loaders")
#     parser.add_argument("--jit_mode_eval", action="store_true", help="JIT mode for evaluation")
#     parser.add_argument("--use_ipex", action="store_true", help="Use IPex")
#     parser.add_argument("--bf16", action="store_true", help="Use bfloat16")
#     parser.add_argument("--fp16", action="store_true", help="Use fp16")
#     parser.add_argument("--fp16_opt_level", type=str, default="O1", help="FP16 optimization level")
#     parser.add_argument("--half_precision_backend", type=str, default="auto", help="Half precision backend")
#     parser.add_argument("--bf16_full_eval", action="store_true", help="Use bfloat16 for full evaluation")
#     parser.add_argument("--fp16_full_eval", action="store_true", help="Use fp16 for full evaluation")
#     parser.add_argument("--tf32", type=bool, help="Use TF32")
#     parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
#     parser.add_argument("--ddp_backend", type=str, help="Distributed data parallel backend")
#     parser.add_argument("--tpu_num_cores", type=int, help="Number of TPU cores")
#     parser.add_argument("--tpu_metrics_debug", action="store_true", help="TPU metrics debug")
#     parser.add_argument("--debug", type=str, nargs="+", help="Debug options")
#     parser.add_argument("--dataloader_drop_last", action="store_true", help="Drop last batch in DataLoader")
#     parser.add_argument("--eval_steps", type=float, help="Evaluate every N steps")
#     parser.add_argument("--dataloader_num_workers", type=int, default=0, help="Number of workers for DataLoader")
#     parser.add_argument("--past_index", type=int, default=-1, help="Past index")
#     parser.add_argument("--run_name", type=str, help="Run name")
#     parser.add_argument("--disable_tqdm", action="store_true", help="Disable tqdm progress bar")
#     parser.add_argument("--remove_unused_columns", type=bool, help="Remove unused columns")
#     parser.add_argument("--label_names", type=str, nargs="+", help="Label names")
#     parser.add_argument("--load_best_model_at_end", action="store_true", help="Load best model at end of training")
#     parser.add_argument("--metric_for_best_model", type=str, help="Metric for best model")
#     parser.add_argument("--greater_is_better", type=bool, help="Greater is better for the metric")
#     parser.add_argument("--ignore_data_skip", action="store_true", help="Ignore data skip")
#     parser.add_argument("--fsdp", type=str, nargs="+", choices=["auto_wrap", "auto_dp_3d", "deepspeed"], help="FSDP options")
#     parser.add_argument("--fsdp_min_num_params", type=int, default=0, help="FSDP minimum number of parameters")
#     parser.add_argument("--fsdp_config", type=str, help="FSDP configuration")
#     parser.add_argument("--fsdp_transformer_layer_cls_to_wrap", type=str, help="FSDP transformer layer class to wrap")
#     parser.add_argument("--deepspeed", type=str, help="DeepSpeed configuration")
#     parser.add_argument("--label_smoothing_factor", type=float, default=0, help="Label smoothing factor")
#     parser.add_argument("--optim", type=str, choices=["adamw", "adam", "sgd", "adamax", "adagrad", "adafactor", "fused_lamb", "fused_sgd", "lamb"], default="default_optim", help="Optimizer name")
#     parser.add_argument("--optim_args", type=str, help="Optimizer arguments")
#     parser.add_argument("--adafactor", action="store_true", help="Use AdaFactor optimizer")
#     parser.add_argument("--group_by_length", action="store_true", help="Group by length for DataLoader")
#     parser.add_argument("--length_column_name", type=str, default="length", help="Length column name")
#     parser.add_argument("--report_to", type=str, nargs="+", help="Report to")
#     parser.add_argument("--ddp_find_unused_parameters", action="store_true", help="Find unused parameters in distributed data parallel")
#     parser.add_argument("--ddp_bucket_cap_mb", type=int, help="Distributed data parallel bucket capacity in MB")
#     parser.add_argument("--ddp_broadcast_buffers", action="store_true", help="Broadcast buffers in distributed data parallel")
#     parser.add_argument("--dataloader_pin_memory", action="store_true", help="Pin memory for DataLoader")
#     parser.add_argument("--dataloader_persistent_workers", action="store_true", help="Persistent workers for DataLoader")
#     parser.add_argument("--skip_memory_metrics", action="store_true", help="Skip memory metrics")
#     parser.add_argument("--use_legacy_prediction_loop", action="store_true", help="Use legacy prediction loop")
#     parser.add_argument("--push_to_hub", action="store_true", help="Push to Hub")
#     parser.add_argument("--resume_from_checkpoint", type=str, help="Resume from checkpoint")
#     parser.add_argument("--hub_model_id", type=str, help="Hub model ID")
#     parser.add_argument("--hub_strategy", type=str, choices=["every_save", "checkpoint"], default="every_save", help="Hub strategy")
#     parser.add_argument("--hub_token", type=str, help="Hub token")
#     parser.add_argument("--hub_private_repo", action="store_true", help="Hub private repo")
#     parser.add_argument("--hub_always_push", action="store_true", help="Always push to Hub")
#     parser.add_argument("--gradient_checkpointing", action="store_true", help="Gradient checkpointing")
#     parser.add_argument("--gradient_checkpointing_kwargs", type=str, help="Gradient checkpointing kwargs")
#     parser.add_argument("--include_inputs_for_metrics", action="store_true", help="Include inputs for metrics")
#     parser.add_argument("--fp16_backend", type=str, default="auto", help="FP16 backend")
#     parser.add_argument("--push_to_hub_model_id", type=str, help="Push to Hub model ID")
#     parser.add_argument("--push_to_hub_organization", type=str, help="Push to Hub organization")
#     parser.add_argument("--push_to_hub_token", type=str, help="Push to Hub token")
#     parser.add_argument("--mp_parameters", type=str, help="MP parameters")
#     parser.add_argument("--auto_find_batch_size", action="store_true", help="Auto find batch size")
#     parser.add_argument("--full_determinism", action="store_true", help="Full determinism")
#     parser.add_argument("--torchdynamo", type=str, help="TorchDynamo")
#     parser.add_argument("--ray_scope", type=str, default="last", help="Ray scope")
#     parser.add_argument("--ddp_timeout", type=int, default=1800, help="Distributed data parallel timeout")
#     parser.add_argument("--torch_compile", action="store_true", help="Torch compile")
#     parser.add_argument("--torch_compile_backend", type=str, help="Torch compile backend")
#     parser.add_argument("--torch_compile_mode", type=str, help="Torch compile mode")
#     parser.add_argument("--dispatch_batches", action="store_true", help="Dispatch batches")
#     parser.add_argument("--split_batches", action="store_true", help="Split batches")
#     parser.add_argument("--include_tokens_per_second", action="store_true", help="Include tokens per second")
#     parser.add_argument("--include_num_input_tokens_seen", action="store_true", help="Include number of input tokens seen")
#     parser.add_argument("--neftune_noise_alpha", type=float, help="Neftune noise alpha")
    
#     args = parser.parse_args()
#     return args

    


# def load_arguments_from_yaml(yaml_file_path):
#     with open(yaml_file_path, 'r') as stream:
#         try:
#             arguments = yaml.safe_load(stream)
#         except yaml.YAMLError as exc:
#             print(exc)
#             arguments = {}
#     return arguments


# args=model_parse_arguments()
# config_args = load_arguments_from_yaml(yaml_file_path='/content/drive/MyDrive/File_and_Operations/transformers_peft_trl/Fine-tuning/Training.yml')
# for key, value in config_args.items():
#     setattr(args, key, value)

# training_args=TrainingArguments(
# output_dir=args.output_dir,
# overwrite_output_dir=args.overwrite_output_dir,
# do_train=args.do_train,
# do_eval=args.do_eval,
# do_predict=args.do_predict,
# evaluation_strategy=args.evaluation_strategy,
# prediction_loss_only=args.prediction_loss_only,
# per_device_train_batch_size=args.per_device_train_batch_size,
# per_device_eval_batch_size=args.per_device_eval_batch_size,
# per_gpu_train_batch_size=args.per_gpu_train_batch_size,
# per_gpu_eval_batch_size=args.per_gpu_eval_batch_size,
# gradient_accumulation_steps=args.gradient_accumulation_steps,
# eval_accumulation_steps=args.eval_accumulation_steps,
# eval_delay=args.eval_delay,
# learning_rate=args.learning_rate,
# weight_decay=args.weight_decay,
# adam_beta1=args.adam_beta1,
# adam_beta2=args.adam_beta2,
# adam_epsilon=args.adam_epsilon,
# max_grad_norm=args.max_grad_norm,
# num_train_epochs=args.num_train_epochs,
# max_steps=args.max_steps,
# lr_scheduler_type=args.lr_scheduler_type,
# lr_scheduler_kwargs=args.lr_scheduler_kwargs,
# warmup_ratio=args.warmup_ratio,
# warmup_steps=args.warmup_steps,
# log_level=args.log_level,
# log_level_replica=args.log_level_replica,
# log_on_each_node=args.log_on_each_node,
# logging_dir=args.logging_dir,
# logging_strategy=args.logging_strategy,
# logging_first_step=args.logging_first_step,
# logging_steps=args.logging_steps,
# logging_nan_inf_filter=args.logging_nan_inf_filter,
# save_strategy=args.save_strategy,
# save_steps=args.save_steps,
# save_total_limit=args.save_total_limit,
# save_safetensors=args.save_safetensors,
# save_on_each_node=args.save_on_each_node,
# save_only_model=args.save_only_model,
# no_cuda=args.no_cuda,
# use_cpu=args.use_cpu,
# use_mps_device=args.use_mps_device,
# seed=args.seed,
# data_seed=args.data_seed,
# jit_mode_eval=args.jit_mode_eval,
# use_ipex=args.use_ipex,
# bf16=args.bf16,
# fp16=args.fp16,
# fp16_opt_level=args.fp16_opt_level,
# half_precision_backend=args.half_precision_backend,
# bf16_full_eval=args.bf16_full_eval,
# fp16_full_eval=args.fp16_full_eval,
# tf32=args.tf32,
# local_rank=args.local_rank,
# ddp_backend=args.ddp_backend,
# tpu_num_cores=args.tpu_num_cores,
# tpu_metrics_debug=args.tpu_metrics_debug,
# debug=args.debug,
# dataloader_drop_last=args.dataloader_drop_last,
# eval_steps=args.eval_steps,
# dataloader_num_workers=args.dataloader_num_workers,
# past_index=args.past_index,
# run_name=args.run_name,
# disable_tqdm=args.disable_tqdm,
# remove_unused_columns=args.remove_unused_columns,
# label_names=args.label_names,
# load_best_model_at_end=args.load_best_model_at_end,
# metric_for_best_model=args.metric_for_best_model,
# greater_is_better=args.greater_is_better,
# ignore_data_skip=args.ignore_data_skip,
# fsdp=args.fsdp,
# fsdp_min_num_params=args.fsdp_min_num_params,
# fsdp_config=args.fsdp_config,
# fsdp_transformer_layer_cls_to_wrap=args.fsdp_transformer_layer_cls_to_wrap,
# deepspeed=args.deepspeed,
# label_smoothing_factor=args.label_smoothing_factor,
# optim=args.optim,
# optim_args=args.optim_args,
# adafactor=args.adafactor,
# group_by_length=args.group_by_length,
# length_column_name=args.length_column_name,
# report_to=args.report_to,
# ddp_find_unused_parameters=args.ddp_find_unused_parameters,
# ddp_bucket_cap_mb=args.ddp_bucket_cap_mb,
# ddp_broadcast_buffers= args.ddp_broadcast_buffers,
# dataloader_pin_memory=args.dataloader_pin_memory,
# dataloader_persistent_workers=args.dataloader_persistent_workers,
# skip_memory_metrics=args.skip_memory_metrics,
# use_legacy_prediction_loop=args.use_legacy_prediction_loop,
# push_to_hub=args.push_to_hub,
# resume_from_checkpoint=args.resume_from_checkpoint,
# hub_model_id=args.hub_model_id,
# hub_strategy=args.hub_strategy,
# hub_token=args.hub_token,
# hub_private_repo=args.hub_private_repo,
# hub_always_push=args.hub_always_push,
# gradient_checkpointing=args.gradient_checkpointing,
# gradient_checkpointing_kwargs=args.gradient_checkpointing_kwargs,
# include_inputs_for_metrics=args.include_inputs_for_metrics,
# fp16_backend=args.fp16_backend,
# push_to_hub_model_id=args.push_to_hub_model_id,
# push_to_hub_organization=args.push_to_hub_organization,
# # push_to_hub_token=args.push_to_hub_token,
# mp_parameters=args.mp_parameters,
# auto_find_batch_size=args.auto_find_batch_size,
# full_determinism=args.full_determinism,
# torchdynamo=args.torchdynamo,
# ray_scope=args.ray_scope,
# ddp_timeout=args.ddp_timeout,
# torch_compile=args.torch_compile,
# torch_compile_backend=args.torch_compile_backend,
# torch_compile_mode=args.torch_compile_mode,
# dispatch_batches=args.dispatch_batches,
# split_batches=args.split_batches,
# include_tokens_per_second=args.include_tokens_per_second,
# include_num_input_tokens_seen=args.include_num_input_tokens_seen,
# neftune_noise_alpha=args.neftune_noise_alpha,
# )


model.to('cuda')

# config = LoraConfig(
#     r=64, lora_alpha=128, lora_dropout=0.0, target_modules=["embed_tokens", "lm_head", "q_proj", "v_proj"]
# )
# model = get_peft_model(model, config)
# print(model.print_trainable_parameters())
# print(model)
trainer=Trainer(
    model=model,
    args=training_args,
    train_dataset=eval_datasets,
    eval_dataset=eval_datasets,
    data_collator=default_data_collator,
)
 



trainer.train()
model.push_to_hub(
        f"HemanthLLMTinyLlama1.1bfulltrained".replace("/", "_"),
        token = "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"
    )
