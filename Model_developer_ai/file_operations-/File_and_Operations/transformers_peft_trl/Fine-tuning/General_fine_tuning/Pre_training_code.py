import os
import argparse
import yaml
import numpy as np
import torch
import functools

from pathlib import Path
from typing import Dict, Any, List,Union,Optional
from datasets import DatasetDict, Dataset
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.utils.data import IterableDataset
from torch.utils.data.dataloader import DataLoader
import random
from datasets import (load_dataset, 
                      DatasetDict,
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

from torch.utils.data import  DataLoader

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
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
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
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    
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

def Split_dataset(dataset):
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
config_args = load_arguments_from_yaml(yaml_file_path='/content/drive/MyDrive/Fine-tuning/General_fine_tuning/Pre_training.yml')
for key, value in config_args.items():
    setattr(args, key, value)
dataset=load_and_prepare_dataset(args.dataset_name_or_path)
tokenizer=create_tokenizer(tokenizer_name_or_path=args.tokenizer_name_or_path)
model=load_model(model_name_or_path=args.model_name_or_path)
get_number_of_trainable_parameters(model=model)
train_dataset, test_dataset,eval_dataset=Split_dataset(dataset=dataset)
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
    num_train_epochs=20,
    save_total_limit=5,
    per_device_train_batch_size=1,
    warmup_steps=100,
    weight_decay=0.0001,
    dataloader_drop_last=True,
    bf16=False,
    logging_steps=10,
    learning_rate=0.001,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    remove_unused_columns=False,
    max_steps=1000

)



trainer=Trainer(

    model=model,
    args=training_args,
    train_dataset=eval_datasets,
    eval_dataset=eval_datasets,
    data_collator=default_data_collator,
)
 



trainer.train()
