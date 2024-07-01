import os
import argparse
import yaml
from pathlib import Path
from typing import Dict, Any, List,Union,Optional
from datasets import DatasetDict, Dataset
from torch.utils.data import Dataset, DataLoader

from datasets import (load_dataset, 
                      DatasetDict,
                      concatenate_datasets
                      )

from transformers import (PreTrainedTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    SchedulerType,
    TrainingArguments,
    default_data_collator,
    get_scheduler,      
    set_seed,
                          )

from torch.utils.data import  DataLoader

Path("__file__")
from model_arguments import ModelArguments,model_parse_arguments

# Define a function to read arguments from a YAML file
def load_arguments_from_yaml(yaml_file_path):
    with open(yaml_file_path, 'r') as stream:
        try:
            arguments = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            arguments = {}
    return arguments


def parse_args():
    parser = argparse.ArgumentParser(description=f"Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--dataset_name_or_path",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
     '--config', type=str, help='Path to the YAML configuration file.'
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv, txt or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv, txt or a json file containing the validation data."
    )
    parser.add_argument(
        "--validation_split_percentage",
        default=5,
        help="The percentage of the train set used as validation set in case there's no validation split",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
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

    parser.add_argument(
        "--ignore_pad_token_for_loss",
        type=bool,
        default=True,
        help="Whether to ignore the tokens corresponding to padded labels in the loss computation or not.",
    )
    parser.add_argument(
        "--max_source_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after "
            "tokenization.Sequences longer than this will be truncated, sequences shorter will be padded."
        ),
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total sequence length for target text after "
            
            "tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
            "during ``evaluate`` and ``predict``."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--no_keep_linebreaks", action="store_true", help="Do not keep line breaks when using TXT files."
    )
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
            "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option"
            "should only be set to `True` for repositories you trust and in which you have read the code, as it will"
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
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--low_cpu_mem_usage",
        action="store_true",
        help=(
            "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
            "If passed, LLM loading time and RAM consumption will be benefited."
        ),
    )
    ##########################
    #   Generation Config    #
    ##########################
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="temperature of 1.0 has no effect, lower tend toward greedy sampling",
    )
    parser.add_argument("--k", type=int, default=40, help="Choose k candidate words")
    parser.add_argument("--p", type=float, default=0.95, help="The sum of probability of candidate words is 0.9 ")

    ##########################
    #        Exp Args        #
    ##########################
    parser.add_argument(
        "--adapter_name_or_path",
        type=str,
        default=None,
        help=(
            "The LoRA adapter checkpoint. Set None if you want to fine-tune from LoftQ."
            "Specify a path if you want to evaluate."
        ),
    )

    args = parser.parse_args()
    
    return args

Main_args=parse_args()
if Main_args.config:
    config_args = load_arguments_from_yaml(Main_args.config)
    for key, value in config_args.items():
        setattr(Main_args, key, value)


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
def preprocess_datasets(dataset_dict: Union[DatasetDict,Dataset], tokenizer: PreTrainedTokenizer, max_length: int) -> DatasetDict:
    """
    Preprocesses text data within a DatasetDict for text generation, tokenizing the inputs and ensuring 
    they are no longer than the max_length specified. Automatically detects columns containing text data.

    Args:
        dataset_dict (DatasetDict): A DatasetDict object containing the datasets.
        tokenizer (PreTrainedTokenizer): A tokenizer object from the Huggingface's transformers library.
        max_length (int): The maximum length of the tokenized input sequences.

    Returns:
        DatasetDict: The preprocessed DatasetDict object with tokenized input sequences.
    """

    def extract_text_data(input_data: Any) -> List[str]:
        """
        Extracts text data from input data which could be a string, a list of strings, or a dictionary.

        Args:
            input_data (Any): The input data.

        Returns:
            List[str]: A list of strings extracted from the input data.
        """
        if isinstance(input_data, str):
            return [input_data]
        elif isinstance(input_data, list):
            # Assuming all elements in the list are strings.
            return input_data
        elif isinstance(input_data, dict):
            # Extracting string values from the dictionary.
            return list(input_data.values())
        else:
            raise ValueError("Unsupported data type for text data extraction.")

    def tokenize_function(examples: Dict[str, Any]) -> Dict[str, List[int]]:
         """
         Tokenizes the text data within the examples.
     
         Args:
             examples (Dict[str, Any]): A dictionary containing the data examples to be tokenized.
     
         Returns:
             Dict[str, List[int]]: A dictionary containing the tokenized examples.
        Examples:
        # # Example usage:
          # from transformers import AutoTokenizer
          # tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
          # max_length = 100  # Maximum sequence length
          
          # # # Assume dataset_dict is the DatasetDict object that you have loaded.
          # # preprocessed_data =preprocess_datasets(dataset_dict['train'], tokenizer, max_length)
          # # from transformers import (
          # #     AutoModelForCausalLM,
          # #     AutoTokenizer,
          # #     HfArgumentParser,
          # #     TrainingArguments,
          # #     Trainer,
          # #     default_data_collator,
          # #     get_linear_schedule_with_warmup,
          # #     BitsAndBytesConfig
          # # )
          # # from torch.utils.data import Dataset, DataLoader
          # # train_dataloader = DataLoader(
          # #     preprocessed_data['train'], shuffle=True, collate_fn=default_data_collator, batch_size=12, pin_memory=True
          # #     )
         """
         # Tokenize each example individually and return the tokenized inputs
         tokenized_inputs = {key: [] for key in tokenizer.model_input_names}
         for column in examples:
             if isinstance(examples[column], (str, list, dict)):
                 text_data = extract_text_data(examples[column])
                 for text in text_data:
                     # Tokenize each piece of text data
                     inputs = tokenizer(text, max_length=max_length, truncation=True, padding='max_length', return_tensors='pt')
                     # Append tokens to corresponding keys
                     for key in tokenized_inputs.keys():
                         tokenized_inputs[key].append(inputs[key].squeeze().tolist())
         return tokenized_inputs

    # Apply tokenization to each split in the dataset dict
    preprocessed_dataset_dict = dataset_dict.map(
        tokenize_function,
        batched=True,
        remove_columns=[column for column in dataset_dict['train'].column_names],
        desc=f"""
        Tokenizing the text data within the dataset dict--->[by ANNA].
        """
    )

    return preprocessed_dataset_dict


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
def pre_processing_input(preprocessed_data):
    train_dataloader = DataLoader(
    preprocessed_data['train'], shuffle=True, collate_fn=default_data_collator, batch_size=12, pin_memory=True
    )
    test_dataloader = DataLoader(
    preprocessed_data['test'], shuffle=True, collate_fn=default_data_collator, batch_size=12, pin_memory=True
    )
    eval_dataloader = DataLoader(
    preprocessed_data['eval'], shuffle=True, collate_fn=default_data_collator, batch_size=12, pin_memory=True
    )
    return train_dataloader, test_dataloader, eval_dataloader

tokenizer=create_tokenizer(tokenizer_name_or_path=Main_args.tokenizer_name_or_path)
max_length=Main_args.max_length
if Main_args.dataset_name_or_path is not None:
   dataset=load_and_prepare_dataset(Main_args.dataset_name_or_path)
   preprocessed_data =preprocess_datasets(dataset, tokenizer, max_length)
elif Main_args.dataset_config_name is not None:
    dataset=load_and_prepare_dataset(Main_args.dataset_config_name)
    preprocessed_data =preprocess_datasets(dataset['train'], tokenizer, max_length)
else:
    ValueError("Either dataset_name_or_path or dataset_config_name must be specified.")


train_dataloader, test_dataloader, eval_dataloader=pre_processing_input(preprocessed_data=preprocessed_data)

model=load_model(model_name_or_path=Main_args.model_name_or_path)

args=model_parse_arguments()
config_args = load_arguments_from_yaml('/content/creating.yml')
for key, value in config_args.items():
    setattr(args, key, value)

training_args=TrainingArguments(
output_dir=args.output_dir,
overwrite_output_dir=args.overwrite_output_dir,
do_train=args.do_train,
do_eval=args.do_eval,
do_predict=args.do_predict,
evaluation_strategy=args.evaluation_strategy,
prediction_loss_only=args.prediction_loss_only,
per_device_train_batch_size=args.per_device_train_batch_size,
per_device_eval_batch_size=args.per_device_eval_batch_size,
per_gpu_train_batch_size=args.per_gpu_train_batch_size,
per_gpu_eval_batch_size=args.per_gpu_eval_batch_size,
gradient_accumulation_steps=args.gradient_accumulation_steps,
eval_accumulation_steps=args.eval_accumulation_steps,
eval_delay=args.eval_delay,
learning_rate=args.learning_rate,
weight_decay=args.weight_decay,
adam_beta1=args.adam_beta1,
adam_beta2=args.adam_beta2,
adam_epsilon=args.adam_epsilon,
max_grad_norm=args.max_grad_norm,
num_train_epochs=args.num_train_epochs,
max_steps=args.max_steps,
lr_scheduler_type=args.lr_scheduler_type,
lr_scheduler_kwargs=args.lr_scheduler_kwargs,
warmup_ratio=args.warmup_ratio,
warmup_steps=args.warmup_steps,
log_level=args.log_level,
log_level_replica=args.log_level_replica,
log_on_each_node=args.log_on_each_node,
logging_dir=args.logging_dir,
logging_strategy=args.logging_strategy,
logging_first_step=args.logging_first_step,
logging_steps=args.logging_steps,

logging_nan_inf_filter=args.logging_nan_inf_filter,

save_strategy=args.save_strategy,
save_steps=args.save_steps,
save_total_limit=args.save_total_limit,
save_safetensors=args.save_safetensors,
save_on_each_node=args.save_on_each_node,
save_only_model=args.save_only_model,
no_cuda=args.no_cuda,
use_cpu=args.use_cpu,
use_mps_device=args.use_mps_device,
seed=args.seed,
data_seed=args.data_seed,
jit_mode_eval=args.jit_mode_eval,
use_ipex=args.use_ipex,
bf16=args.bf16,
fp16=args.fp16,
fp16_opt_level=args.fp16_opt_level,
half_precision_backend=args.half_precision_backend,
bf16_full_eval=args.bf16_full_eval,
fp16_full_eval=args.fp16_full_eval,
tf32=args.tf32,
local_rank=args.local_rank,
ddp_backend=args.ddp_backend,
tpu_num_cores=args.tpu_num_cores,
tpu_metrics_debug=args.tpu_metrics_debug,
debug=args.debug,
dataloader_drop_last=args.dataloader_drop_last,
eval_steps=args.eval_steps,
dataloader_num_workers=args.dataloader_num_workers,
past_index=args.past_index,
run_name=args.run_name,
disable_tqdm=args.disable_tqdm,
remove_unused_columns=args.remove_unused_columns,
label_names=args.label_names,
load_best_model_at_end=args.load_best_model_at_end,
metric_for_best_model=args.metric_for_best_model,
greater_is_better=args.greater_is_better,
ignore_data_skip=args.ignore_data_skip,
fsdp=args.fsdp,
fsdp_min_num_params=args.fsdp_min_num_params,
fsdp_config=args.fsdp_config,
fsdp_transformer_layer_cls_to_wrap=args.fsdp_transformer_layer_cls_to_wrap,
deepspeed=args.deepspeed,
label_smoothing_factor=args.label_smoothing_factor,
optim=args.optim,
optim_args=args.optim_args,
adafactor=args.adafactor,
group_by_length=args.group_by_length,
length_column_name=args.length_column_name,
report_to=args.report_to,
ddp_find_unused_parameters=args.ddp_find_unused_parameters,
ddp_bucket_cap_mb=args.ddp_bucket_cap_mb,
ddp_broadcast_buffers= args.ddp_broadcast_buffers,
dataloader_pin_memory=args.dataloader_pin_memory,
dataloader_persistent_workers=args.dataloader_persistent_workers,
skip_memory_metrics=args.skip_memory_metrics,
use_legacy_prediction_loop=args.use_legacy_prediction_loop,
push_to_hub=args.push_to_hub,
resume_from_checkpoint=args.resume_from_checkpoint,
hub_model_id=args.hub_model_id,
hub_strategy=args.hub_strategy,
hub_token=args.hub_token,
hub_private_repo=args.hub_private_repo,
hub_always_push=args.hub_always_push,
gradient_checkpointing=args.gradient_checkpointing,
gradient_checkpointing_kwargs=args.gradient_checkpointing_kwargs,
include_inputs_for_metrics=args.include_inputs_for_metrics,
fp16_backend=args.fp16_backend,
push_to_hub_model_id=args.push_to_hub_model_id,
push_to_hub_organization=args.push_to_hub_organization,
push_to_hub_token=args.push_to_hub_token,
mp_parameters=args.mp_parameters,
auto_find_batch_size=args.auto_find_batch_size,
full_determinism=args.full_determinism,
torchdynamo=args.torchdynamo,
ray_scope=args.ray_scope,
ddp_timeout=args.ddp_timeout,
torch_compile=args.torch_compile,
torch_compile_backend=args.torch_compile_backend,
torch_compile_mode=args.torch_compile_mode,
dispatch_batches=args.dispatch_batches,
split_batches=args.split_batches,
include_tokens_per_second=args.include_tokens_per_second,
include_num_input_tokens_seen=args.include_num_input_tokens_seen,
neftune_noise_alpha=args.neftune_noise_alpha,

)
print(training_args)
