import os
import warnings
import torch 
from peft import (
    LoraConfig,
    PeftConfig,
    PeftModel,
    TaskType,
    get_peft_model,
    prepare_model_for_int8_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    default_data_collator,
    get_linear_schedule_with_warmup,
    BitsAndBytesConfig
)
from tqdm import tqdm
from accelerate import Accelerator
from dataclasses import dataclass, field
from torch.utils.data import Dataset, DataLoader
from enum import Enum
from datasets import load_dataset,DatasetDict
from typing import Union , Dict,Optional,Any,List


warnings.filterwarnings('ignore')

DEVICE = "cuda"
MODEL_NAME_OR_PATH = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
TOKENIZER_NAME_OR_PATH = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MAX_LENGTH = 512
LR = 3e-2
NUM_EPOCHS = 3
BATCH_SIZE = 16

#========= * DATASET ARGUMENTS * =====================


DATASET_LOCAL_PATH = " "
DATASET_NAME = " "
TEXT_COLUMN = " "
LABEL_COLUMN = " "

#========= * PEFT MODEL ARGUMENTS * =====================
inference_mode=False,
r=64,
lora_alpha=256,
lora_dropout=0.2,
def advanced_data_loader(input: Union[str, Dict[str, str]], format: Optional[str] = None, split_ratios: Optional[Dict[str, float]] = None) -> Optional[DatasetDict]:
    """
    Loads a dataset from a given input path or dictionary specifying file paths and splits it.

    :param input: A string representing the dataset name or directory, or a dictionary containing file paths.
    :param format: The format of the dataset if loading from a file (e.g., 'csv' or 'json').
    :param split_ratios: A dictionary with keys 'train', 'test', and 'eval' containing split ratios.
    :return: A loaded and split dataset or None in case of failure.
    """
    if split_ratios is None:
        split_ratios = {'train': 0.8, 'test': 0.1, 'eval': 0.1}

    try:
        # Load the dataset
        if isinstance(input, dict) and format in ['csv', 'json']:
            dataset = load_dataset(format, data_files=input)
        elif isinstance(input, str) and format == 'text':
            dataset = load_dataset(format, data_dir=input)
        elif isinstance(input, str) and format is None:
            dataset = load_dataset(input)
        else:
            warnings.warn("Invalid input or format. Please provide a valid dataset name, directory, or file paths.")
            return None
    except FileNotFoundError as e:
        warnings.warn(str(e))
        return None

    # Split the dataset
    if dataset:
        split_dataset = dataset['train'].train_test_split(test_size=split_ratios['test'] + split_ratios['eval'])
        test_eval_dataset = split_dataset['test'].train_test_split(test_size=split_ratios['eval'] / (split_ratios['test'] + split_ratios['eval']))
        dataset = DatasetDict({
            'train': split_dataset['train'],
            'test': test_eval_dataset['train'],
            'eval': test_eval_dataset['test']
        })

    print("Splits: ", dataset.keys())
    print("Columns: ", {split: dataset[split].column_names for split in dataset.keys()})
    return dataset


def create_tokenizer(tokenizer_name_or_path: str = 'gpt2') -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.bos_token_id is None:
        tokenizer.bos_token_id = tokenizer.pad_token_id
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token_id = tokenizer.pad_token_id
    if tokenizer.unk_token_id is None:
        tokenizer.unk_token_id = tokenizer.pad_token_id
    if tokenizer.sep_token_id is None:
        tokenizer.sep_token_id = tokenizer.pad_token_id
    if tokenizer.cls_token_id is None:
        tokenizer.cls_token_id = tokenizer.pad_token_id
    if tokenizer.mask_token_id is None:
        tokenizer.mask_token_id = tokenizer.pad_token_id
    return tokenizer

def preprocess_function(examples):
  """
  Preprocess the dataset.
  """
  batch_size = len(examples[TEXT_COLUMN])
  inputs = [f"{TEXT_COLUMN } : {x} Label : " for x in examples[TEXT_COLUMN ]]
  targets = [str(x) for x in examples[LABEL_COLUMN]]
  model_inputs = tokenizer(inputs)
  labels = tokenizer(targets, add_special_tokens=False)  # don't add bos token because we concatenate with inputs
  for i in range(batch_size):
      sample_input_ids = model_inputs["input_ids"][i]
      label_input_ids = labels["input_ids"][i] + [tokenizer.eos_token_id]
      model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
      labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids
      model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])
  for i in range(batch_size):
      sample_input_ids = model_inputs["input_ids"][i]
      label_input_ids = labels["input_ids"][i]
      model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (
          MAX_LENGTH - len(sample_input_ids)
      ) + sample_input_ids
      model_inputs["attention_mask"][i] = [0] * (MAX_LENGTH - len(sample_input_ids)) + model_inputs[
          "attention_mask"
      ][i]
      labels["input_ids"][i] = [-100] * (MAX_LENGTH - len(sample_input_ids)) + label_input_ids
      model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:MAX_LENGTH])
      model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:MAX_LENGTH])
      labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][:MAX_LENGTH])
  model_inputs["labels"] = labels["input_ids"]
  return model_inputs

def test_preprocess_function(examples):
  batch_size = len(examples[TEXT_COLUMN])
  inputs = [f"{TEXT_COLUMN} : {x} Label : " for x in examples[TEXT_COLUMN]]
  model_inputs = tokenizer(inputs)
  # print(model_inputs)
  for i in range(batch_size):
      sample_input_ids = model_inputs["input_ids"][i]
      model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (
          MAX_LENGTH - len(sample_input_ids)
      ) + sample_input_ids
      model_inputs["attention_mask"][i] = [0] * (MAX_LENGTH - len(sample_input_ids)) + model_inputs[
          "attention_mask"
      ][i]
      model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:MAX_LENGTH])
      model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:MAX_LENGTH])
  return model_inputs

input_dir = {
    'train':'',
    'test':'',
    'eval':''
    
}
dataset=advanced_data_loader(input=input_dir, format="csv")
peft_config =LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=64,
    lora_alpha=256,
    lora_dropout=0.2,
    # fan_in_fan_out=True,
    # bias="all",
    # modules_to_save=["classifier/score", "pooler"],
    # init_lora_weights="gaussian",
    # target_modules=["q_proj", "k_proj"],
    # modules_to_save=["lm_head"],

    # layers_to_transform=[2, 4, 6],
    # layers_pattern="custom_pattern",
    # rank_pattern={
    #     "model.decoder.layers.0.encoder_attn.k_proj": 16,
    #     "model.decoder.layers.2.encoder_attn.k_proj": 32
    # },
    # alpha_pattern={
    #     "model.decoder.layers.0.encoder_attn.k_proj": 64,
    #     "model.decoder.layers.4.encoder_attn.k_proj": 128
    # },
    # megatron_config={
    #     "hidden_size": 4096,
    #     "num_attention_heads": 32,
    #     "num_layers": 24
    # },
    # megatron_core="custom_megatron_core",
    # loftq_config=LoraConfig(
    #     quantization_bits=8,
    #     quantization_range=128
    # )
)



#=========*   creating tokenizer * =====================
tokenizer = create_tokenizer(tokenizer_name_or_path=TOKENIZER_NAME_OR_PATH)
#=========*    creating model * =====================
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME_OR_PATH)

model = get_peft_model(model, peft_config)
accelerator = Accelerator()
model.to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=NUM_EPOCHS*10)
with accelerator.main_process_first():
  processed_datasets = dataset.map(
            preprocess_function,
            batched=True,
            num_proc=1,
            remove_columns=dataset["train"].column_names,
            load_from_cache_file=True,
            desc="Running tokenizer on dataset",
)
train_dataset = processed_datasets["train"]
with accelerator.main_process_first():
  processed_datasets = dataset.map(
            test_preprocess_function,
            batched=True,
            num_proc=1,
            remove_columns=dataset["test"].column_names,
            load_from_cache_file=False,
            desc="Running tokenizer on dataset",
  )
test_dataset = processed_datasets["test"]
with accelerator.main_process_first():
  processed_datasets = dataset.map(
            test_preprocess_function,
            batched=True,
            num_proc=1,
            remove_columns=dataset["test"].column_names,
            load_from_cache_file=False,
            desc="Running tokenizer on dataset",
  )
eval_dataset = processed_datasets["eval"]
train_dataloader = DataLoader(
    train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=BATCH_SIZE, pin_memory=True
)
eval_dataloader = DataLoader(
    eval_dataset, shuffle=True,collate_fn=default_data_collator, batch_size=BATCH_SIZE, pin_memory=True
)
test_dataloader = DataLoader(
    test_dataset, collate_fn=default_data_collator, batch_size=BATCH_SIZE, pin_memory=True
)

# ================== * TRAINING AND EVALUATION * ============================

accelerator.wait_for_everyone()
model, train_dataloader, eval_dataloader, test_dataloader, optimizer, scheduler = accelerator.prepare(
model, train_dataloader, eval_dataloader, test_dataloader, optimizer, scheduler
    )
model.to(DEVICE)
train_losses = []
eval_losses = []
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    for step, batch in enumerate(tqdm(train_dataloader)):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.detach().float()
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    train_epoch_loss = total_loss / len(train_dataloader)
    train_losses.append(train_epoch_loss)  # Store train loss for this epoch
model.push_to_hub(
        f"{DATASET_NAME}_{MODEL_NAME_OR_PATH}_{peft_config.task_type}".replace("/", "_"),
        token = "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"
    )

peft_model_id = f"{DATASET_NAME}_{MODEL_NAME_OR_PATH}_{peft_config.task_type}".replace("/", "_")
model.save_pretrained(peft_model_id)



# config = PeftConfig.from_pretrained(peft_model_id)
# model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
# model = PeftModel.from_pretrained(model, peft_model_id)
model.to(DEVICE)
model.eval()
i = 4
inputs = tokenizer(f'{TEXT_COLUMN} : {dataset["test"][i][TEXT_COLUMN]} Label : ', return_tensors="pt")
with torch.no_grad():
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    outputs = model.generate(
        input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], max_new_tokens=512, eos_token_id=3
    )
    tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)



