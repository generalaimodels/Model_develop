# 🔒 Confidentiality Notice 
#
# DARFA-LLMs, is developed under the auspices of CDAC (Center for Development of Advanced Computing). 
# The project's codebase, algorithms, and any associated documentation are considered confidential 
# and proprietary to CDACB. 
#
# Copyright Statement ©️
#
# All code within this project is the intellectual property of Kandimalla Hemanth. 👨‍💻 Any reproduction, 
# distribution, or unauthorized use of the code or its components without explicit permission from the
# copyright holder is strictly prohibited. 
#
# Project Purpose 
#
# The DARFA-llms project is dedicated to advancing human development through the application of artificial 
# intelligence (AI) technologies. Its mission is to address real-world challenges and contribute 
# to the betterment of society. 
#
#  Non-Disclosure Agreement (NDA) 
#
# Access to this Python project and its associated materials is restricted to authorized personnel only.
# 🔒 By accessing or using any part of this project, you agree to maintain strict confidentiality 
# and adhere to any applicable non-disclosure agreements (NDAs) or confidentiality policies established
# by CDACB. 
#
# ⚠️ Disclaimer of Liability 📜
#
# While every effort has been made to ensure the accuracy and reliability of the code and information 
# contained within this project, CDACB and Kandimalla Hemanth shall not be held liable for any direct, 
# indirect, incidental, special, or consequential damages arising out of the use or inability to use the
# project, even if advised of the possibility of such damages. 
#
# 🚫 Usage Restrictions 📝
#
# The code and algorithms developed as part of the DARFA project are intended for research, educational,
# and non-commercial purposes only. Any commercial use, modification, or adaptation of the codebase 
# requires explicit written consent from CDACB and Kandimalla Hemanth.
#
#  Ethical Considerations 
#
# The DARFA project adheres to ethical principles and guidelines governing the responsible 
# use of AI technologies. All efforts are made to ensure that the project's objectives 
# align with ethical standards and promote positive outcomes for humanity. 



from typing import List, Optional, Dict, Union
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    default_data_collator,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)

import data_loader_llm
import model_loader_llm
import data_load_cleaning
import pre_processing_data

def preprocess_function(examples: Dict[str, List[str]], text_columns: Union[str, List[str]], label_column: str, tokenizer: AutoTokenizer, max_length: int) -> Dict[str, List[torch.Tensor]]:
    if isinstance(text_columns, str):
        text_columns = [text_columns]
    
    input_ids = []
    attention_masks = []
    labels = []
    
    for example in zip(*[examples[col] for col in text_columns + [label_column]]):
        input_text = " ".join([f"{col}: {val}" for col, val in zip(text_columns, example[:-1]) if val is not None])
        label = example[-1]
        
        # Tokenize input text
        input_tokens = tokenizer.tokenize(input_text)
        
        # Handle input text longer than max_length
        if len(input_tokens) > max_length - 2:  # Subtract 2 for special tokens
            input_tokens = input_tokens[:max_length - 2]
        
        # Add special tokens and create input_ids
        input_tokens = [tokenizer.bos_token] + input_tokens + [tokenizer.eos_token]
        input_ids_example = tokenizer.convert_tokens_to_ids(input_tokens)
        
        # Create attention_mask
        attention_mask = [1] * len(input_ids_example)
        
        # Pad or truncate to max_length
        padding_length = max_length - len(input_ids_example)
        if padding_length > 0:
            input_ids_example = input_ids_example + [tokenizer.pad_token_id] * padding_length
            attention_mask = attention_mask + [0] * padding_length
        else:
            input_ids_example = input_ids_example[:max_length]
            attention_mask = attention_mask[:max_length]
        
        input_ids.append(input_ids_example)
        attention_masks.append(attention_mask)
        labels.append(input_ids_example)  # Set labels to be identical to input_ids
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "labels": labels,
    }

def process_dataset(dataset, text_columns: Union[str, List[str]], label_column: str, tokenizer: AutoTokenizer, max_length: int, batch_size: int):
    processed_datasets = dataset.map(
        lambda examples: preprocess_function(examples, text_columns, label_column, tokenizer, max_length),
        batched=True,
        num_proc=1,
        remove_columns=dataset["train"].column_names,
        load_from_cache_file=False,
        desc=" || DARFA Project || by  Hemanth Kandimalla",
    )
    
    train_dataset = processed_datasets["train"]
    test_dataset = processed_datasets["test"]
    eval_dataset=processed_datasets["eval"]
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True
    )
    test_dataloader=DataLoader(
        test_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True
    )
    eval_dataloader = DataLoader(
        eval_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True
    )
    
    return train_dataloader,test_dataloader,eval_dataloader


