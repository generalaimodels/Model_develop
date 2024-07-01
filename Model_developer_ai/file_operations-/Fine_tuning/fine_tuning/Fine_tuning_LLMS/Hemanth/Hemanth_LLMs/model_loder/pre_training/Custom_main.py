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

import argparse
import yaml

from typing import List, Optional
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
import fine_tuning_pre_processing
import functionality_LLMS


def load_arguments_from_yaml(yaml_file_path: str) -> dict:
    with open(yaml_file_path, 'r') as stream:
        try:
            arguments = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            arguments = {}
    return arguments


def main():
    parser = argparse.ArgumentParser(description="DARFA-LLM's Project")
    parser.add_argument('--config', type=str, default='./config.yml',
                        help='Path to the configuration YAML file')
    args = parser.parse_args()

    # Load arguments from YAML file
    config_args = load_arguments_from_yaml(args.config)

    # Update args with values from the YAML file
    for key, value in config_args.items():
        setattr(args, key, value)

    # ===============|datasets|=====================
    # dataset=loading_folder_using_datasets(folder_path=f'{DIR_OUTPUT}/reformatted')
    dataset = data_loader_llm.load_and_prepare_dataset("fka/awesome-chatgpt-prompts")
    model_parameters_print=True
    # =================Model and tokenizer loading===
    model = model_loader_llm.load_model_TEST(model_name_or_path='gpt2')
    tokenizer = model_loader_llm.create_tokenizer('gpt2')
    if model_parameters_print:
        model_loader_llm.calculate_model_parameters(model)

    MAX_LENGTH: int = 100
    TEXT_COLUMNS: List[str] = dataset['train'].column_names
    LABEL_COLUMN: str = 'act'
    BATCH_SIZE: int = 10
    learning_rate = 0.001
    weight_decay = 0.1
    num_epochs = 2
    checkpoint_dir: str = './output'
    plot_dir: str = './plots'

    train_dataloader, test_dataloader, eval_dataloader = fine_tuning_pre_processing.process_dataset(
        dataset, TEXT_COLUMNS, LABEL_COLUMN, tokenizer, MAX_LENGTH, BATCH_SIZE
    )

    trained_model, metrics = functionality_LLMS.train_model_test(
        model, train_dataloader, learning_rate, weight_decay, num_epochs,
        checkpoint_dir=checkpoint_dir, plot_dir=plot_dir
    )


model = model_loader_llm.load_model_TEST(model_name_or_path='gpt2')
tokenizer = model_loader_llm.create_tokenizer('gpt2')
tokenizer.pad_token_id=tokenizer.eos_token_id
input_text = "tell a concepts of love? "
num_tokens = 100
text="tell a concepts of love? "
model_loader_llm.visualize_text_generation_test(
    model=model,
    tokenizer=tokenizer,
    input_text=input_text,
    num_tokens=num_tokens,
    device='cpu'
    
    
    
)

# if __name__ == '__main__':
#     main()