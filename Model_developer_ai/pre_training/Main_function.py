import argparse
import yaml
from typing import List, Optional

from transformers import TrainingArguments,Trainer
# from transformers.data.data_collator import default_data_collator
# import dataset_collection
import model_loader_llm
import dataset_loader
import pre_processing_data
import Testing_preprocessing
import functionality_LLMS

dataset = dataset_loader.load_and_prepare_dataset("fka/awesome-chatgpt-prompts")
tokenizer = model_loader_llm.create_tokenizer('gpt2')
model = model_loader_llm.load_model_TEST(model_name_or_path='gpt2')
train_datasets1 =Testing_preprocessing.ConstantLengthDataset1(
        tokenizer,
        dataset['train'],
        infinite=True,
        seq_length=512,
        chars_per_token=2.5, 
        fim_rate=0.5,
        fim_spm_rate=0.25,
        seed=0,
    )

x,y=functionality_LLMS.train_model(
    model, train_datasets1, learning_rate=0.01,weight_decay=0.01,num_epochs=1,checkpoint_dir="test_hemanth",plot_dir='plot_hemanth'
)

