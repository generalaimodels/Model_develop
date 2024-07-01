````python
from FAST_ANALYSIS import (
rgb_print,
AdvancedPipelineForhemanth,
AiModelForHemanth,
AdvancedPreProcessForHemanth,
prepare_datasetsforhemanth,
get_dataset_info_for_hemanth,
summarizemodelforhemanth,
printmodelsummaryforhemanth,
data_tokenization,
create_data_loaders,
generate_text,
)

from pre_training_pipeline import (
    create_logo,
    train,
    evaluate,
    load_model_for_inference,
    load_sharded_model,
    plot_training_metrics,

)
import logging
from typing import Dict
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
dataset=prepare_datasetsforhemanth(
    path=r"C:\Users\heman\Desktop\Coding\data\fka___awesome-chatgpt-prompts"
)
# text=rgb_print(dataset, (255,0,0))
# print(text)
# dataset_info=get_dataset_info_for_hemanth(dataset=dataset)
# for split_name, split_data in dataset_info.items():
#     for key, value in split_data.items():
#         print(rgb_print(f"  {key} : {value}", (0, 255, 0)))
seq_max_len=100
batch_size=10
tokenizer=AdvancedPreProcessForHemanth(
    model_type="text",
    pretrained_model_name_or_path="gpt2",
    cache_dir=r"C:\Users\heman\Desktop\Coding\data"
)
tokenizer=tokenizer.process_data()
tokenizer.pad_token = tokenizer.eos_token
model=AiModelForHemanth.load_model(
    model_type="causal_lm",
    model_name_or_path="gpt2",
    cache_dir=r"C:\Users\heman\Desktop\Coding\data"
)
dataset = data_tokenization(
            dataset=dataset,
            tokenizer=tokenizer,
            seq_max_length=seq_max_len,
            pad_to_max_length=True,
            return_tensors="pt",
)
train_loader, test_loader, eval_loader = create_data_loaders(
            dataset=dataset, batch_size=batch_size
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    epochs = 2
    learning_rate = 2e-5
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    history = train(model, train_loader, optimizer, scheduler, epochs, device)
    plot_training_metrics(history)
    eval_results = evaluate(model, eval_loader, device)
    print("Evaluation Results:", eval_results)
    test_results = evaluate(model, test_loader, device) 
    print("Testing Results:", test_results) 
    model=load_model_for_inference(model=model,model_dir=r"C:\Users\heman\Desktop\Coding\model_checkpoints\best_model_epoch_5",device=device)
    text=generate_text(model=model,tokenizer=tokenizer,prompt="what is the meanings of life??",max_length=250)
    text=rgb_print(text=text,rgb=(124, 200, 124))
    print(text)



```













```python

def Pretraining_pipeline():
    """
    Main function to execute the data loading and preprocessing pipeline.
    """
    try:
        dataset_path = r"E:\LLMS\Fine-tuning\data\fka___awesome-chatgpt-prompts"
        cache_dir = r"E:\LLMS\Fine-tuning\data"
        seq_max_len = 50
        batch_size = 32

        dataset =prepare_datasetsforhemanth(path=dataset_path)
        print(dataset)

        tokenizer = AutoTokenizer.from_pretrained(
            "gpt2", cache_dir=cache_dir
        )
        model=AiModelForHemanth.load_model(
            model_type="causal_lm",
            model_name_or_path="gpt2",
            cache_dir=cache_dir,
        )
        tokenizer.pad_token = tokenizer.eos_token
        dataset = data_tokenization(
            dataset=dataset,
            tokenizer=tokenizer,
            seq_max_length=seq_max_len,
            pad_to_max_length=True,
            return_tensors="pt",
        )

        train_loader, test_loader, eval_loader = create_data_loaders(
            dataset=dataset, batch_size=batch_size
        )
        for batch in train_loader:  
           output=model(**batch)
           print(output.logits.shape)
           input_ids = batch['input_ids'] 
        #    attention_mask = batch['labels']
           print(tokenizer.decode(input_ids[0]))
        #    print(tokenizer.decode(attention_mask[0]))

        # Example usage
        # print(tokenizer.encode(next(iter(train_loader))['input_ids'][0]))
    except Exception as e:
        print(f"An error occurred: {e}")




```