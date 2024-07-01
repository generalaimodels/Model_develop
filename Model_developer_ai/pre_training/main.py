import argparse
import yaml
import time
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot
from typing import List, Optional
from torch.utils.data import DataLoader

from transformers import TrainingArguments, default_data_collator, Trainer

from dataset_collection import (
    get_files_with_extensions,
    reformat_txt_files,
    write_to_csv,
    process_pdfs_from_csv,
    process_files_txtfile,
    loading_folder_using_datasets
)
import model_loader_llm
# import dataset_loader
import pre_processing_data





def load_arguments_from_yaml(yaml_file_path:str):
    with open(yaml_file_path, 'r') as stream:
        try:
            arguments = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            arguments = {}
    return arguments
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tuning LLMs")
    
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
    parser.add_argument("--seed", type=int, default=10)
    
    # Model and training arguments
    parser.add_argument("--model_output_dir", type=str, default="./Hemanth_LLMs")
    parser.add_argument("--num_train_epochs", type=int, default=250)
    parser.add_argument("--save_total_limit", type=int, default=5)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--dataloader_drop_last", action="store_true", default=True)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--push_to_hub", action="store_true", default=True)
    parser.add_argument("--remove_unused_columns", action="store_true", default=True)
    parser.add_argument("--max_steps", type=int, default=10000)
    
    return parser.parse_args()


args=parse_arguments()
config_args = load_arguments_from_yaml(yaml_file_path='E:/LLMS/Fine-tuning/llms-data/Fine_TuningS/fine_tuning/Fine_tuning_LLMS/Hemanth/Hemanth_LLMs/model_loder/pre_training/argument.yml')
for key, value in config_args.items():
    setattr(args, key, value)

def main(args=args) -> None:
    start_time = time.time()
    # Data loading and preprocessing
    ext_files = get_files_with_extensions(args.dir_path)
    reformat_txt_files([args.dir_output])
    write_to_csv(args.csv_file_path, ext_files)
    process_pdfs_from_csv(csv_path=args.csv_file_path, output_folder=args.dir_output)
    process_files_txtfile(args.dir_path, args.dir_output)
    dataset = loading_folder_using_datasets(folder_path=f'{args.dir_output}/reformatted')
    dataset=dataset['train'].train_test_split(0.15)
    print(dataset)
    
    # Model and tokenizer loading
    model = model_loader_llm.load_model_TEST(model_name_or_path=args.model_name_path)
    tokenizer = model_loader_llm.create_tokenizer(args.model_name_path)
    
    if args.model_parameters_print:
        model_loader_llm.calculate_model_parameters(model)
    
    train_dataset = dataset['train']
    
    train_datasets1 = pre_processing_data.ConstantLengthDataset(
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
    train_dataloader = DataLoader(
        train_dataset, shuffle=False, collate_fn=default_data_collator, batch_size=1
    )
    eval_dataloader = DataLoader(train_dataset,shuffle=False, collate_fn=default_data_collator, batch_size=1)
    
    print(train_dataloader,eval_dataloader )
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.model_output_dir,
        num_train_epochs=args.num_train_epochs,
        # save_total_limit=args.save_total_limit,
        # per_device_train_batch_size=args.per_device_train_batch_size,
        # warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        # dataloader_drop_last=args.dataloader_drop_last,
        # bf16=args.bf16,
        # logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        # gradient_checkpointing=args.gradient_checkpointing,
        # push_to_hub=args.push_to_hub,
        # gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=args.remove_unused_columns,
        max_steps=args.max_steps
    )
    
    # Trainer setup
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset= train_datasets1,
        eval_dataset= train_datasets1,
        data_collator=default_data_collator,
    )
    
#     # Start training
#     trainer.train()
#     # model.push_to_hub(
#     #     f"Hemanth_LLMS_OUTPUT".replace("/", "_"),
#     #     token = "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"
#     # )
    
#   # Plotting losses and metrics
#     losses = [epoch_log["loss"] for epoch_log in trainer.state.log_history]
#     metrics = [epoch_log["eval_loss"] for epoch_log in trainer.state.log_history]

#     fig = go.Figure(data=[go.Scatter(x=range(len(losses)), y=losses, name="Loss")])
#     fig.update_layout(title="Loss Curve", xaxis_title="Epochs", yaxis_title="Loss")
#     plot(fig, filename=f"{args.model_output_dir}/loss_curve.html")

#     fig = go.Figure(data=[go.Scatter(x=range(len(metrics)), y=metrics, name="Metrics")])
#     fig.update_layout(title="Metrics Curve", xaxis_title="Epochs", yaxis_title="Metrics")
#     plot(fig, filename=f"{args.model_output_dir}/metrics_curve.html")

#     print(f"Training completed in {time.time() - start_time:.2f} seconds")



if __name__ == "__main__":
    main()
    