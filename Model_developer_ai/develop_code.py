from FAST_ANALYSIS import (
convertfolders_to_txtfolder,
prepare_datasetsforhemanth,
rgb_print,
AdvancedDatasetProcessor,
AdvancedPreProcessForHemanth,
AiModelForHemanth,
printmodelsummaryforhemanth,
summarizemodelforhemanth,
TrainerForHemanth,
)

# input_folders=[r"E:\LLMS\Fine-tuning\Deep_learning"]
output_folder=r"E:\LLMS\Fine-tuning\data\data"
# max_workers=4
# convertfolders_to_txtfolder(
#     input_folders=input_folders,
#     output_folder=output_folder,
#     max_workers=max_workers,
# )

tokenizer=AdvancedPreProcessForHemanth(
    model_type="text",
    pretrained_model_name_or_path="gpt2",
    revision="main",
    cache_dir=r"E:\LLMS\Fine-tuning\data",
  
)
tokenizer=tokenizer.process_data()
tokenizer.pad_token = tokenizer.eos_token

pre_processing=AdvancedDatasetProcessor(
    tokenizer=tokenizer,
    dataset_path=r"E:\LLMS\Fine-tuning\data\data"
)
train_loader, test_loader,eval_tester=pre_processing.load_and_process_dataset()

text=tokenizer.decode(next(iter(train_loader))['input_ids'][0])
text=rgb_print(text=text , rgb=(0,225,0))
print(text)
model=AiModelForHemanth.load_model(
    model_type="causal_lm",
    model_name_or_path="gpt2",
    cache_dir=r"E:\LLMS\Fine-tuning\data",
)
result=summarizemodelforhemanth(model=model)
printmodelsummaryforhemanth(model_summary=result)