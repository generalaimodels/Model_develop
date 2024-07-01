from FAST_ANALYSIS.generalised_data_loader import (
    prepare_datasetsforhemanth,
    load_and_prepare_dataset_for_hemanth,
    get_dataset_info_for_hemanth,
    rgb_print,
)
from FAST_ANALYSIS import (
    printmodelsummaryforhemanth,
    summarizemodelforhemanth,
    generate_text,
    generate_text_with_strategies,
    generate_text_generalised,
    sample_train_Listofstr,
)
from FAST_ANALYSIS.generalised_pipline_any_task import AdvancedDatasetProcessor
from pre_processing.text import AdvancedPipeline,AdvancedDatasetProcessor


from FAST_ANALYSIS import (
    AiModelForHemanth, 
    AdvancedPreProcessForHemanth,
    TrainerForHemanth,
    create_advanced_model_trainer,# customsed from public api
)

from pre_training_pipeline import (train,
                                   plot_training_metrics,
                                   evaluate,
                                   load_model_for_inference
)
from torch.optim import AdamW
from  transformers import get_linear_schedule_with_warmup
import torch
#=============================datacollecttion====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
text=rgb_print(text=f"Loading Dataset...",rgb=(54, 209, 23))
print(text)
dataset=prepare_datasetsforhemanth(
    path=r"E:\LLMS\Fine-tuning\data\fka___awesome-chatgpt-prompts"
)
text=rgb_print(text=f"Loaded Dataset....{dataset}",rgb=(214, 237, 7))
print(text)
#===========================tokenizerandmodel=====================
tokenizer=AdvancedPreProcessForHemanth(
    model_type="text",
    pretrained_model_name_or_path="gpt2",
    cache_dir=r"E:\LLMS\Fine-tuning\data",
    
)
print(rgb_print(str(tokenizer.config),rgb=(214, 237, 7)))

tokenizer=tokenizer.process_data()
tokenizer.pad_token = tokenizer.eos_token
model=AiModelForHemanth.load_model(
    model_type="causal_lm",
    model_name_or_path="gpt2",
    cache_dir=r"E:\LLMS\Fine-tuning\data",
)
result=summarizemodelforhemanth(model=model)
for key ,values in result['layers'][0].items():
    print(rgb_print(f":{key}:{values}",rgb=(214, 237, 7)))

printmodelsummaryforhemanth(result)
print(rgb_print(str(model.config),rgb=(214, 237, 7)))
#===========================datapreprocessing======================
preprocessing_data=AdvancedPipeline(
    tokenizer=tokenizer,
    max_sequence_length=50,
)
dataset=preprocessing_data.pre_training_llms(
    dataset=dataset
)
print(rgb_print(str(dataset),rgb=(214, 237,7)))
train_loader=preprocessing_data.prepare_dataloader(dataset=dataset["train"],batch_size=8)
test_loader=preprocessing_data.prepare_dataloader(dataset=dataset["test"],batch_size=8)
eval_loader=preprocessing_data.prepare_dataloader(dataset=dataset["eval"],batch_size=8)

from typing import Union, List, Dict, Any
import time
from rich import print as rprint
def color_text(text: str, rgb: tuple = (255, 0, 0)) -> str:
    """
    Returns the given text wrapped in ANSI escape codes to display it in the specified RGB color.

    Args:
        text: The text to colorize.
        rgb: A tuple of three integers (0-255) representing the Red, Green, and Blue 
             components of the desired color. Defaults to red (255, 0, 0).

    Returns:
        str: The input text wrapped in ANSI escape codes for colorization.
    """
    r, g, b = rgb
    return f"\033[38;2;{r};{g};{b}m{text}\033[0m"

def process_data(data: Union[str, List, Dict], 
                  rgb: tuple = (255, 0, 0)) -> Union[str, List, Dict]:
    """
    Processes the input data, applying color to string elements while preserving 
    the original data structure. Handles potential errors and provides basic 
    performance logging.

    Args:
        data: The data to process. Can be a string, a list, or a dictionary. 
              Lists and dictionaries can contain nested structures of the same types.
        rgb: A tuple of three integers (0-255) representing the Red, Green, and Blue 
             components of the desired color. Defaults to red (255, 0, 0).

    Returns:
        Union[str, List, Dict]: The processed data, with string elements colorized 
                                using ANSI escape codes, or the original data 
                                if an error occurs. 
    """
    start_time = time.time()

    try:
        if isinstance(data, str):
            return color_text(data, rgb)
        elif isinstance(data, list):
            return [process_data(item, rgb) for item in data]
        elif isinstance(data, dict):
            return {key: process_data(value, rgb) for key, value in data.items()}
        else:
            return data  # Return the data as is for unsupported types
    except TypeError as e:
        print(f"Error processing data: {e}")
        return data  # Return original data in case of error
    finally:
        end_time = time.time()
        print(f"Data processed in {end_time - start_time:.4f} seconds")
# Example usage
rprint(process_data("Hello, World!"))
rprint(process_data(["Apple", "Banana", "Cherry"]))
rprint(process_data({"name": "John", "age": "25"}))
rprint(process_data([{"name": "John", "age": "25"}, {"name": "Jane", "age": "30"}]))
# epochs = 2
# learning_rate = 2e-5
# model = model.to(device)
# optimizer = AdamW(model.parameters(), lr=learning_rate)
# total_steps = len(train_loader) * epochs
# scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
# history = train(model=model, train_loader=train_loader, optimizer=optimizer,
#                  scheduler=scheduler, 
#                  epochs=epochs, 
#                  device=device,
#                  save_dir=r"E:\LLMS\Fine-tuning\data\test_model")
# plot_training_metrics(history)
# eval_results = evaluate(model, eval_loader, device)
# print("Evaluation Results:", eval_results)
# test_results = evaluate(model, test_loader, device) 
# print("Testing Results:", test_results) 
# model=load_model_for_inference(model=model,model_dir=r"E:\LLMS\Fine-tuning\data\test_model\best_model_epoch_1",device=device)