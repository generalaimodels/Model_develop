import sys
from pathlib import Path
file=Path(__file__).resolve()
sys.path.append(str(file.parents[2]))

from Model_developer_ai.FAST_ANALYSIS import (
    AdvancedPreProcessForHemanth
)
from pre_processing.text import (
    AdvancedDatasetProcessor,
    AdvancedPipeline
)
tokenizer=AdvancedPreProcessForHemanth(
    model_type="text",
    pretrained_model_name_or_path='gpt2',   
)
tokenizer=tokenizer.process_data()
tokenizer.pad_token=tokenizer.eos_token

Dataset=AdvancedDatasetProcessor(
    tokenizer=tokenizer,
    dataset_path=r"E:\LLMS\Fine-tuning\data\data",
    max_seq_length=100,
    batch_size=32,
)

train_loader, test_loader, eval_loader=Dataset.load_and_process_dataset()
print(next(iter(train_loader))['input_ids'][0])
