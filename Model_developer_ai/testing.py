from FAST_ANALYSIS import (
AdvancedModelLoader,                        
AdvancedPreProcessForHemanth,
prepare_datasetsforhemanth,
summarizemodelforhemanth,
printmodelsummaryforhemanth,
get_dataset_info_for_hemanth,
rgb_print,
create_data_loaders,
data_tokenization,

)
from pre_processing.data_processing import (
    AdvancedDataset,
    AdvancedDataset_update,
    SimpleDataset,
    create_dataloader_1
    
)
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
#============================data_collection========================
tokenizer=AdvancedPreProcessForHemanth(
    model_type="text",
    pretrained_model_name_or_path='gpt2',
    cache_dir=r'C:\Users\heman\Desktop\Coding\data'
)
tokenizer=tokenizer.process_data()

dataset=prepare_datasetsforhemanth(
    path=r"C:\Users\heman\Desktop\Coding\data\fka___awesome-chatgpt-prompts"
)
print(rgb_print(text=dataset,rgb=(229, 235, 52)))
dataset_info=get_dataset_info_for_hemanth(dataset=dataset)
dataset=data_tokenization(dataset=dataset,
                  tokenizer=tokenizer,
                  seq_max_length=100,
                  pad_to_max_length=False,
                  return_tensors="pt")

dataset=AdvancedDataset(data=dataset['train'])

train_loader=create_dataloader_1(dataset=dataset,batch_size=2)
test_loader=create_dataloader_1(dataset=dataset,batch_size=2)
print(next(iter(train_loader)))



      
# model=AdvancedModelLoader.load_model(
#     task="causal_lm",
#     cache_dir=r"C:\Users\heman\Desktop\Coding\data",
#     model_name="gpt2",
    
# )
# result=summarizemodelforhemanth(model=model)
# prepare_datasetsforhemanth(result)
