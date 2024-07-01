import logging
from typing import Dict, List, Optional,Tuple, Union
from datasets import load_dataset, Dataset, DatasetDict
from transformers import PreTrainedTokenizer, AutoTokenizer
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedPipeline:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_sequence_length: int,
        padding: str = "max_length",
        return_tensors: str = "pt"
    ):
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
        self.padding = padding
        self.return_tensors = return_tensors

    @staticmethod
    def combined_text(dataset: Union[str, Dict, Dataset, DatasetDict]) -> Dict[str, str]:
        """
        Combines text from multiple columns into a single 'text' column.
        """
        combined_text_content = " ".join(
            dataset[col] for col in dataset.keys()
        )
        return {"text": combined_text_content}

    def process_dataset(self, dataset: Union[str, Dict, Dataset, DatasetDict]) -> DatasetDict:
        """
        Process the dataset by combining text and tokenizing.
        """
        try:
           
            
            logger.info(f"Dataset loaded successfully. Number of splits: {len(dataset)}")

            for split_name in dataset.keys():
                dataset[split_name] = dataset[split_name].map(
                    self.combined_text, 
                    remove_columns=dataset[split_name].column_names
                )
                dataset[split_name] = dataset[split_name].map(
                    lambda example: self.tokenizer(
                        example["text"],
                        truncation=True,
                        padding=self.padding,
                        max_length=self.max_sequence_length,
                    ),
                    batched=True,
                    remove_columns=["text"],
                )
                dataset[split_name] = dataset[split_name].map(
                    lambda example: {'labels': example['input_ids']}, 
                    batched=True
                )
                dataset[split_name].set_format(
                    type=self.return_tensors, 
                    columns=["input_ids", "labels", "attention_mask"]
                )

            logger.info("Dataset processed successfully.")
            return dataset
        except Exception as e:
            logger.error(f"Error processing dataset: {str(e)}")
            raise

    def pre_training_llms(
        self,
        dataset: Union[str, Dict, Dataset, DatasetDict]
    ) -> DatasetDict:
        """
        Process the dataset for pre-training LLMs.
        """
        logger.info("Starting pre-training LLMs.")
        return self.process_dataset(dataset)

    def fine_tuning(
        self,
        dataset: Union[str, Dict, Dataset, DatasetDict],
        prompt_template: Optional[str] = None,
        target_columns: Optional[List[str]] = None,
        instruction: Optional[str] = None,
        response: Optional[str] = None
    ) -> DatasetDict:
        """
        Process the dataset for fine-tuning.
        """
        logger.info("Starting fine-tuning.")
        # Apply prompt template and target columns processing before tokenization
        
        if prompt_template is not None:
            dataset = self.apply_prompt_template(dataset, prompt_template, target_columns, instruction, response)
        return self.process_dataset(dataset)

    def instruction_tuning(
        self,
        dataset: Union[str, Dict, Dataset, DatasetDict],
        prompt_template: Optional[str] = None,
        target_columns: Optional[List[str]] = None,
        instruction: Optional[str] = None,
        response: Optional[str] = None,
        api_function: Optional[callable] = None
    ) -> DatasetDict:
        """
        Process the dataset for instruction tuning.
        """
        logger.info("Starting instruction tuning.")
        # Apply prompt template, target columns processing, and API function before tokenization
        if prompt_template is not None:
            dataset = self.apply_prompt_template(dataset, prompt_template, target_columns, instruction, response)
        if api_function is not None:
            dataset = self.apply_api_function(dataset, api_function)
        return self.process_dataset(dataset)

    def apply_prompt_template(
        self,
        dataset: Union[str, Dict, Dataset, DatasetDict],
        prompt_template: str,
        target_columns: Optional[List[str]] = None,
        instruction: Optional[str] = None,
        response: Optional[str] = None
    ) -> Union[Dataset, DatasetDict]:
        """
        Apply prompt template and process target columns for the given dataset.
    
        Args:
            dataset: Input dataset in various formats.
            prompt_template: Template string for the prompt.
            target_columns: List of target column names to process.
            instruction: Optional instruction to include in the prompt.
            response: Optional response to include in the prompt.
    
        Returns:
            Processed dataset with applied prompt template.
        """
        try:
            # # Load dataset if it's a string (file path)
            # if isinstance(dataset, str):
            #     dataset = Dataset.from_json(dataset)
            #     logger.info(f"Loaded dataset from file: {dataset}")
    
            # # Convert single Dataset to DatasetDict if necessary
            # if isinstance(dataset, Dataset):
            #     dataset = DatasetDict({"train": dataset})
            #     logger.info("Converted single Dataset to DatasetDict")
    
            # # Ensure dataset is a DatasetDict
            # if not isinstance(dataset, DatasetDict):
            #     raise ValueError("Invalid dataset format. Expected DatasetDict.")
    
            # Process each split in the dataset
            for split_name, split_dataset in dataset.items():
                logger.info(f"Processing split: {split_name}")
    
                # Define a function to apply the prompt template
                def apply_template(example):
                    processed_example = example.copy()
                    for column in split_dataset.column_names:
                        if target_columns and column not in target_columns:
                            continue
    
                        prompt = prompt_template.format(
                            instruction=instruction or "",
                            response=response or "",
                            **{col: example[col] for col in split_dataset.column_names}
                        )
                        processed_example[column] = f"{example[column]} {prompt}".strip()
    
                    return processed_example
    
                # Apply the template to the dataset
                processed_split = split_dataset.map(
                    apply_template,
                    desc=f"Applying prompt template to {split_name}"
                )
    
                # Update the dataset with the processed split
                dataset[split_name] = processed_split
    
            logger.info("Prompt template applied successfully to all splits")
            return dataset
    
        except Exception as e:
            logger.error(f"Error occurred while applying prompt template: {str(e)}")
            raise

    def apply_api_function(
        self,
        dataset: Union[str, Dict, Dataset, DatasetDict],
        api_function: callable
    ) -> Union[Dataset, DatasetDict]:
        """
        Apply API function to the dataset.
        """
        logger.info("Applying API function to the dataset.")
        # Implement the logic to apply the API function to the dataset
        # This may involve calling the API function on the dataset examples
        # You can use the dataset mapping functions to apply the transformations
        return dataset

    def prepare_dataloader(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool = True
    ) -> DataLoader:
        """
        Prepare a DataLoader from a processed dataset.
        """
        logger.info("Preparing DataLoader.")
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


class AdvancedDatasetProcessor:
    def __init__(self, 
        tokenizer:PreTrainedTokenizer, 
        dataset_path: str, 
        max_seq_length: int = 300, 
        batch_size: int = 32 ):

        self.dataset_path = dataset_path
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.tokenizer =tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
    def load_and_process_dataset(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        dataset = self._load_dataset()
        dataset = self._combine_columns(dataset)
        train_dataset, test_dataset, eval_dataset = self._split_dataset(dataset)
        train_dataset = self._tokenize_dataset(train_dataset)
        test_dataset = self._tokenize_dataset(test_dataset)
        eval_dataset = self._tokenize_dataset(eval_dataset)
        train_loader = self._create_dataloader(train_dataset)
        test_loader = self._create_dataloader(test_dataset)
        eval_loader = self._create_dataloader(eval_dataset)
        return train_loader, test_loader, eval_loader

    def _load_dataset(self) -> Dataset:
        logger.info(f"Loading dataset from {self.dataset_path}")
        try:
            dataset = load_dataset(self.dataset_path)
            return dataset[list(dataset.keys())[0]]  
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            raise

    def _combine_columns(self, dataset: Dataset) -> Dataset:
        logger.info("Combining all columns into a single 'text' column")
        def combine_text(example):
            return {"text": " ".join(str(value) for value in example.values())}
        return dataset.map(combine_text, remove_columns=dataset.column_names)

    def _split_dataset(self, dataset: Dataset) -> Tuple[Dataset, Dataset, Dataset]:
        logger.info("Splitting dataset into train, test, and eval sets")
        train_test = dataset.train_test_split(test_size=0.2)
        test_eval = train_test['test'].train_test_split(test_size=0.5)
        return train_test['train'], test_eval['train'], test_eval['test']

    def _tokenize_dataset(self, dataset: Dataset) -> Dataset:
        logger.info("Tokenizing dataset")
        def tokenize_and_chunk(examples: Dict[str, List[str]]) -> Dict[str, List[torch.Tensor]]:
            tokenized_inputs = []
            for text in examples['text']:
                chunks = [text[i:i+self.max_seq_length] for i in range(0, len(text), self.max_seq_length)]
                for chunk in chunks:
                    tokens = self.tokenizer(chunk, padding="max_length", truncation=True, 
                                            max_length=self.max_seq_length, return_tensors="pt")
                    tokenized_inputs.append({
                        'input_ids': tokens['input_ids'].squeeze(),
                        'attention_mask': tokens['attention_mask'].squeeze()
                    })
            return {
                'input_ids': [item['input_ids'] for item in tokenized_inputs],
                'attention_mask': [item['attention_mask'] for item in tokenized_inputs],
                'labels': [item['input_ids'] for item in tokenized_inputs]
            }
        
        return dataset.map(tokenize_and_chunk, batched=True, remove_columns=dataset.column_names)

    def _create_dataloader(self, dataset: Dataset) -> DataLoader:
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

# if __name__ == "__main__":
#     processor = AdvancedDatasetProcessor(
#         tokenizer=AutoTokenizer.from_pretrained('gpt2',cache_dir=r"C:\Users\heman\Desktop\Coding\data"),
#           dataset_path=r"C:\Users\heman\Desktop\Coding\data\fka___awesome-chatgpt-prompts",
#         )
#     try:
#         train_loader, test_loader, eval_loader = processor.load_and_process_dataset()

#         print(next(iter(train_loader))['input_ids'][0])
#         logger.info("Dataset processing completed successfully")
#     except Exception as e:
#         logger.error(f"An error occurred during dataset processing: {str(e)}")  
# # Example usage
# if __name__ == "__main__":
#     tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
#     pipeline = AdvancedPipeline(
#         tokenizer=tokenizer,
#         max_sequence_length=128
#     )

#     example_dataset = r"C:\Users\heman\Desktop\Coding\data\fka___awesome-chatgpt-prompts"  # or a dictionary, Dataset, or DatasetDict
#     example_dataset=prepare_datasetsforhemanth(example_dataset )
#     # # Pre-training LLMs
#     # logger.info("Performing pre-training LLMs task...")
#     # pre_training_result = pipeline.pre_training_llms(example_dataset)
#     # for split_name, split_dataset in pre_training_result.items():
#     #     logger.info(f"Pre-training result shape for {split_name}: {tokenizer.decode(split_dataset['input_ids'][0])}")

#     # Fine-tuning
    
#     logger.info("Performing fine-tuning task...")
#     fine_tuning_result = pipeline.fine_tuning(
#         dataset=example_dataset,
#         prompt_template="Kandimalla Hemanth",
#         target_columns=[ "prompt"],
#         instruction="Perform the action",
#         response="Completed action"
#                                             )
#     for split_name, split_dataset in fine_tuning_result.items():
#         logger.info(f"Fine-tuning result shape for {split_name}: {tokenizer.decode(split_dataset['input_ids'][0])}")

#     # # Instruction tuning
#     # logger.info("Performing instruction tuning task...")
#     # instruction_tuning_result = pipeline.instruction_tuning(example_dataset)
#     # for split_name, split_dataset in instruction_tuning_result.items():
#     #     logger.info(f"Instruction tuning result shape for {split_name}: {split_dataset['input_ids'].shape}")

#     # # Prepare DataLoader (example for the 'train' split)
#     # batch_size = 32
#     # train_dataloader = pipeline.prepare_dataloader(instruction_tuning_result['train'], batch_size=batch_size)
#     # train_dataloader = pipeline.prepare_dataloader(instruction_tuning_result['test'], batch_size=batch_size)
#     # logger.info(f"DataLoader created for 'train' split with batch size: {batch_size}")