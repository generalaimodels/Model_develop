import sys
from pathlib import Path
file=Path(__file__).resolve()
sys.path.append(str(file.parents[1]))
from FAST_ANALYSIS import prepare_datasetsforhemanth,DatasetDictForHemanth
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Sampler
from torch.utils.data import IterableDataset
import tiktoken
from typing import List, Dict, Union, Any
import tiktoken
TOKENIZER = tiktoken.encoding_for_model("gpt2")

print(TOKENIZER.n_vocab)
print(TOKENIZER.eot_token)
print(TOKENIZER.max_token_value)
print(TOKENIZER.name)
# print(TOKENIZER._core_bpe)
# print(TOKENIZER._mergeable_ranks)
# print(TOKENIZER.__dict__)
print(TOKENIZER.__doc__)

class DataHemanthGPT2(Dataset):
    """
    A generalized dataset class for GPT-2 text processing.

    This class handles various input data formats, tokenizes text using the
    GPT-2 tokenizer, and prepares data for training/evaluation.

    Args:
        data (Union[List[str], List[List[str]], Dict, str, Dict[str, Dict[str]], Dict[str, Dict[List[str]]]]): 
            The input data.
            Supported formats:
                - List of strings: Each string is a separate sample.
                - List of lists of strings: Each inner list represents a sample,
                  potentially with multiple segments.
                - Dictionary: Keys can be used as IDs, and values should be
                  strings or lists of strings.
                - String: A single text sample.
                - Dictionary of dictionaries:
                    - Keys of the outer dictionary can be used as IDs,
                      values are inner dictionaries.
                    - Inner dictionaries can have 'text' as a key with a string value 
                      or a list of strings.
                - Dictionary of dictionaries with lists:
                    - Keys of the outer dictionary can be used as IDs,
                      values are inner dictionaries.
                    - Inner dictionaries can have 'text' as a key with a list of strings value.
        tokenizer (tiktoken.Encoding): The GPT-2 tokenizer.
        seq_max_length (int): The maximum sequence length for padding/truncation.
        pad_to_max_length (bool, optional): Whether to pad sequences to
            `seq_max_length`. Defaults to True.
        return_tensors (str, optional): The desired tensor format for output.
            Can be "pt" (PyTorch), "tf" (TensorFlow), or None. Defaults to "pt".

    Raises:
        TypeError: If `data` is not of a supported type.
        ValueError: If an invalid `return_tensors` value is provided.
    Example:# Example 1: List of Strings
data = ["This is the first sample.", "This is the second sample."]
dataset = DataHemanthGPT2(data,
                          tokenizer=TOKENIZER,
                          return_tensors="pt",
                          pad_to_max_length=False,
                          seq_max_length=512)
print(dataset[0])
# Example 2: Dictionary with Strings
data = {"id1": "Sample 1", "id2": "Sample 2"}
dataset = DataHemanthGPT2(data,
                          tokenizer=TOKENIZER,
                          return_tensors="pt",
                          pad_to_max_length=False,
                          seq_max_length=512)
print(dataset[0])
# Example 3: Dictionary with Lists of Strings
data = {"id1": ["Sample 1 part 1", "Sample 1 part 2"], "id2": ["Sample 2"]}
dataset = DataHemanthGPT2(data,
                          tokenizer=TOKENIZER,
                          return_tensors="pt",
                          pad_to_max_length=False,
                          seq_max_length=512)
print(dataset[0])
# Example 4: Dictionary of Dictionaries with Strings
data = {"id1": {"text": "Sample 1"}, "id2": {"text": "Sample 2"}}
dataset = DataHemanthGPT2(data,
                          tokenizer=TOKENIZER,
                          return_tensors="pt",
                          pad_to_max_length=False,
                          seq_max_length=512)
print(dataset[0])
# Example 5: Dictionary of Dictionaries with Lists of Strings
data = {"id1": {"text": ["Sample 1 part 1", "Sample 1 part 2"]}, "id2": {"text": ["Sample 2"]}}
dataset = DataHemanthGPT2(data,
                          tokenizer=TOKENIZER,
                          return_tensors="pt",
                          pad_to_max_length=False,
                          seq_max_length=512)


print(dataset[0])
# Accessing data samples
sample = dataset[0]  # Get the first sample
input_ids = sample["input_ids"]
labels = sample["labels"]
id = sample.get("id")

print(input_ids, labels, id)


# Example 1: List of Strings
data = ["This is the first sample.", "This is the second sample."]
dataset = DataHemanthGPT2(data,
                          tokenizer=TOKENIZER,
                          return_tensors="pt",
                          pad_to_max_length=False,
                          seq_max_length=512)
print(dataset[0])
# Example 2: Dictionary with Strings
data = {"id1": "Sample 1", "id2": "Sample 2"}
dataset = DataHemanthGPT2(data,
                          tokenizer=TOKENIZER,
                          return_tensors="pt",
                          pad_to_max_length=False,
                          seq_max_length=512)
print(dataset[0])
# Example 3: Dictionary with Lists of Strings
data = {"id1": ["Sample 1 part 1", "Sample 1 part 2"], "id2": ["Sample 2"]}
dataset = DataHemanthGPT2(data,
                          tokenizer=TOKENIZER,
                          return_tensors="pt",
                          pad_to_max_length=False,
                          seq_max_length=512)
print(dataset[0])
# Example 4: Dictionary of Dictionaries with Strings
data = {"id1": {"text": "Sample 1"}, "id2": {"text": "Sample 2"}}
dataset = DataHemanthGPT2(data,
                          tokenizer=TOKENIZER,
                          return_tensors="pt",
                          pad_to_max_length=False,
                          seq_max_length=512)
print(dataset[0])
# Example 5: Dictionary of Dictionaries with Lists of Strings
data = {"id1": {"text": ["Sample 1 part 1", "Sample 1 part 2"]}, "id2": {"text": ["Sample 2"]}}
dataset = DataHemanthGPT2(data,
                          tokenizer=TOKENIZER,
                          return_tensors="pt",
                          pad_to_max_length=False,
                          seq_max_length=512)
print(dataset[0])
# Accessing data samples
sample = dataset[0]  # Get the first sample
input_ids = sample["input_ids"]
labels = sample["labels"]
id = sample.get("id")

print(input_ids, labels, id)
    """

    def __init__(
        self,
        data,
        tokenizer: tiktoken.Encoding = TOKENIZER,
        seq_max_length: int = 1024,
        pad_to_max_length: bool = True,
        return_tensors: str = "pt",
    ):
        if not isinstance(tokenizer, tiktoken.Encoding):
            raise TypeError("`tokenizer` must be a tiktoken.Encoding object.")
        if not isinstance(seq_max_length, int) or seq_max_length <= 0:
            raise ValueError("`seq_max_length` must be a positive integer.")
        if not isinstance(pad_to_max_length, bool):
            raise TypeError("`pad_to_max_length` must be a boolean.")
        if return_tensors not in ["pt", "tf", None]:
            raise ValueError(
                "`return_tensors` must be one of: 'pt', 'tf', None."
            )

        self.tokenizer = tokenizer
        self.seq_max_length = seq_max_length
        self.pad_to_max_length = pad_to_max_length
        self.return_tensors = return_tensors

        # Validate and process the input data
        self.data = self._process_data(data)

    def _process_data(self, data: Any) -> List[Dict[str, Any]]:
        """
        Validates and preprocesses the input data into a standardized format.

        Args:
            data: The input data in one of the supported formats.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, where each dictionary
                represents a data sample and contains keys like "input_ids",
                "labels", and potentially an "id" if provided.

        Raises:
            TypeError: If the input data format is not supported.
        """
        processed_data = []
        if isinstance(data, str):
            processed_data.append({"text": data})
        elif isinstance(data, list) and all(isinstance(item, str) for item in data):
            processed_data.extend([{"text": item} for item in data])
        elif (
            isinstance(data, list)
            and all(isinstance(item, list) for item in data)
            and all(isinstance(subitem, str) for sublist in data for subitem in sublist)
        ):
            processed_data.extend([{"text": " ".join(item)} for item in data])
        elif isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, str):
                    processed_data.append({"id": key, "text": value})
                elif isinstance(value, list) and all(isinstance(item, str) for item in value):
                    processed_data.append({"id": key, "text": " ".join(value)})
                elif isinstance(value, dict) and "text" in value:
                    if isinstance(value["text"], str):
                        processed_data.append({"id": key, "text": value["text"]})
                    elif isinstance(value["text"], list) and all(isinstance(item, str) for item in value["text"]):
                        processed_data.append({"id": key, "text": " ".join(value["text"])})
                    else:
                        raise TypeError("Unsupported data format for inner dictionary.")
                else:
                    raise TypeError("Unsupported data format for dictionary value.")
        else:
            raise TypeError(
                "Unsupported data format. Please provide a string, "
                "list of strings, list of lists of strings, or a dictionary."
            )

        # Tokenize and prepare each sample
        for item in processed_data:
            item["input_ids"] = self.tokenizer.encode(item["text"])
            item["labels"] = item["input_ids"].copy()  # Assuming labels are the same

            if self.pad_to_max_length:
                self._pad_sequence(item["input_ids"])
                self._pad_sequence(item["labels"])
            else:
                item["input_ids"] = item["input_ids"][: self.seq_max_length]
                item["labels"] = item["labels"][: self.seq_max_length]
        return processed_data

    def _pad_sequence(self, sequence: List[int]):
        """Pads a sequence to `self.seq_max_length`."""
        padding_length = self.seq_max_length - len(sequence)
        sequence.extend([self.tokenizer.pad_token_id] * padding_length)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Retrieves a data sample.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            Dict[str, Any]: A dictionary containing the processed data sample,
                including "input_ids", "labels", and potentially an "id".

        Raises:
            IndexError: If the provided index is out of range.
        """
        if 0 <= idx < len(self.data):
            item = self.data[idx]
            if self.return_tensors == "pt":
                import torch

                return {
                    "input_ids": torch.tensor(item["input_ids"]),
                    "labels": torch.tensor(item["labels"]),
                    "id": item.get("id"),  # Include ID if available
                }
            elif self.return_tensors == "tf":
                import tensorflow as tf

                return {
                    "input_ids": tf.convert_to_tensor(item["input_ids"]),
                    "labels": tf.convert_to_tensor(item["labels"]),
                    "id": item.get("id"),  # Include ID if available
                }
            elif self.return_tensors is None:
                return item
            else:
                raise ValueError(
                    "Invalid value for `return_tensors`. Must be one of: 'pt', 'tf', None"
                )
        else:
            raise IndexError("Index out of range.")
        


dataset=prepare_datasetsforhemanth(
    "fka/awesome-chatgpt-prompts",
    cache_dir=r"E:\LLMS\Fine-tuning\data"
)

print(dataset)
dataset=DatasetDictForHemanth(dataset)
dataset = dataset['train']  # Extract the training split
# Combine all text columns into a new column
def combine_text_columns(row):
    combined_text = " ".join([str(row[col]) for col in dataset.column_names])
    return {'text': combined_text}  # Return as a dictionary
# Apply the function to each row to create the new column 
dataset = dataset.map(combine_text_columns, remove_columns=dataset.column_names)
# Print the updated dataset structure
print(dataset)
# Access the combined text column from the first example
combined_text_example = dataset[0]['text']  # Access directly 
print(f"Text : {combined_text_example}") 

dataset=DataHemanthGPT2(
    data=combined_text_example + " 1234567890 ",
    tokenizer=TOKENIZER,
    seq_max_length=1024,
    return_tensors="pt",
    pad_to_max_length=False,
)
print(dataset[0])

print(TOKENIZER.decode(dataset[0]["input_ids"].numpy()))
print(TOKENIZER.decode(dataset[0]["labels"].numpy()))
