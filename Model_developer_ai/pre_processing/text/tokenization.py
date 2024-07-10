import sys
from pathlib import Path

file=Path(__file__).resolve()
sys.path.append(str(file.parents[3]))

from transformers import AutoTokenizer
import torch
from typing import List, Dict, Any, Tuple
from torch.utils.data import Dataset, DataLoader
import tiktoken

class DataTokenization(Dataset):

    def __init__(self,
                 data,
                 tokenizer: AutoTokenizer,
                 seq_max_length: int = 100,
                 pad_to_max_length: bool = True,
                 return_tensors: str = "pt"):
        
        self.tokenizer = tokenizer
        self.seq_max_length = seq_max_length
        self.pad_to_max_length = pad_to_max_length
        self.return_tensors = return_tensors
        self.data = self._process_data(data) 

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if 0 <= idx < len(self.data):
            item = self.data[idx]
            if self.return_tensors == "pt":
                return {
                    "input_ids": torch.tensor(item["input_ids"]),
                    "labels": torch.tensor(item["labels"]),
                    "id": item.get("id"),  
                }
            elif self.return_tensors == "tf":
                import tensorflow as tf
                return {
                    "input_ids": tf.convert_to_tensor(item["input_ids"]),
                    "labels": tf.convert_to_tensor(item["labels"]),
                    "id": item.get("id"), 
                }
            elif self.return_tensors is None:
                return item
            else:
                raise ValueError(
                    "Invalid value for `return_tensors`. Must be one of: 'pt', 'tf', None"
                )
        else:
            raise IndexError("Index out of range.")

    def __len__(self):
        return len(self.data)

    def _process_data(self, data: Any) -> List[Dict[str, Any]]:
        processed_data = []

        # Handle different data formats consistently, ensuring no None values slip through
        if isinstance(data, str):
            processed_data.append({"text": data})
        elif isinstance(data, list) and all(isinstance(item, str) for item in data):
            for item in data:
                if item: # Skip empty strings 
                    processed_data.append({"text": item}) 
        else:
            raise TypeError( 
                "Unsupported data format. Please provide a string or a list of strings."
            )

        # Tokenize and prepare each sample
        for item in processed_data:
            item["input_ids"] = self.tokenizer.encode(item["text"])
            item["labels"] = item["input_ids"].copy()  

            if self.pad_to_max_length:
                self._pad_sequence(item["input_ids"])
                self._pad_sequence(item["labels"])
            else:
                item["input_ids"] = item["input_ids"][: self.seq_max_length]
                item["labels"] = item["labels"][: self.seq_max_length]

        return processed_data

    def _pad_sequence(self, sequence: List[int]):
        padding_length = self.seq_max_length - len(sequence)
        sequence.extend([self.tokenizer.pad_token_id] * padding_length)





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
    """

    def __init__(
        self,
        data,
        tokenizer: tiktoken.Encoding ,
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


def combined_text(example: Dict[str, str]) -> Dict[str, str]:
    """
    Combines text from multiple columns into a single 'text' column.

    Args:
        example (Dict[str, str]): A dictionary representing a single data example.

    Returns:
        Dict[str, str]: The modified data example with combined text.
    """
    combined_text_content = "".join(
        example[col] for col in example.keys()
    )
    return {"text": combined_text_content}

from transformers import AutoTokenizer

def data_tokenization(
    dataset,
    tokenizer: AutoTokenizer,
    seq_max_length: int,
    pad_to_max_length: bool,
    return_tensors: str,
) :
    """
    Tokenizes the text data in the dataset, setting labels the same as input_ids.

    Args:
        dataset (DatasetType): The dataset to tokenize.
        tokenizer (AutoTokenizer): The tokenizer to use for tokenization.
        seq_max_length (int): The maximum sequence length for padding.
        pad_to_max_length (bool): Whether to pad sequences to the maximum length.
        return_tensors (str): The type of tensors to return ('pt' for PyTorch).

    Returns:
        DatasetType: The tokenized dataset.
    """
    try:
        dataset = dataset.map(combined_text, remove_columns=dataset["train"].column_names)
        dataset = dataset.map(
            lambda example: tokenizer(
                example["text"],
                truncation=True,
                padding="max_length" if pad_to_max_length else False,
                max_length=seq_max_length,
            ),
            batched=True,
            remove_columns=["text"],
        )

        # Set labels equal to input_ids
        dataset = dataset.map(lambda example: {'labels': example['input_ids']}, batched=True)

        dataset.set_format(type="torch", columns=["input_ids", "labels", "attention_mask"])
        return dataset
    except Exception as e:
        raise SystemExit(f"An error occurred during data tokenization: {e}") from e


def create_data_loaders(
    dataset, batch_size: int
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates data loaders for training, testing, and evaluation.

    Args:
        dataset (DatasetType): The tokenized dataset.
        batch_size (int): The batch size for the data loaders.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: A tuple containing the train, test, and eval data loaders.
    """
    try:
        train_loader = DataLoader(dataset["train"], batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(dataset["test"], batch_size=batch_size)
        eval_loader = DataLoader(dataset["eval"], batch_size=batch_size)

        return train_loader, test_loader, eval_loader
    except Exception as e:
        raise SystemExit(
            f"An error occurred while creating data loaders: {e}"
        ) from e


