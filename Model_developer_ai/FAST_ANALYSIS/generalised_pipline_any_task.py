
import logging
import torch
import random
import numpy as np
from itertools import chain
from FAST_ANALYSIS.generalised_data_loader import prepare_datasetsforhemanth
from datasets import Audio
from transformers import AutoFeatureExtractor
from typing import Dict, List,Any, Optional, Tuple, Union
from transformers import ( AutoConfig,
                          AutoModelForImageClassification, 
                          AutoImageProcessor,
                           PreTrainedTokenizer
)
from datasets import load_dataset, DatasetDict
from torchvision.transforms import ( Compose,
                                   Lambda, 
                                   RandomResizedCrop, 
                                   RandomHorizontalFlip, 
                                   ToTensor,
                                   Normalize,
                                   Resize,
                                   CenterCrop
)
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Sampler
from torch.utils.data import IterableDataset
import tiktoken
from typing import List, Dict, Union, Any,Tuple
import random
import tiktoken
import torch
import torch
import functools
import numpy as np
import random
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Any, List, Union, Optional

from torch.utils.data import Dataset, DataLoader
from torch.utils.data import IterableDataset
from torch.utils.data.dataloader import DataLoader
from transformers import AutoTokenizer, default_data_collator, AutoModelForCausalLM

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

#===============================================AudioDataPipeline==============================================================
class AudioDataPipeline:
    def __init__(
        self, dataset_name: str, model_args: Any, data_args: Any
    ) -> None:
        self.dataset_name = dataset_name
        self.model_args = model_args
        self.data_args = data_args
        self.feature_extractor = None
        self.raw_datasets = None
        self.labels = None
        self.label2id = {}
        self.id2label = {}
        self.initialize_pipeline()

    def initialize_pipeline(self) -> None:
        """Initializes the feature extractor and datasets, and prepares label mappings."""
        self.load_feature_extractor()
        self.load_datasets()
        self.prepare_labels()

    def load_feature_extractor(self) -> None:
        """Loads the feature extractor based on the provided model arguments."""
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            self.model_args.feature_extractor_name or self.model_args.model_name_or_path,
            return_attention_mask=self.model_args.attention_mask,
            cache_dir=self.model_args.cache_dir,
            revision=self.model_args.model_revision,
            token=self.model_args.token,
            trust_remote_code=self.model_args.trust_remote_code,
        )

    def load_datasets(self) -> None:
        """Loads and processes the datasets, setting the correct audio column type."""
        self.raw_datasets = prepare_datasetsforhemanth(self.dataset_name)
        self.raw_datasets = self.raw_datasets.cast_column(
            self.data_args.audio_column_name, Audio(sampling_rate=self.feature_extractor.sampling_rate)
        )

    def prepare_labels(self) -> None:
        """Prepares label mappings for use in the model."""
        self.labels = self.raw_datasets['train'].features[self.data_args.label_column_name].names
        for i, label in enumerate(self.labels):
            self.label2id[label] = str(i)
            self.id2label[str(i)] = label

    def train_transforms(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Processes a batch of data for training."""
        subsampled_wavs = [self.random_subsample(audio["array"])
                           for audio in batch[self.data_args.audio_column_name]]
        inputs = self.feature_extractor(subsampled_wavs, sampling_rate=self.feature_extractor.sampling_rate)
        return {
            self.feature_extractor.model_input_names[0]: inputs[self.feature_extractor.model_input_names[0]],
            "labels": [self.label2id[label] for label in batch[self.data_args.label_column_name]]
        }

    def val_transforms(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Processes a batch of data for validation."""
        wavs = [audio["array"] for audio in batch[self.data_args.audio_column_name]]
        inputs = self.feature_extractor(wavs, sampling_rate=self.feature_extractor.sampling_rate)
        return {
            self.feature_extractor.model_input_names[0]: inputs[self.feature_extractor.model_input_names[0]],
            "labels": [self.label2id[label] for label in batch[self.data_args.label_column_name]]
        }

    def random_subsample(self, audio: List[float]) -> List[float]:
        """Randomly subsamples the audio data to a fixed length."""
        max_samples = int(self.data_args.max_length_seconds * self.feature_extractor.sampling_rate)
        if len(audio) <= max_samples:
            return audio
        start = random.randint(0, len(audio) - max_samples)
        return audio[start:start + max_samples]


#====================================================NLP=====================================================


class DataProcessor:
    def __init__(self, tokenizer: PreTrainedTokenizer, block_size: int, max_length: int):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.max_length = max_length

    def tokenize(self, examples: Dict[str, List[str]], text_column_name: str) -> Dict[str, List[int]]:
        """Tokenizes text data in a dataset."""
        return {text_column_name: self.tokenizer(examples[text_column_name], truncation=True, padding='max_length', max_length=self.max_length)["input_ids"]}

    def group_texts(self, examples: Dict[str, List[int]]) -> Dict[str, List[List[int]]]:
        """Groups texts into chunks of the specified block size."""
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // self.block_size) * self.block_size
        return {
            k: [t[i:i + self.block_size] for i in range(0, total_length, self.block_size)]
            for k, t in concatenated_examples.items()
        }

class Transformer:
    def __init__(self, tokenizer: PreTrainedTokenizer, fim_rate: float, fim_spm_rate: float, max_length: int):
        self.tokenizer = tokenizer
        self.fim_rate = fim_rate
        self.fim_spm_rate = fim_spm_rate
        self.max_length = max_length
        self.rng = np.random.default_rng(seed=42)
        """
        # Example usage
        #Initialize tokenizer, data processor, and transformer
        tokenizer = PreTrainedTokenizer.from_pretrained('bert-base-uncased')
        processor = DataProcessor(tokenizer, block_size=128, max_length=512)
        transformer = Transformer(tokenizer, fim_rate=0.1, fim_spm_rate=0.1, max_length=512)
        dataset = {'text': ['some example text']}
        tokenized = processor.tokenize(dataset, 'text')
        grouped = processor.group_texts(tokenized)
        transformed = transformer.apply_fim(grouped)
        
        
        """

    def fim_transform(self, example: List[int]) -> List[int]:
        """Applies FIM transformation to a single example."""
        if self.rng.binomial(1, self.fim_rate):
            boundaries = sorted(self.rng.integers(low=0, high=len(example) + 1, size=2))
            prefix, suffix = example[:boundaries[0]], example[boundaries[1]:]
            middle = example[boundaries[0]:boundaries[1]]
            transformed_example = (
                [self.tokenizer.cls_token_id] + prefix + [self.tokenizer.sep_token_id] + suffix +
                [self.tokenizer.mask_token_id] + middle
            ) if self.rng.binomial(1, self.fim_spm_rate) else (
                [self.tokenizer.cls_token_id] + prefix + [self.tokenizer.sep_token_id] + suffix +
                [self.tokenizer.mask_token_id] + middle
            )
        else:
            transformed_example = example
        return transformed_example[:self.max_length]

    def apply_fim(self, examples: Dict[str, List[List[int]]]) -> Dict[str, List[Any]]:
        """Applies FIM transformation to a batch of examples."""
        transformed = [self.fim_transform(ids) for ids in examples["input_ids"]]
        padded_sequences, attention_masks = [], []

        for seq in transformed:
            if len(seq) < self.max_length:
                padded_seq = seq + [self.tokenizer.pad_token_id] * (self.max_length - len(seq))
                attention_mask = [1] * len(seq) + [0] * (self.max_length - len(seq))
            else:
                padded_seq = seq[:self.max_length]
                attention_mask = [1] * self.max_length
            padded_sequences.append(padded_seq)
            attention_masks.append(attention_mask)

        examples["input_ids"] = padded_sequences
        examples["labels"] = padded_sequences
        examples["attention_mask"] = attention_masks
        return examples



def tokenize_function(examples: Dict[str, List[str]], tokenizer: PreTrainedTokenizer, text_column_name: str) -> Dict[str, List[int]]:
    """Tokenizes text data in a dataset."""
    return tokenizer(examples[text_column_name])

def group_texts(examples: Dict[str, List[int]], block_size: int) -> Dict[str, List[List[int]]]:
    """Groups texts into chunks of specified block size."""
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // block_size) * block_size
    return {
        k: [t[i:i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }

def fim_transform(example: List[int], tokenizer: PreTrainedTokenizer, fim_rate: float, fim_spm_rate: float, max_length: int) -> List[int]:
    """Applies FIM transformation to a single example."""
    np_rng = np.random.default_rng(seed=42)
    if np_rng.binomial(1, fim_rate):
        boundaries = sorted(np_rng.integers(low=0, high=len(example) + 1, size=2))
        prefix, middle, suffix = example[:boundaries[0]], example[boundaries[0]:boundaries[1]], example[boundaries[1]:]
        transformed_example = (
            [tokenizer.cls_token_id] + prefix + [tokenizer.sep_token_id] + suffix + [tokenizer.mask_token_id] + middle
        ) if np_rng.binomial(1, fim_spm_rate) else (
            [tokenizer.cls_token_id] + prefix + [tokenizer.sep_token_id] + suffix + [tokenizer.mask_token_id] + middle
        )
    else:
        transformed_example = example
    return transformed_example[:max_length]

def apply_fim(examples: Dict[str, List[List[int]]], tokenizer: PreTrainedTokenizer, fim_rate: float, fim_spm_rate: float, max_length: int) -> Dict[str, List[Any]]:
    """Applies FIM transformation to a batch of examples."""
    transformed = [fim_transform(ids, tokenizer, fim_rate, fim_spm_rate, max_length) for ids in examples["input_ids"]]
    
    # Pad or truncate sequences to max_length
    padded_sequences = []
    attention_masks = []
    for seq in transformed:
        if len(seq) < max_length:
            padded_seq = seq + [tokenizer.pad_token_id] * (max_length - len(seq))
            attention_mask = [1] * len(seq) + [0] * (max_length - len(seq))
        else:
            padded_seq = seq[:max_length]
            attention_mask = [1] * max_length
        padded_sequences.append(padded_seq)
        attention_masks.append(attention_mask)
    
    examples["input_ids"] = padded_sequences
    examples["labels"] = padded_sequences
    examples["attention_mask"] = attention_masks
    return examples

def load_and_preprocess_dataset(dataset_name: str, tokenizer: PreTrainedTokenizer, block_size: int, fim_rate: float, fim_spm_rate: float, max_length: int) -> Dict[str, Any]:
    """Loads and preprocesses the dataset."""
    raw_datasets =prepare_datasetsforhemanth(dataset_name)
    tokenized_datasets = raw_datasets.map(
        lambda examples: tokenize_function(examples, tokenizer, "text"),
        batched=True,
        remove_columns=["text"],
    )
    lm_datasets = tokenized_datasets.map(
        lambda examples: group_texts(examples, block_size),
        batched=True,
    )
    fim_datasets = lm_datasets.map(
        lambda examples: apply_fim(examples, tokenizer, fim_rate, fim_spm_rate, max_length),
        batched=True,
    )
    return fim_datasets


#======================================================IMAGE_CLASSIFICATION=======================================================

class ImageProcessingPipeline:
    def __init__(self, model_args: Dict, data_args: Dict, training_args: Dict):
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        self.config = self._load_config()
        self.model = self._load_model()
        self.image_processor = self._load_image_processor()
        self.dataset = self._load_dataset()
        self.transforms = self._setup_transforms()
        """_summary_
        # Example usage of the pipeline
        model_args = {
            "model_name_or_path": "microsoft/resnet-50",
            "cache_dir": "./model_cache",
            "model_revision": "main",
            "token": None,
            "trust_remote_code": False,
            "ignore_mismatched_sizes": False,
            "image_processor_name": "microsoft/resnet-50",
        }
        
        data_args = {
            "dataset_name": "cifar10",
            "image_column_name": "img",
        }
        
        training_args = {
            "do_train": True,
        }
        
        pipeline = ImageProcessingPipeline(model_args, data_args, training_args)
        train_dataset, test_datset, eval_dataset=pipeline.run_pipeline()
        print(train_dataset, test_datset, eval_dataset)
        """

    def _load_config(self) -> AutoConfig:
        config_kwargs = {
            "cache_dir": self.model_args.get("cache_dir"),
            "revision": self.model_args.get("model_revision"),
            "token": self.model_args.get("token"),
            "trust_remote_code": self.model_args.get("trust_remote_code"),
        }
        if "config_name_or_path" in self.model_args:
            config = AutoConfig.from_pretrained(self.model_args["config_name_or_path"], **config_kwargs)
        elif "model_name_or_path" in self.model_args:
            config = AutoConfig.from_pretrained(self.model_args["model_name_or_path"], **config_kwargs)
        else:
            raise ValueError("Config path or model name/path must be specified.")
        return config

    def _load_model(self) -> AutoModelForImageClassification:
        model = AutoModelForImageClassification.from_pretrained(
            self.model_args["model_name_or_path"],
            from_tf=bool(".ckpt" in self.model_args["model_name_or_path"]),
            config=self.config,
            cache_dir=self.model_args.get("cache_dir"),
            revision=self.model_args.get("model_revision"),
            token=self.model_args.get("token"),
            trust_remote_code=self.model_args.get("trust_remote_code"),
            ignore_mismatched_sizes=self.model_args.get("ignore_mismatched_sizes", False),
        )
        return model

    def _load_image_processor(self) -> AutoImageProcessor:
        image_processor = AutoImageProcessor.from_pretrained(
            self.model_args.get("image_processor_name", self.model_args["model_name_or_path"]),
            cache_dir=self.model_args.get("cache_dir"),
            revision=self.model_args.get("model_revision"),
            token=self.model_args.get("token"),
            trust_remote_code=self.model_args.get("trust_remote_code"),
        )
        return image_processor

    def _load_dataset(self) -> DatasetDict:
        if "dataset_name" not in self.data_args:
            raise ValueError("Dataset name must be specified.")
        dataset = prepare_datasetsforhemanth(self.data_args["dataset_name"])
        return dataset

    def _setup_transforms(self) -> Compose:
        if "train" in self.dataset:
            size = self.model_args.get("image_size", 224)
            transforms = Compose([
                Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                RandomResizedCrop(size),
                RandomHorizontalFlip(),
                ToTensor(),
                Normalize(mean=self.image_processor.image_mean, std=self.image_processor.image_std),
            ])
        else:
            size = (self.image_processor.size.get("height"), self.image_processor.size.get("width"))
            transforms = Compose([
                Resize(size),
                CenterCrop(size),
                ToTensor(),
                Normalize(mean=self.image_processor.image_mean, std=self.image_processor.image_std),
            ])
        return transforms

    def preprocess_images(self, examples: Dict) -> Dict:
        examples["pixel_values"] = [self.transforms(image) for image in examples[self.data_args.get("image_column_name", "image")]]
        return examples

    def run_pipeline(self):
        if "train" in list(self.dataset.keys()):
           train_dataset=self.dataset["train"] = self.dataset["train"].map(self.preprocess_images, batched=True)
           logger.info("Training dataset processed.")
        if "test" in list(self.dataset.keys()):
            test_datset=self.dataset["test"] = self.dataset["test"].map(self.preprocess_images, batched=True)
            logger.info("Validation dataset processed.")
        if "eval" in list(self.dataset.keys()):
            eval_dataset=self.dataset["eval"] = self.dataset["eval"].map(self.preprocess_images, batched=True)
            logger.info("Evaluation dataset processed.")
        return train_dataset, test_datset, eval_dataset



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
        

# FIM Functions
def fim_psm(sample: List[int], fim_rate: float) -> List[int]:
    """Applies Probabilistic Substitution Model (PSM) for FIM.

    Args:
        sample: List of tokens.
        fim_rate: Probability of masking a token.

    Returns:
        List of tokens with masking applied.
    """
    return [random.choice(sample) if random.random() < fim_rate else token 
            for token in sample]


def fim_spm(sample: List[int], fim_rate: float, fim_spm_rate: float) -> List[int]:
    """Applies Stochastic Positional Masking (SPM) for FIM.

    Args:
        sample: List of tokens.
        fim_rate: Probability of masking a token.
        fim_spm_rate: Probability of masking a span of tokens.

    Returns:
        List of tokens with masking applied.
    """
    masked_sample = []
    i = 0
    while i < len(sample):
        if random.random() < fim_rate:
            if random.random() < fim_spm_rate:
                span_length = min(random.randint(1, 3), len(sample) - i) 
                masked_sample.extend([random.choice(sample)] * span_length)
                i += span_length
            else:
                masked_sample.append(random.choice(sample))
                i += 1
        else:
            masked_sample.append(sample[i])
            i += 1
    return masked_sample

# Dataset Class
class ConstantLengthDataset:
    """Dataset that chunks text into constant-length samples with optional overlap."""

    def __init__(self, text: str,tokenizer, max_length: int, padding: int, overlap: int = 0,):
        """Initializes the ConstantLengthDataset.

        Args:
            text: The input text.
            max_length: Maximum length of each sample.
            padding: Padding value for shorter samples.
            overlap: Number of tokens to overlap between consecutive samples.
        """
        self.tokens = tokenizer(text, return_tensors='pt')
        self.max_length = max_length
        self.padding = padding
        self.overlap = overlap
        self.effective_length = max_length - overlap

    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        if len(self.tokens) <= self.max_length:
            return 1  
        return (len(self.tokens) - self.overlap) // self.effective_length + (1 if (len(self.tokens) - self.overlap) % self.effective_length > 0 else 0)  

    def __getitem__(self, idx: int) -> List[int]:
        """Returns a single sample from the dataset."""
        start = idx * self.effective_length
        end = min(start + self.max_length, len(self.tokens))  
        sample = self.tokens[start:end]
        if len(sample) < self.max_length:
            sample.extend([self.padding] * (self.max_length - len(sample)))
        return sample

def preprocess_text(
    text: str, 
    tokenizer,
    max_length: int, 
    paddings: int, 
    fim_rate: float = 0.15, 
    fim_spm_rate: float = 0.5,
    overlap: int = 0  ):
    """
    Preprocesses large text data using chunking, FIM, and optional padding.

    Args:
        text (str): The input text.
        max_length (int): Maximum length of each sample after processing.
        paddings (int): Padding value for shorter samples (if applicable).
        fim_rate (float, optional): Probability of masking a token during FIM. Defaults to 0.15.
        fim_spm_rate (float, optional): Probability of masking a span during SPM FIM. Defaults to 0.5.
        overlap (int, optional): Number of tokens to overlap between samples. Defaults to 0.

    Returns:
        Dict[str, List[List[int]]]: A dictionary containing:
            - "input_ids":  A list of processed input samples (chunked and potentially augmented with FIM).
            - "labels": A copy of "input_ids" used for training (assuming a self-supervised setup). 

    """
    dataset = ConstantLengthDataset(text,tokenizer, max_length, paddings, overlap)
    input_ids = []
    for i in range(len(dataset)):
        sample = dataset[i]
        fim_method = random.choices(["psm", "spm", "none"], weights=[0.45, 0.45, 0.1])[0]  # Adjust weights as needed
        if fim_method == "psm":
            processed_sample = fim_psm(sample.copy(), fim_rate)  # Copy to avoid modifying original data
        elif fim_method == "spm":
            processed_sample = fim_spm(sample.copy(), fim_rate, fim_spm_rate) 
        else: 
            processed_sample = sample  # No FIM applied 
        input_ids.append(processed_sample)

    return {"input_ids": torch.tensor(input_ids), "labels": torch.tensor(input_ids.copy())}




# Constants for better readability and maintainability
FIM_MODES = ["PSM", "SPM"]


@functools.lru_cache(maxsize=None)
def get_fim_token_ids(tokenizer: AutoTokenizer) -> Dict[str, Optional[int]]:
    """Retrieves token IDs for FIM special tokens from a tokenizer."""
    fim_token_names = ["FIM_SUFFIX", "FIM_PREFIX", "FIM_MIDDLE", "FIM_PAD"]
    try:
        fim_tokens = tokenizer.special_tokens_map["additional_special_tokens"][1:5]
        return {name: tokenizer.vocab[token] for name, token in zip(fim_token_names, fim_tokens)}
    except KeyError:
        print("Warning: FIM special tokens not found in tokenizer. Disabling FIM.")
        return {name: None for name in fim_token_names}


def permute(
    sample: List[int],
    np_rng: np.random.RandomState,
    fim_token_ids: Dict[str, int],
    fim_rate: float = 0.5,
    fim_mode: str = "PSM",
) -> List[int]:
    """
    Applies FIM transformation to a sample with a given probability.

    Args:
        sample: List of tokens.
        np_rng: NumPy random number generator.
        fim_token_ids: Dictionary containing IDs for FIM special tokens.
        fim_rate: Probability of applying FIM transformation.
        fim_mode: FIM mode to use ("PSM" or "SPM").

    Returns:
        Transformed sample (list of tokens).
    """
    if np_rng.binomial(1, fim_rate):
        boundaries = sorted(np_rng.randint(low=0, high=len(sample) + 1, size=2))
        prefix, middle, suffix = (
            sample[: boundaries[0]],
            sample[boundaries[0] : boundaries[1]],
            sample[boundaries[1] :],
        )

        if fim_mode == "SPM":
            new_sample = (
                [fim_token_ids["FIM_PREFIX"], fim_token_ids["FIM_SUFFIX"]]
                + suffix
                + [fim_token_ids["FIM_MIDDLE"]]
                + prefix
                + middle
            )
        elif fim_mode == "PSM":
            new_sample = (
                [fim_token_ids["FIM_PREFIX"]]
                + prefix
                + [fim_token_ids["FIM_SUFFIX"]]
                + suffix
                + [fim_token_ids["FIM_MIDDLE"]]
                + middle
            )
        else:
            raise ValueError(f"Invalid FIM mode: {fim_mode}")

        return new_sample
    else:
        return sample


class ConstantLengthDataset_iter(IterableDataset):
    """
    Dataset that generates constant-length sequences from text data.

    This dataset reads text data, tokenizes it, applies FIM transformations,
    and yields batches of constant-length sequences.
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        dataset: Dataset,
        infinite: bool = False,
        seq_length: int = 1024,
        buffer_size_gb: float = 1.0,
        chars_per_token: float = 3.6,
        content_field: str = "content",
        fim_rate: float = 0.5,
        fim_mode: str = "PSM",
        seed: int = 0,
    ):
        """
        Initializes the ConstantLengthDataset.

        Args:
            tokenizer: Tokenizer to use for text processing.
            dataset: Hugging Face Dataset object containing the text data.
            infinite: Whether to iterate over the dataset infinitely.
            seq_length: Length of the output sequences.
            buffer_size_gb: Size of the buffer in gigabytes for accumulating text.
            chars_per_token: Average number of characters per token.
            content_field: Name of the field in the dataset containing the text.
            fim_rate: Probability of applying FIM transformation.
            fim_mode: FIM mode to use ("PSM" or "SPM").
            seed: Random seed for reproducibility.
        """
        self.tokenizer = tokenizer
        self.concat_token_id = tokenizer.eos_token_id
        self.dataset = dataset
        self.seq_length = seq_length
        self.infinite = infinite
        self.current_size = 0
        self.buffer_size_bytes = int(buffer_size_gb * 1024**3)
        self.chars_per_token = chars_per_token
        self.content_field = content_field
        self.fim_rate = fim_rate
        self.fim_mode = fim_mode
        self.seed = seed
        self.fim_token_ids = get_fim_token_ids(self.tokenizer)
        self.np_rng = np.random.RandomState(seed=self.seed)

    def __len__(self):
        """Returns the approximate length of the dataset."""
        if self.infinite:
            return 100000  # Return a large constant value for infinite dataset
        else:
            total_chars = sum(len(str(example[self.content_field])) for example in self.dataset)
            estimated_tokens = total_chars / self.chars_per_token
            estimated_sequences = estimated_tokens // self.seq_length
            return int(estimated_sequences)

    def __iter__(self):
        """Iterates over the dataset and yields batches of sequences."""
        iterator = iter(self.dataset)
        more_examples = True

        while more_examples:
            buffer, buffer_len = [], 0
            while True:
                if buffer_len >= self.buffer_size_bytes:
                    break
                try:
                    example = next(iterator)
                    text = str(example[self.content_field])
                    buffer.append(text)
                    buffer_len += len(text.encode())  # Calculate buffer length in bytes
                except StopIteration:
                    if self.infinite:
                        iterator = iter(self.dataset)
                    else:
                        more_examples = False
                        break

            # Tokenize the accumulated text in the buffer
            tokenized_inputs = self.tokenizer(
                buffer,
                truncation=True,
                max_length=self.seq_length,
                return_overflowing_tokens=True,
            )
            
            all_token_ids = []
            for tokenized_input in tokenized_inputs["input_ids"]:
                if self.fim_rate > 0:
                    tokenized_input = permute(
                        tokenized_input,
                        self.np_rng,
                        self.fim_token_ids,
                        fim_rate=self.fim_rate,
                        fim_mode=self.fim_mode,
                    )
                all_token_ids.extend(tokenized_input + [self.concat_token_id])

            examples = []
            for i in range(0, len(all_token_ids), self.seq_length):
                input_ids = all_token_ids[i : i + self.seq_length]
                if len(input_ids) == self.seq_length:
                    examples.append(input_ids)

            # Shuffle the examples within the buffer to improve randomness
            random.shuffle(examples)

            for example in examples:
                self.current_size += 1
                yield {
                    "input_ids": torch.LongTensor(example),
                    "labels": torch.LongTensor(example),
                }



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
        # Load dataset
        dataset = self._load_dataset()
        
        # Combine all columns into a single 'text' column
        dataset = self._combine_columns(dataset)
        
        # Split dataset
        train_dataset, test_dataset, eval_dataset = self._split_dataset(dataset)
        
        # Tokenize datasets
        train_dataset = self._tokenize_dataset(train_dataset)
        test_dataset = self._tokenize_dataset(test_dataset)
        eval_dataset = self._tokenize_dataset(eval_dataset)
        
        # Create data loaders
        train_loader = self._create_dataloader(train_dataset)
        test_loader = self._create_dataloader(test_dataset)
        eval_loader = self._create_dataloader(eval_dataset)
        
        return train_loader, test_loader, eval_loader

    def _load_dataset(self) -> Dataset:
        logger.info(f"Loading dataset from {self.dataset_path}")
        try:
            dataset = load_dataset(self.dataset_path)
            return dataset[list(dataset.keys())[0]]  # Assuming the dataset has a 'train' split
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