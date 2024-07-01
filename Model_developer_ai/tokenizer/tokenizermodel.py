import logging
import multiprocessing
import os
import re
import time
from functools import lru_cache
from typing import List, Optional, Iterator, Dict, Tuple

import sentencepiece as spm
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors
from tqdm import tqdm
from transformers import AutoTokenizer


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class AdvancedTokenizer_v3:
    """
    An advanced tokenizer class that supports BPE and Unigram tokenization with 
    features like custom tokenization rules, vocabulary pruning, and benchmarking.

    Args:
        vocab_size (int, optional): The size of the vocabulary. Defaults to 30000.
        min_frequency (int, optional): The minimum frequency of a token to be included 
                                       in the vocabulary. Defaults to 2.
        model_type (str, optional): The type of tokenization model. Must be either 
                                     "bpe" or "unigram". Defaults to "bpe".
        special_tokens (Optional[List[str]], optional): A list of special tokens 
                                                         to be added to the vocabulary. 
                                                         Defaults to ["[UNK]", "[CLS]", 
                                                         "[SEP]", "[PAD]", "[MASK]"].
    """

    def __init__(
        self,
        vocab_size: int = 30000,
        min_frequency: int = 2,
        model_type: str = "bpe",
        special_tokens: Optional[List[str]] = None,
    ):
        if model_type not in ["bpe", "unigram"]:
            raise ValueError("Invalid model_type. Must be 'bpe' or 'unigram'.")

        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.model_type = model_type
        self.special_tokens = special_tokens or ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
        self.tokenizer: Optional[Tokenizer] = None  # For BPE
        self.sp_model: Optional[spm.SentencePieceProcessor] = None  # For Unigram

        self.custom_rules: List[Tuple[re.Pattern, str]] = []  # Store custom rules

    @staticmethod
    def parallel_file_reader(file_paths: List[str], chunk_size: int = 1000) -> Iterator[str]:
        """
        Reads multiple files in parallel and yields lines.

        Args:
            file_paths (List[str]): A list of file paths to read.
            chunk_size (int, optional): The number of lines to read at a time. 
                                        Defaults to 1000.

        Yields:
            Iterator[str]: An iterator over the lines of the files.
        """

        def process_file(file_path: str) -> List[str]:
            """Reads a single file and returns a list of lines."""
            with open(file_path, "r", encoding="utf-8") as f:
                return f.readlines(chunk_size)

        with multiprocessing.Pool() as pool:
            for chunk in pool.imap_unordered(process_file, file_paths):
                yield from chunk

    def train_from_files(self, file_paths: List[str]) -> None:
        """
        Trains the tokenizer on a list of files.

        Args:
            file_paths (List[str]): A list of file paths to train on.
        """

        try:
            if self.model_type == "bpe":
                self._train_bpe(self.parallel_file_reader(file_paths))
            elif self.model_type == "unigram":
                self._train_unigram(file_paths)  # Unigram uses a different training method
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise

    @lru_cache(maxsize=10000)  # Cache encoded results for faster processing
    def cached_encode(self, text: str) -> Optional[List[int]]:
        """Encodes a text using the cached tokenizer."""
        return self.encode(text)

    def _train_bpe(self, text_iterator: Iterator[str]) -> None:
        """
        Trains a BPE tokenizer.

        Args:
            text_iterator (Iterator[str]): An iterator over the text to train on.
        """
        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)

        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=self.min_frequency,
            special_tokens=self.special_tokens,
        )

        tokenizer.train_from_iterator(text_iterator, trainer=trainer)

        tokenizer.post_processor = processors.TemplateProcessing(
            single="[CLS] $A [SEP]",
            pair="[CLS] $A [SEP] $B:1 [SEP]:1",
            special_tokens=[("[CLS]", 1), ("[SEP]", 2)],
        )

        self.tokenizer = tokenizer
        logger.info("BPE Tokenizer training completed.")

    def _train_unigram(self, file_paths: List[str]) -> None:
        """
        Trains a Unigram tokenizer.

        Args:
            file_paths (List[str]): A list of file paths to train on.
        """

        input_file = self._merge_files(file_paths)
        model_prefix = "unigram_model"

        spm.SentencePieceTrainer.train(
            input=input_file,
            model_prefix=model_prefix,
            vocab_size=self.vocab_size,
            model_type="unigram",
            character_coverage=1.0,
            user_defined_symbols=self.special_tokens,
        )

        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.load(f"{model_prefix}.model")

        # Clean up temporary files
        os.remove(input_file)
        os.remove(f"{model_prefix}.model")
        os.remove(f"{model_prefix}.vocab")
        logger.info("Unigram Tokenizer training completed.")

    @staticmethod
    def _merge_files(file_paths: List[str]) -> str:
        """Merges multiple files into a single temporary file for Unigram training."""
        merged_file = "merged_input.txt"
        with open(merged_file, "w", encoding="utf-8") as outfile:
            for file_path in tqdm(file_paths, desc="Merging files"):
                with open(file_path, "r", encoding="utf-8") as infile:
                    outfile.write(infile.read() + "\n")
        return merged_file

    def save_model(self, path: str) -> None:
        """
        Saves the trained tokenizer model to a file.

        Args:
            path (str): The path to save the model to.
        """
        try:
            if self.tokenizer:
                self.tokenizer.save(path)
            elif self.sp_model:
                self.sp_model.save(path)
            else:
                raise ValueError("No model trained to save.")
            logger.info(f"Model saved successfully at {path}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

    def prune_vocabulary(self, min_frequency: int) -> None:
        """
        Prunes the vocabulary by removing tokens that appear less frequently
        than min_frequency. This method works for both BPE and Unigram models.
        """
        if self.tokenizer:  # For BPE model
            vocab: Dict[str, int] = self.tokenizer.get_vocab()
            pruned_vocab: Dict[str, int] = {
                token: count
                for token, count in vocab.items()
                if count >= min_frequency or token in self.special_tokens
            }

            # Create a new tokenizer with the pruned vocabulary
            new_tokenizer = Tokenizer(models.BPE())
            new_tokenizer.pre_tokenizer = self.tokenizer.pre_tokenizer
            new_tokenizer.post_processor = self.tokenizer.post_processor

            # Add the pruned vocabulary to the new tokenizer
            for token, _ in pruned_vocab.items():
                new_tokenizer.add_tokens([token])

            self.tokenizer = new_tokenizer
            logger.info(f"Vocabulary pruned. New size: {len(pruned_vocab)}")

        elif self.sp_model:  # For Unigram model
            # SentencePiece doesn't support direct vocabulary pruning.
            # We need to retrain the model with a new vocabulary.
            vocab: List[Tuple[str, int]] = self.sp_model.get_piece_size()
            pruned_vocab: List[str] = [
                token
                for token, count in vocab
                if count >= min_frequency or token in self.special_tokens
            ]

            # Save the pruned vocabulary to a temporary file
            temp_vocab_file = "temp_pruned_vocab.txt"
            with open(temp_vocab_file, "w", encoding="utf-8") as f:
                for token in pruned_vocab:
                    f.write(f"{token}\n")

            # Retrain the model with the pruned vocabulary
            model_prefix = "pruned_unigram_model"
            spm.SentencePieceTrainer.train(
                vocab_size=len(pruned_vocab),
                model_prefix=model_prefix,
                model_type="unigram",
                input_vocabulary=temp_vocab_file,
                user_defined_symbols=self.special_tokens,
            )

            self.sp_model = spm.SentencePieceProcessor()
            self.sp_model.load(f"{model_prefix}.model")

            # Clean up temporary files
            os.remove(temp_vocab_file)
            os.remove(f"{model_prefix}.model")
            os.remove(f"{model_prefix}.vocab")

            logger.info(f"Vocabulary pruned. New size: {len(pruned_vocab)}")

        else:
            raise ValueError("No model loaded.")

    def add_custom_tokenization_rule(self, pattern: str, replacement: str) -> None:
        """
        Adds a custom tokenization rule using regex pattern matching.
        This method works as a pre-processing step before the main tokenization.

        Args:
            pattern (str): The regex pattern to match.
            replacement (str): The replacement string for the matched pattern.
        """
        try:
            compiled_pattern = re.compile(pattern)
            self.custom_rules.append((compiled_pattern, replacement))
            logger.info(f"Custom tokenization rule added: {pattern} -> {replacement}")
        except re.error as e:
            logger.error(f"Invalid regex pattern: {e}")
            raise ValueError(f"Invalid regex pattern: {e}")

    def _apply_custom_rules(self, text: str) -> str:
        """Applies all custom tokenization rules to the input text."""
        for pattern, replacement in self.custom_rules:
            text = pattern.sub(replacement, text)
        return text

    def encode(self, text: str) -> Optional[List[int]]:
        """
        Encodes a text string into a list of token IDs.

        Args:
            text (str): The text to encode.

        Returns:
            Optional[List[int]]: A list of token IDs, or None if encoding fails.
        """
        try:
            text = self._apply_custom_rules(text)  # Apply custom rules before encoding

            if self.tokenizer:
                return self.tokenizer.encode(text).ids
            elif self.sp_model:
                return self.sp_model.encode_as_ids(text)
            else:
                raise ValueError("No model loaded.")
        except Exception as e:
            logger.error(f"Error during encoding: {str(e)}")
            return None

    def decode(self, tokens: List[int]) -> Optional[str]:
        """
        Decodes a list of token IDs back into a text string.

        Args:
            tokens (List[int]): A list of token IDs.

        Returns:
            Optional[str]: The decoded text string, or None if decoding fails.
        """
        try:
            if self.tokenizer:
                return self.tokenizer.decode(tokens)
            elif self.sp_model:
                return self.sp_model.decode_ids(tokens)
            else:
                raise ValueError("No model loaded.")
        except Exception as e:
            logger.error(f"Error during decoding: {str(e)}")
            return None

    @staticmethod
    def get_pretrained_tokenizer(model_name: str) -> AutoTokenizer:
        """
        Loads a pretrained tokenizer from the Hugging Face Model Hub.

        Args:
            model_name (str): The name of the pretrained model.

        Returns:
            AutoTokenizer: The pretrained tokenizer.
        """
        try:
            return AutoTokenizer.from_pretrained(model_name)
        except Exception as e:
            logger.error(f"Error loading pretrained tokenizer: {str(e)}")
            raise

    def benchmark_tokenization(self, text: str, iterations: int = 1000) -> float:
        """
        Benchmarks the tokenization speed.

        Args:
            text (str): The text to tokenize.
            iterations (int): The number of iterations to run.

        Returns:
            float: The average time per tokenization in seconds.
        """
        start_time = time.time()
        for _ in range(iterations):
            self.encode(text)
        end_time = time.time()

        total_time = end_time - start_time
        avg_time = total_time / iterations

        # Calculate and log tokenization speed metrics
        tokens_per_second = len(text.split()) / avg_time
        chars_per_second = len(text) / avg_time
        logger.info(f"Tokenization benchmark results:")
        logger.info(f"  Total time: {total_time:.4f} seconds")
        logger.info(f"  Average time per tokenization: {avg_time:.6f} seconds")
        logger.info(f"  Tokens per second: {tokens_per_second:.2f}")
        logger.info(f"  Characters per second: {chars_per_second:.2f}")

        return avg_time

    def load_model(self, path: str) -> None:
        """
        Loads a trained tokenizer model from a file.

        Args:
            path (str): The path to the saved model file.
        """
        try:
            if self.model_type == "bpe":
                self.tokenizer = Tokenizer.from_file(path)
            elif self.model_type == "unigram":
                self.sp_model = spm.SentencePieceProcessor()
                self.sp_model.load(path)
            else:
                raise ValueError(
                    f"Unsupported model type: {self.model_type}. Cannot load model."
                )
            logger.info(f"Model loaded successfully from {path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise


def get_file_paths(folder_path: str, file_extensions: List[str]) -> Iterator[str]:
    """
    Gets a list of file paths from a folder, recursively,
    filtering by file extension.

    Args:
        folder_path (str): The path to the folder.
        file_extensions (List[str]): A list of file extensions to filter by.

    Yields:
        Iterator[str]: An iterator over the file paths.
    """
    for root, _, files in os.walk(folder_path):
        for file in files:
            if any(file.endswith(ext) for ext in file_extensions):
                yield os.path.join(root, file)


if __name__ == "__main__":
    # Example usage
    folder_path = r"E:\LLMS\Fine-tuning\data\data"  # Replace with your folder path
    file_extensions = [".txt", ".md", ".py"]
    file_paths = list(get_file_paths(folder_path, file_extensions))

    # --- Training ---
    # Initialize and train the tokenizer
    tokenizer = AdvancedTokenizer_v3(
        vocab_size=50000, min_frequency=2, model_type="bpe"
    )
    tokenizer.train_from_files(file_paths)
    tokenizer.save_model("advanced_tokenizer.json")

    # --- Loading ---
    # Load the saved tokenizer
    tokenizer.load_model("advanced_tokenizer.json")

    # --- Basic Encoding/Decoding ---
    sample_text = "This is a test of the advanced tokenizer model."
    encoded = tokenizer.encode(sample_text)
    if encoded:
        print("Encoded:", encoded)
        decoded = tokenizer.decode(encoded)
        if decoded:
            print("Decoded:", decoded)

    # --- Pretrained Tokenizer ---
    gpt2_tokenizer = AdvancedTokenizer_v3.get_pretrained_tokenizer("gpt2")
    gpt2_encoded = gpt2_tokenizer.encode(sample_text)
    print("GPT-2 Encoded:", gpt2_encoded)
    print("GPT-2 Decoded:", gpt2_tokenizer.decode(gpt2_encoded))

    # --- Pruning ---
    tokenizer.prune_vocabulary(min_frequency=5)

    # --- Custom Rules ---
    # Add custom tokenization rules
    tokenizer.add_custom_tokenization_rule(r"\b(https?:\/\/\S+)", "[URL]")
    tokenizer.add_custom_tokenization_rule(
        r"\b([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,})\b", "[EMAIL]"
    )
    # Test custom rules
    custom_text = "Check out https://example.com or email john@example.com for more info."
    encoded_custom = tokenizer.encode(custom_text)
    decoded_custom = tokenizer.decode(encoded_custom)
    print("Custom rules applied:")
    print("Original:", custom_text)
    print("Encoded:", encoded_custom)
    print("Decoded:", decoded_custom)

    # --- Benchmarking ---
    benchmark_text = "This is a benchmark test for tokenization speed. " * 100
    avg_time = tokenizer.benchmark_tokenization(benchmark_text)
    print(f"Average tokenization time: {avg_time:.6f} seconds")