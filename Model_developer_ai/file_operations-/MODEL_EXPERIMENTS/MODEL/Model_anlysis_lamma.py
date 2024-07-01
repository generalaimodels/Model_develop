import os
from typing import List
from typing_extensions import Literal

import torch
from transformers import AutoTokenizer


def tokenize_and_save(model_name_or_path: str, output_file: str, save_model: bool = True) -> None:
    """
    Tokenizes a pre-trained model's vocabulary and saves it to a file.

    Args:
    - model_name_or_path (str): The path to the pre-trained model or its identifier.
    - output_file (str): The path to the output file where the vocabulary will be saved.
    - save_model (bool, optional): Whether to save the modified tokenizer with a padding token. Defaults to True.

    Returns:
    None
    """
    # Input validation
    if not isinstance(model_name_or_path, str) or not isinstance(output_file, str):
        raise TypeError("Both model_name_or_path and output_file should be strings.")

    if not os.path.exists(model_name_or_path):
        raise FileNotFoundError(f"Model path '{model_name_or_path}' not found.")

    try:
        # Initialize the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        # Set pad_token_id if not available
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
            if save_model:
                tokenizer.save_pretrained(model_name_or_path)

        # Get vocabulary details
        vocab = list(tokenizer.get_vocab().keys())
        vocab_size = len(vocab)

        # Print vocabulary details
        print(f"Vocabulary Details:")
        print(f"Model: {model_name_or_path}")
        print(f"Vocabulary Size: {vocab_size}")
        print(f"Sample Vocabulary:")
        print(", ".join(vocab[:10]))  # Display the first 10 tokens as a sample

        # Save the vocabulary to a file
        save_vocab_to_file(vocab, output_file)

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise


def save_vocab_to_file(vocab: List[str], output_file: str) -> None:
    """
    Saves a vocabulary list to a text file, one token per line.

    Args:
    - vocab (List[str]): The list of tokens to be saved.
    - output_file (str): The path to the output file.

    Returns:
    None
    """
    with open(output_file, "w", encoding="utf-8") as f:
        for token in vocab:
            f.write(token + "\n")


# Example usage:
model_name_or_path = "bert-base-uncased"
output_file = "vocab.txt"
tokenize_and_save(model_name_or_path, output_file)







