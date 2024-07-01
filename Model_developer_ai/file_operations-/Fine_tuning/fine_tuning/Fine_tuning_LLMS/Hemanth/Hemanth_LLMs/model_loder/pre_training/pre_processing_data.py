# 🔒 Confidentiality Notice 
#
# DARFA-LLMs, is developed under the auspices of CDAC (Center for Development of Advanced Computing). 
# The project's codebase, algorithms, and any associated documentation are considered confidential 
# and proprietary to CDACB. 
#
# Copyright Statement ©️
#
# All code within this project is the intellectual property of Kandimalla Hemanth. 👨‍💻 Any reproduction, 
# distribution, or unauthorized use of the code or its components without explicit permission from the
# copyright holder is strictly prohibited. 
#
# Project Purpose 
#
# The DARFA-llms project is dedicated to advancing human development through the application of artificial 
# intelligence (AI) technologies. Its mission is to address real-world challenges and contribute 
# to the betterment of society. 
#
#  Non-Disclosure Agreement (NDA) 
#
# Access to this Python project and its associated materials is restricted to authorized personnel only.
# 🔒 By accessing or using any part of this project, you agree to maintain strict confidentiality 
# and adhere to any applicable non-disclosure agreements (NDAs) or confidentiality policies established
# by CDACB. 
#
# ⚠️ Disclaimer of Liability 📜
#
# While every effort has been made to ensure the accuracy and reliability of the code and information 
# contained within this project, CDACB and Kandimalla Hemanth shall not be held liable for any direct, 
# indirect, incidental, special, or consequential damages arising out of the use or inability to use the
# project, even if advised of the possibility of such damages. 
#
# 🚫 Usage Restrictions 📝
#
# The code and algorithms developed as part of the DARFA project are intended for research, educational,
# and non-commercial purposes only. Any commercial use, modification, or adaptation of the codebase 
# requires explicit written consent from CDACB and Kandimalla Hemanth.
#
#  Ethical Considerations 
#
# The DARFA project adheres to ethical principles and guidelines governing the responsible 
# use of AI technologies. All efforts are made to ensure that the project's objectives 
# align with ethical standards and promote positive outcomes for humanity. 
import torch
import functools
import numpy as np
import random
import torch
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Any, List,Union,Optional
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import IterableDataset
from torch.utils.data.dataloader import DataLoader
from transformers import AutoTokenizer, default_data_collator,AutoModelForCausalLM


# Helper function to get token ids of the special tokens for prefix, suffix and middle for FIM transformations.
@functools.lru_cache(maxsize=None)
def get_fim_token_ids(tokenizer):
    try:
        FIM_PREFIX, FIM_MIDDLE, FIM_SUFFIX, FIM_PAD = tokenizer.special_tokens_map["additional_special_tokens"][1:5]
        suffix_tok_id, prefix_tok_id, middle_tok_id, pad_tok_id = (
            tokenizer.vocab[tok] for tok in [FIM_SUFFIX, FIM_PREFIX, FIM_MIDDLE, FIM_PAD]
        )
    except KeyError:
        suffix_tok_id, prefix_tok_id, middle_tok_id, pad_tok_id = None, None, None, None
    return suffix_tok_id, prefix_tok_id, middle_tok_id, pad_tok_id


## Adapted from https://github.com/bigcode-project/Megatron-LM/blob/6c4bf908df8fd86b4977f54bf5b8bd4b521003d1/megatron/data/gpt_dataset.py
def permute(
    sample,
    np_rng,
    suffix_tok_id,
    prefix_tok_id,
    middle_tok_id,
    pad_tok_id,
    fim_rate=0.5,
    fim_spm_rate=0.5,
    truncate_or_pad=False,
    ):
    """
    Take in a sample (list of tokens) and perform a FIM transformation on it with a probability of fim_rate, using two FIM modes:
    PSM and SPM (with a probability of fim_spm_rate).
    """

    # The if condition will trigger with the probability of fim_rate
    # This means FIM transformations will apply to samples with a probability of fim_rate
    if np_rng.binomial(1, fim_rate):

        # Split the sample into prefix, middle, and suffix, based on randomly generated indices stored in the boundaries list.
        boundaries = list(np_rng.randint(low=0, high=len(sample) + 1, size=2))
        boundaries.sort()

        prefix = np.array(sample[: boundaries[0]], dtype=np.int64)
        middle = np.array(sample[boundaries[0] : boundaries[1]], dtype=np.int64)
        suffix = np.array(sample[boundaries[1] :], dtype=np.int64)

        if truncate_or_pad:
            # calculate the new total length of the sample, taking into account tokens indicating prefix, middle, and suffix
            new_length = suffix.shape[0] + prefix.shape[0] + middle.shape[0] + 3
            diff = new_length - len(sample)

            # trancate or pad if there's a difference in length between the new length and the original
            if diff > 0:
                if suffix.shape[0] <= diff:
                    return sample, np_rng
                suffix = suffix[: suffix.shape[0] - diff]
            elif diff < 0:
                suffix = np.concatenate([suffix, np.full((-1 * diff), pad_tok_id)])

        # With the probability of fim_spm_rateapply SPM variant of FIM transformations
        # SPM: suffix, prefix, middle
        if np_rng.binomial(1, fim_spm_rate):
            new_sample = np.concatenate(
                [
                    [prefix_tok_id, suffix_tok_id],
                    suffix,
                    [middle_tok_id],
                    prefix,
                    middle,
                ]
            )
        # Otherwise, apply the PSM variant of FIM transformations
        # PSM: prefix, suffix, middle
        else:

            new_sample = np.concatenate(
                [
                    [prefix_tok_id],
                    prefix,
                    [suffix_tok_id],
                    suffix,
                    [middle_tok_id],
                    middle,
                ]
            )
    else:
        # don't apply FIM transformations
        new_sample = sample

    return list(new_sample), np_rng

# Create an Iterable dataset that returns constant-length chunks of tokens from a stream of text files.
class ConstantLengthDataset(IterableDataset):
    """
    Iterable dataset that returns constant length chunks of tokens from stream of text files.
        Args:
            tokenizer (Tokenizer): The processor used for proccessing the data.
            dataset (dataset.Dataset): Dataset with text files.
            infinite (bool): If True the iterator is reset after dataset reaches end else stops.
            seq_length (int): Length of token sequences to return.
            num_of_sequences (int): Number of token sequences to keep in buffer.
            chars_per_token (int): Number of characters per token used to estimate number of tokens in text buffer.
            fim_rate (float): Rate (0.0 to 1.0) that sample will be permuted with FIM.
            fim_spm_rate (float): Rate (0.0 to 1.0) of FIM permuations that will use SPM.
            seed (int): Seed for random number generator.
            
     Returns:
     # # Create datasets for each column
     # train_datasets = {}
     # eval_datasets = {}
     # for column in columns:
     #     train_datasets[column] = ConstantLengthDataset(
     #         tokenizer,
     #         train_data,
     #         infinite=True,
     #         seq_length=SEQ_LENGTH,
     #         chars_per_token=chars_per_token_dict[column],
     #         content_field=column,
     #         fim_rate=FIM_RATE,
     #         fim_spm_rate=FIM_SPM_RATE,
     #         seed=SEED,
     #     )
     #     eval_datasets[column] = ConstantLengthDataset(
     #         tokenizer,
     #         valid_data,
     #         infinite=False,
     #         seq_length=SEQ_LENGTH,
     #         chars_per_token=chars_per_token_dict[column],
     #         content_field=column,
     #         fim_rate=FIM_RATE,
     #         fim_spm_rate=FIM_SPM_RATE,
     #         seed=SEED,
     #     )
    """

    def __init__(
        self,
        tokenizer,
        dataset,
        infinite=False,
        seq_length=1024,
        num_of_sequences=1024,
        chars_per_token=3.6,
        content_field="content",
        fim_rate=0.5,
        fim_spm_rate=0.5,
        seed=0,
    ):
        self.tokenizer = tokenizer
        self.concat_token_id = tokenizer.eos_token_id
        self.dataset = dataset
        self.seq_length = seq_length
        self.infinite = infinite
        self.current_size = 0
        self.max_buffer_size = seq_length * chars_per_token * num_of_sequences
        self.content_field = content_field
        self.fim_rate = fim_rate
        self.fim_spm_rate = fim_spm_rate
        self.seed = seed

        (
            self.suffix_tok_id,
            self.prefix_tok_id,
            self.middle_tok_id,
            self.pad_tok_id,
        ) = get_fim_token_ids(self.tokenizer)
        if not self.suffix_tok_id and self.fim_rate > 0:
            print("FIM is not supported by tokenizer, disabling FIM")
            self.fim_rate = 0

   
    def __iter__(self):
       
       iterator = iter(self.dataset)
       more_examples = True
       np_rng = np.random.RandomState(seed=self.seed)
       while more_examples:
        buffer, buffer_len = [], 0
        while True:
            if buffer_len >= self.max_buffer_size:
                break
            try:
                # Concatenate all columns into a single text
                example = next(iterator)
                text = ' '.join([str(example[col]) for col in self.dataset.column_names])
                buffer.append(text)
                buffer_len += len(buffer[-1])
            except StopIteration:
                if self.infinite:
                    iterator = iter(self.dataset)
                else:
                    more_examples = False
                    break
            tokenized_inputs = self.tokenizer(buffer, truncation=False)["input_ids"]
            all_token_ids = []
            for tokenized_input in tokenized_inputs:
                # optionally do FIM permutations
                if self.fim_rate > 0:
                    tokenized_input, np_rng = permute(
                        tokenized_input,
                        np_rng,
                        self.suffix_tok_id,
                        self.prefix_tok_id,
                        self.middle_tok_id,
                        self.pad_tok_id,
                        fim_rate=self.fim_rate,
                        fim_spm_rate=self.fim_spm_rate,
                        truncate_or_pad=False,
                    )
                all_token_ids.extend(tokenized_input + [self.concat_token_id])
            examples = []
            for i in range(0, len(all_token_ids), self.seq_length):
                input_ids = all_token_ids[i : i + self.seq_length]
                if len(input_ids) == self.seq_length:
                    examples.append(input_ids)
            random.shuffle(examples)
            for example in examples:
                self.current_size += 1
                yield {
                    "input_ids": torch.LongTensor(example),
                    "labels": torch.LongTensor(example),
                }





