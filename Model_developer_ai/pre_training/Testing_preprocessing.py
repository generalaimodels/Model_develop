import functools
import random
import numpy as np
import torch

from torch.utils.data import IterableDataset
from transformers import AutoTokenizer
from typing import Dict, Any, List, Union, Optional,Tuple
from torch.utils.data import Dataset
from itertools import cycle

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


def permute(
    sample: List[int],
    np_rng: np.random.RandomState,
    suffix_tok_id: int,
    prefix_tok_id: int,
    middle_tok_id: int,
    pad_tok_id: int,
    fim_rate: float = 0.5,
    fim_spm_rate: float = 0.5,
) -> Tuple[List[int], np.random.RandomState]:
    """
    Perform a FIM transformation on the sample with a probability of fim_rate.
    """
    if np_rng.binomial(1, fim_rate):
        boundaries = sorted(np_rng.randint(low=0, high=len(sample) + 1, size=2))

        prefix = np.array(sample[: boundaries[0]], dtype=np.int64)
        middle = np.array(sample[boundaries[0]: boundaries[1]], dtype=np.int64)
        suffix = np.array(sample[boundaries[1]:], dtype=np.int64)

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
        new_sample = sample

    return list(new_sample), np_rng


class ConstantLengthDataset(IterableDataset):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        dataset: Dataset,
        infinite: bool = False,
        seq_length: int = 1024,
        num_of_sequences: int = 1024,
        chars_per_token: float = 3.6,
        fim_rate: float = 0.5,
        fim_spm_rate: float = 0.5,
        seed: int = 0,
    ):
        self.tokenizer = tokenizer
        self.concat_token_id = tokenizer.eos_token_id
        self.dataset = dataset
        self.seq_length = seq_length
        self.infinite = infinite
        self.max_buffer_size = int(seq_length * chars_per_token * num_of_sequences)
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
            while buffer_len < self.max_buffer_size:
                try:
                    example = next(iterator)
                    text = " ".join(str(example[col]) for col in self.dataset.column_names)
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
                    )
                all_token_ids.extend(tokenized_input + [self.concat_token_id])

            examples = [
                all_token_ids[i: i + self.seq_length]
                for i in range(0, len(all_token_ids), self.seq_length)
                if len(all_token_ids[i: i + self.seq_length]) == self.seq_length
            ]
            random.shuffle(examples)

            for example in examples:
                yield {
                    "input_ids": torch.LongTensor(example),
                    "labels": torch.LongTensor(example),
                }
                
                


# def get_fim_token_ids(tokenizer):
#     fim_tokens = tokenizer.special_tokens_map["additional_special_tokens"][1:5]
#     token_ids = defaultdict(lambda: None)
#     for tok, tok_id in zip(fim_tokens, ["FIM_SUFFIX", "FIM_PREFIX", "FIM_MIDDLE", "FIM_PAD"]):
#         token_ids[tok_id] = tokenizer.vocab.get(tok, None)
#     return token_ids["FIM_SUFFIX"], token_ids["FIM_PREFIX"], token_ids["FIM_MIDDLE"], token_ids["FIM_PAD"]


def permute1(
    sample: np.ndarray,
    np_rng: np.random.RandomState,
    suffix_tok_id: int,
    prefix_tok_id: int,
    middle_tok_id: int,
    pad_tok_id: int,
    fim_rate: float = 0.5,
    fim_spm_rate: float = 0.5,
) -> Tuple[np.ndarray, np.random.RandomState]:
    """
    Perform a FIM transformation on the sample with a probability of fim_rate.
    """
    if np_rng.binomial(1, fim_rate):
        boundaries = sorted(np_rng.randint(low=0, high=len(sample) + 1, size=2))

        prefix = sample[: boundaries[0]]
        middle = sample[boundaries[0]: boundaries[1]]
        suffix = sample[boundaries[1]:]

        if np_rng.binomial(1, fim_spm_rate):
            new_sample = np.concatenate([
                [prefix_tok_id, suffix_tok_id],
                suffix,
                [middle_tok_id],
                prefix,
                middle,
            ])
        else:
            new_sample = np.concatenate([
                [prefix_tok_id],
                prefix,
                [suffix_tok_id],
                suffix,
                [middle_tok_id],
                middle,
            ])
    else:
        new_sample = sample

    return new_sample, np_rng


class ConstantLengthDataset1(IterableDataset):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        dataset: Dataset,
        infinite: bool = False,
        seq_length: int = 1024,
        num_of_sequences: int = 1024,
        chars_per_token: float = 3.6,
        fim_rate: float = 0.5,
        fim_spm_rate: float = 0.5,
        seed: int = 0,
    ):
        self.tokenizer = tokenizer
        self.concat_token_id = tokenizer.eos_token_id
        self.dataset = dataset
        self.seq_length = seq_length
        self.infinite = infinite
        self.max_buffer_size = int(seq_length * chars_per_token * num_of_sequences)
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
        iterator = cycle(self.dataset) if self.infinite else iter(self.dataset)
        np_rng = np.random.RandomState(seed=self.seed)

        while True:
            buffer, buffer_len = [], 0
            while buffer_len < self.max_buffer_size:
                try:
                    example = next(iterator)
                    text = " ".join(str(example[col]) for col in self.dataset.column_names)
                    buffer.append(text)
                    buffer_len += len(buffer[-1])
                except StopIteration:
                    break

            tokenized_inputs = self.tokenizer(buffer, truncation=False)["input_ids"]
            all_token_ids = np.concatenate([
                np.array(tokenized_input + [self.concat_token_id])
                for tokenized_input in tokenized_inputs
            ])

            if self.fim_rate > 0:
                all_token_ids, np_rng = permute1(
                    all_token_ids,
                    np_rng,
                    self.suffix_tok_id,
                    self.prefix_tok_id,
                    self.middle_tok_id,
                    self.pad_tok_id,
                    fim_rate=self.fim_rate,
                    fim_spm_rate=self.fim_spm_rate,
                )

            examples = np.array([
                all_token_ids[i: i + self.seq_length]
                for i in range(0, len(all_token_ids), self.seq_length)
                if len(all_token_ids[i: i + self.seq_length]) == self.seq_length
            ])
            examples = examples[np.random.permutation(len(examples))]

            yield from (
                {
                    "input_ids": torch.tensor(example, dtype=torch.long),
                    "labels": torch.tensor(example, dtype=torch.long),
                }
                for example in examples
            )

            if not self.infinite:
                break