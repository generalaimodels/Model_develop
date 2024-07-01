from pathlib import Path
from typing import List

import sentencepiece as spm


class Tokenizer:
    def __init__(self, model_path: str):
        assert Path(model_path).exists(), model_path
        self._model = spm.SentencePieceProcessor(model_file=model_path)
        assert self._model.vocab_size() == self._model.get_piece_size()

    @property
    def n_words(self) -> int:
        return self._model.vocab_size()

    @property
    def bos_id(self) -> int:
        return self._model.bos_id()

    @property
    def eos_id(self) -> int:
        return self._model.eos_id()

    @property
    def pad_id(self) -> int:
        return self._model.pad_id()

    def encode(self, text: str, bos: bool = True) -> List[int]:
        assert isinstance(text, str)
        tokens = self._model.encode(text)
        if bos:
            tokens = [self.bos_id] + tokens
        return tokens

    def decode(self, tokens: List[int]) -> str:
        return self._model.decode(tokens)


def train_tokenizer(input_dir: Path, model_prefix: str, vocab_size: int) -> None:
    assert input_dir.exists() and input_dir.is_dir(), input_dir
    input_files = list(input_dir.glob("**/*.txt"))
    assert input_files, f"No .txt files found in {input_dir}"

    spm.SentencePieceTrainer.train(
        input=",".join(map(str, input_files)),
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=1.0,
        model_type="unigram",
        # input_sentence_size=1e7,
        shuffle_input_sentence=True,
        num_threads=1,  # Set num_threads to a valid value (e.g., 1)
    )

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default="C:/Users/heman/Desktop/deeplearning/output")
    parser.add_argument("--model-prefix", type=str, default="hemanth")
    parser.add_argument("--vocab-size", type=int, default=1024)
    args = parser.parse_args()

    train_tokenizer(args.input_dir, args.model_prefix, args.vocab_size)