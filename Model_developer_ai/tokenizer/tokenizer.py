import sys
from pathlib import Path
file=Path(__file__).resolve()
sys.path.append(str(file.parents[1]))
from typing import List, Union, Optional
import tiktoken
import sentencepiece as spm
import re
import time
import logging
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import sentencepiece as spm
from typing import List, Union, Optional
import os
import time

from typing import( Dict,
                   List,
                   Literal, 
                   Sequence, 
                   Union,
                   AbstractSet, 
                   Collection,
                   Iterator,
                   TypedDict,
                   cast,
                   Optional
)
from tiktoken.load import load_tiktoken_bpe
from tqdm import tqdm
import sentencepiece as spm
from transformers import AutoTokenizer
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvancedTokenizer:
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.model_name = model_name
        self.tokenizers = {}
        self._initialize_tokenizers()

    def _initialize_tokenizers(self) -> None:
        try:
            self.tokenizers["tiktoken"] = tiktoken.encoding_for_model(self.model_name)
            self.tokenizers["sentencepiece"] = spm.SentencePieceProcessor()
            self.tokenizers["sentencepiece"].Load("path/to/sentencepiece/model.model")
            self.tokenizers["huggingface"] = AutoTokenizer.from_pretrained(self.model_name)
        except Exception as e:
            logger.error(f"Error initializing tokenizers: {str(e)}")
            raise

    def encode(self, text: str, tokenizer_type: str = "tiktoken") -> List[int]:
        start_time = time.time()
        try:
            if tokenizer_type == "tiktoken":
                tokens = self.tokenizers["tiktoken"].encode(text)
            elif tokenizer_type == "sentencepiece":
                tokens = self.tokenizers["sentencepiece"].EncodeAsIds(text)
            elif tokenizer_type == "huggingface":
                tokens = self.tokenizers["huggingface"].encode(text)
            else:
                raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")
            
            logger.info(f"Encoding completed in {time.time() - start_time:.4f} seconds")
            return tokens
        except Exception as e:
            logger.error(f"Error during encoding: {str(e)}")
            raise

    def decode(self, tokens: List[int], tokenizer_type: str = "tiktoken") -> str:
        start_time = time.time()
        try:
            if tokenizer_type == "tiktoken":
                text = self.tokenizers["tiktoken"].decode(tokens)
            elif tokenizer_type == "sentencepiece":
                text = self.tokenizers["sentencepiece"].DecodeIds(tokens)
            elif tokenizer_type == "huggingface":
                text = self.tokenizers["huggingface"].decode(tokens)
            else:
                raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")
            
            logger.info(f"Decoding completed in {time.time() - start_time:.4f} seconds")
            return text
        except Exception as e:
            logger.error(f"Error during decoding: {str(e)}")
            raise

    def tokenize(self, text: str, tokenizer_type: str = "tiktoken") -> List[str]:
        start_time = time.time()
        try:
            if tokenizer_type == "tiktoken":
                tokens = [self.tokenizers["tiktoken"].decode([token]) for token in self.encode(text, tokenizer_type)]
            elif tokenizer_type == "sentencepiece":
                tokens = self.tokenizers["sentencepiece"].EncodeAsPieces(text)
            elif tokenizer_type == "huggingface":
                tokens = self.tokenizers["huggingface"].tokenize(text)
            else:
                raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")
            
            logger.info(f"Tokenization completed in {time.time() - start_time:.4f} seconds")
            return tokens
        except Exception as e:
            logger.error(f"Error during tokenization: {str(e)}")
            raise

    def get_vocab_size(self, tokenizer_type: str = "tiktoken") -> int:
        try:
            if tokenizer_type == "tiktoken":
                return self.tokenizers["tiktoken"].n_vocab
            elif tokenizer_type == "sentencepiece":
                return len(self.tokenizers["sentencepiece"])
            elif tokenizer_type == "huggingface":
                return self.tokenizers["huggingface"].vocab_size
            else:
                raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")
        except Exception as e:
            logger.error(f"Error getting vocabulary size: {str(e)}")
            raise

    def batch_encode(self, texts: List[str], tokenizer_type: str = "tiktoken", 
                     max_length: Optional[int] = None) -> List[List[int]]:
        start_time = time.time()
        try:
            batch_tokens = []
            for text in texts:
                tokens = self.encode(text, tokenizer_type)
                if max_length:
                    tokens = tokens[:max_length]
                batch_tokens.append(tokens)
            
            logger.info(f"Batch encoding completed in {time.time() - start_time:.4f} seconds")
            return batch_tokens
        except Exception as e:
            logger.error(f"Error during batch encoding: {str(e)}")
            raise




class AdvancedTokenizer_v2:
    def __init__(self, model_name: str = "advanced_tokenizer"):
        self.model_name = model_name
        self.tiktoken_encoder = tiktoken.get_encoding("gpt2")
        self.tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        self.tokenizer.pre_tokenizer = Whitespace()
        self.sp_model = None

    def train(self, texts: List[str], vocab_size: int = 30000, min_frequency: int = 2) -> None:
        try:
            start_time = time.time()

            # Train Tokenizer
            trainer = BpeTrainer(vocab_size=vocab_size, min_frequency=min_frequency, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
            self.tokenizer.train_from_iterator(texts, trainer=trainer)

            # Train SentencePiece
            with open("temp_corpus.txt", "w", encoding="utf-8") as f:
                for text in texts:
                    f.write(text + "\n")
            
            spm.SentencePieceTrainer.train(
                input="temp_corpus.txt",
                model_prefix=self.model_name,
                vocab_size=vocab_size,
                model_type="bpe",
                character_coverage=1.0
            )
            self.sp_model = spm.SentencePieceProcessor()
            self.sp_model.load(f"{self.model_name}.model")

            os.remove("temp_corpus.txt")

            end_time = time.time()
            print(f"Training completed in {end_time - start_time:.2f} seconds")

        except Exception as e:
            print(f"An error occurred during training: {str(e)}")

    def encode(self, text: str) -> Union[List[int], None]:
        try:
            tiktoken_tokens = self.tiktoken_encoder.encode(text)
            custom_tokens = self.tokenizer.encode(text).ids
            sp_tokens = self.sp_model.encode_as_ids(text) if self.sp_model else []

            # Combine tokens from different tokenizers
            combined_tokens = tiktoken_tokens + custom_tokens + sp_tokens
            return combined_tokens
        except Exception as e:
            print(f"An error occurred during encoding: {str(e)}")
            return None

    def decode(self, tokens: List[int]) -> Optional[str]:
        try:
            # Separate tokens for different tokenizers
            tiktoken_length = len(self.tiktoken_encoder.decode(tokens))
            custom_length = len(self.tokenizer.decode(tokens[:tiktoken_length]))
            
            tiktoken_tokens = tokens[:tiktoken_length]
            custom_tokens = tokens[tiktoken_length:tiktoken_length+custom_length]
            sp_tokens = tokens[tiktoken_length+custom_length:]

            # Decode using respective tokenizers
            tiktoken_text = self.tiktoken_encoder.decode(tiktoken_tokens)
            custom_text = self.tokenizer.decode(custom_tokens)
            sp_text = self.sp_model.decode(sp_tokens) if self.sp_model else ""

            # Combine decoded texts
            return tiktoken_text + custom_text + sp_text
        except Exception as e:
            print(f"An error occurred during decoding: {str(e)}")
            return None



class AdvancedTokenizer_v3:


    def __init__(
        self,
        vocab_size: int = 30000,
        min_frequency: int = 2,
        model_type: str = "bpe",
        special_tokens: Optional[List[str]] = None
    ):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.model_type = model_type
        self.special_tokens = special_tokens or ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
        self.tokenizer: Optional[Tokenizer] = None
        self.sp_model: Optional[spm.SentencePieceProcessor] = None

    def train_from_files(self, file_paths: List[str]) -> None:
        try:
            if self.model_type == "bpe":
                self._train_bpe(file_paths)
            elif self.model_type == "unigram":
                self._train_unigram(file_paths)
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise

    def _train_bpe(self, file_paths: List[str]) -> None:
        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)

        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=self.min_frequency,
            special_tokens=self.special_tokens
        )

        tokenizer.train(file_paths, trainer)

        tokenizer.post_processor = processors.TemplateProcessing(
            single="[CLS] $A [SEP]",
            pair="[CLS] $A [SEP] $B:1 [SEP]:1",
            special_tokens=[("[CLS]", 1), ("[SEP]", 2)],
        )

        self.tokenizer = tokenizer

    def _train_unigram(self, file_paths: List[str]) -> None:
        input_file = self._merge_files(file_paths)
        model_prefix = "unigram_model"
        
        spm.SentencePieceTrainer.train(
            input=input_file,
            model_prefix=model_prefix,
            vocab_size=self.vocab_size,
            model_type="unigram",
            character_coverage=1.0,
            user_defined_symbols=self.special_tokens
        )

        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.load(f"{model_prefix}.model")

        os.remove(input_file)
        os.remove(f"{model_prefix}.model")
        os.remove(f"{model_prefix}.vocab")

    @staticmethod
    def _merge_files(file_paths: List[str]) -> str:
        merged_file = "merged_input.txt"
        with open(merged_file, "w", encoding="utf-8") as outfile:
            for file_path in tqdm(file_paths, desc="Merging files"):
                with open(file_path, "r", encoding="utf-8") as infile:
                    outfile.write(infile.read() + "\n")
        return merged_file

    def save_model(self, path: str) -> None:
        try:
            if self.tokenizer:
                self.tokenizer.save(path)
            elif self.sp_model:
                self.sp_model.save(path)
            logger.info(f"Model saved successfully at {path}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

    def load_model(self, path: str) -> None:
        try:
            if self.model_type == "bpe":
                self.tokenizer = Tokenizer.from_file(path)
            elif self.model_type == "unigram":
                self.sp_model = spm.SentencePieceProcessor()
                self.sp_model.load(path)
            logger.info(f"Model loaded successfully from {path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def encode(self, text: str) -> Optional[List[int]]:
        try:
            if self.tokenizer:
                return self.tokenizer.encode(text).ids
            elif self.sp_model:
                return self.sp_model.encode_as_ids(text)
            else:
                raise ValueError("No model loaded")
        except Exception as e:
            logger.error(f"Error during encoding: {str(e)}")
            return None

    def decode(self, tokens: List[int]) -> Optional[str]:
        try:
            if self.tokenizer:
                return self.tokenizer.decode(tokens)
            elif self.sp_model:
                return self.sp_model.decode_ids(tokens)
            else:
                raise ValueError("No model loaded")
        except Exception as e:
            logger.error(f"Error during decoding: {str(e)}")
            return None

    @staticmethod
    def get_pretrained_tokenizer(model_name: str) -> AutoTokenizer:
        try:
            return AutoTokenizer.from_pretrained(model_name)
        except Exception as e:
            logger.error(f"Error loading pretrained tokenizer: {str(e)}")
            raise


def get_file_paths(folder_path: str, file_extensions: List[str]) -> Iterator[str]:
    for root, _, files in os.walk(folder_path):
        for file in files:
            if any(file.endswith(ext) for ext in file_extensions):
                yield os.path.join(root, file)







Role = Literal["system", "user", "assistant"]

class Message(TypedDict):
    role: Role
    content: str

Dialog = Sequence[Message]

class CustomTokenizer:
    """
    Custom tokenizer class for handling large language models and various tokenization tasks.
    """

    def __init__(self, vocab_file: str):
        """
        Initialize the CustomTokenizer with a vocabulary file.

        Args:
            vocab_file (str): Path to the vocabulary file.
        """
        self._validate_vocab_file(vocab_file)
        self.vocab_file = vocab_file
        self.vocab, self.inverse_vocab = self._load_vocabulary()
        self.special_tokens = self._initialize_special_tokens()
        self._initialize_token_ids()
        logger.info(f"Initialized CustomTokenizer with vocabulary: {vocab_file}")

    def _validate_vocab_file(self, vocab_file: str) -> None:
        if not os.path.isfile(vocab_file):
            raise FileNotFoundError(f"Vocabulary file not found: {vocab_file}")

    def _load_vocabulary(self) -> tuple[Dict[str, int], Dict[int, str]]:
        vocab = {}
        with open(self.vocab_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                token = line.strip()
                vocab[token] = i
        inverse_vocab = {v: k for k, v in vocab.items()}
        return vocab, inverse_vocab

    def _initialize_special_tokens(self) -> Dict[str, int]:
        special_tokens = {
            "<|begin_of_text|>": len(self.vocab),
            "<|end_of_text|>": len(self.vocab) + 1,
            "<|pad|>": len(self.vocab) + 2,
            "<|unk|>": len(self.vocab) + 3,
            "<|start_header_id|>": len(self.vocab) + 4,
            "<|end_header_id|>": len(self.vocab) + 5,
            "<|eot_id|>": len(self.vocab) + 6,
        }
        self.vocab.update(special_tokens)
        self.inverse_vocab.update({v: k for k, v in special_tokens.items()})
        return special_tokens

    def _initialize_token_ids(self) -> None:
        self.n_words: int = len(self.vocab)
        self.bos_id: int = self.special_tokens["<|begin_of_text|>"]
        self.eos_id: int = self.special_tokens["<|end_of_text|>"]
        self.pad_id: int = self.special_tokens["<|pad|>"]
        self.unk_id: int = self.special_tokens["<|unk|>"]
        self.stop_tokens = {self.eos_id, self.special_tokens["<|eot_id|>"]}
        logger.info(f"Vocabulary size: {self.n_words}")
        logger.info(f"Special token IDs - BOS: {self.bos_id}, EOS: {self.eos_id}, PAD: {self.pad_id}, UNK: {self.unk_id}")

    def encode(
        self,
        text: str,
        bos: bool = False,
        eos: bool = False,
        allowed_special: Union[Literal["all"], AbstractSet[str]] = set(),
        disallowed_special: Union[Literal["all"], Collection[str]] = (),
    ) -> List[int]:
        """
        Encode a string into a list of token IDs.

        Args:
            text (str): Input string to be encoded.
            bos (bool): Whether to prepend the beginning-of-sequence token.
            eos (bool): Whether to append the end-of-sequence token.
            allowed_special (Union[Literal["all"], AbstractSet[str]]): Allowed special tokens.
            disallowed_special (Union[Literal["all"], Collection[str]]): Disallowed special tokens.

        Returns:
            List[int]: List of token IDs.
        """
        if not isinstance(text, str):
            raise TypeError("Input must be a string")

        tokens = []
        if bos:
            tokens.append(self.bos_id)

        words = self._tokenize(text)
        for word in words:
            if word in self.vocab:
                tokens.append(self.vocab[word])
            else:
                tokens.append(self.unk_id)

        if eos:
            tokens.append(self.eos_id)

        return tokens

    def decode(self, tokens: Sequence[int]) -> str:
        """
        Decode a list of token IDs into a string.

        Args:
            tokens (Sequence[int]): List of token IDs to be decoded.

        Returns:
            str: Decoded string.
        """
        return ' '.join(self.inverse_vocab.get(token, '<|unk|>') for token in tokens)

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize the input text into words.

        Args:
            text (str): Input text to be tokenized.

        Returns:
            List[str]: List of tokens.
        """
        # This is a simple word-based tokenization. You can implement more sophisticated
        # tokenization methods here, such as subword tokenization or byte-pair encoding.
        return re.findall(r'\b\w+\b|[^\w\s]', text.lower())


class AdvancedTokenizer:
    """
    Advanced tokenizer class for handling large language models and various tokenization tasks.
    """

    def __init__(self, model_path: str):
        """
        Initialize the AdvancedTokenizer with a Tiktoken model.

        Args:
            model_path (str): Path to the Tiktoken model file.
        """
        self._validate_model_path(model_path)
        self.model_path = model_path
        self.special_tokens = self._initialize_special_tokens()
        self.model = self._load_tiktoken_model()
        self._initialize_token_ids()
        logger.info(f"Initialized AdvancedTokenizer with model: {model_path}")

    def _validate_model_path(self, model_path: str) -> None:
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

    def _initialize_special_tokens(self) -> Dict[str, int]:
        mergeable_ranks = load_tiktoken_bpe(self.model_path)
        num_base_tokens = len(mergeable_ranks)
        special_tokens = [
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|pad|>",
            "<|unk|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|eot_id|>",
        ] + [f"<|reserved_special_token_{i}|>" for i in range(249)]
        return {token: num_base_tokens + i for i, token in enumerate(special_tokens)}

    def _load_tiktoken_model(self) -> tiktoken.Encoding:
        return tiktoken.Encoding(
            name=Path(self.model_path).name,
            pat_str=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
            mergeable_ranks=load_tiktoken_bpe(self.model_path),
            special_tokens=self.special_tokens,
        )

    def _initialize_token_ids(self) -> None:
        self.n_words: int = self.model.n_vocab
        self.bos_id: int = self.special_tokens["<|begin_of_text|>"]
        self.eos_id: int = self.special_tokens["<|end_of_text|>"]
        self.pad_id: int = self.special_tokens["<|pad|>"]
        self.unk_id: int = self.special_tokens["<|unk|>"]
        self.stop_tokens = {self.eos_id, self.special_tokens["<|eot_id|>"]}
        logger.info(f"Vocabulary size: {self.n_words}")
        logger.info(f"Special token IDs - BOS: {self.bos_id}, EOS: {self.eos_id}, PAD: {self.pad_id}, UNK: {self.unk_id}")

    def encode(
        self,
        text: str,
        bos: bool = False,
        eos: bool = False,
        allowed_special: Union[Literal["all"], AbstractSet[str]] = set(),
        disallowed_special: Union[Literal["all"], Collection[str]] = (),
    ) -> List[int]:
        """
        Encode a string into a list of token IDs.

        Args:
            text (str): Input string to be encoded.
            bos (bool): Whether to prepend the beginning-of-sequence token.
            eos (bool): Whether to append the end-of-sequence token.
            allowed_special (Union[Literal["all"], AbstractSet[str]]): Allowed special tokens.
            disallowed_special (Union[Literal["all"], Collection[str]]): Disallowed special tokens.

        Returns:
            List[int]: List of token IDs.
        """
        if not isinstance(text, str):
            raise TypeError("Input must be a string")

        tokens = []
        if bos:
            tokens.append(self.bos_id)

        chunks = self._split_text(text)
        for chunk in chunks:
            tokens.extend(
                self.model.encode(
                    chunk,
                    allowed_special=allowed_special,
                    disallowed_special=disallowed_special,
                )
            )

        if eos:
            tokens.append(self.eos_id)

        return tokens

    def decode(self, tokens: Sequence[int]) -> str:
        """
        Decode a list of token IDs into a string.

        Args:
            tokens (Sequence[int]): List of token IDs to be decoded.

        Returns:
            str: Decoded string.
        """
        return self.model.decode(cast(List[int], tokens))

    def _split_text(self, text: str, max_chunk_size: int = 400_000) -> Iterator[str]:
        """
        Split the input text into smaller chunks to handle very large inputs.

        Args:
            text (str): Input text to be split.
            max_chunk_size (int): Maximum size of each chunk.

        Yields:
            str: Text chunks.
        """
        for i in range(0, len(text), max_chunk_size):
            yield text[i:i + max_chunk_size]

class ChatFormatter:
    """
    Format chat dialogs for tokenization.
    """

    def __init__(self, tokenizer: AdvancedTokenizer):
        self.tokenizer = tokenizer

    def encode_header(self, message: Message) -> List[int]:
        tokens = []
        tokens.append(self.tokenizer.special_tokens["<|start_header_id|>"])
        tokens.extend(self.tokenizer.encode(message["role"], bos=False, eos=False))
        tokens.append(self.tokenizer.special_tokens["<|end_header_id|>"])
        tokens.extend(self.tokenizer.encode("\n\n", bos=False, eos=False))
        return tokens

    def encode_message(self, message: Message) -> List[int]:
        tokens = self.encode_header(message)
        tokens.extend(self.tokenizer.encode(message["content"].strip(), bos=False, eos=False))
        tokens.append(self.tokenizer.special_tokens["<|eot_id|>"])
        return tokens

    def encode_dialog_prompt(self, dialog: Dialog) -> List[int]:
        tokens = [self.tokenizer.bos_id]
        for message in dialog:
            tokens.extend(self.encode_message(message))
        tokens.extend(self.encode_header({"role": "assistant", "content": ""}))
        return tokens





model_path = r"C:\Users\heman\Desktop\Coding\model_outputs\tokenizer\tokenizer (1).model"
tokenizer = AdvancedTokenizer(model_path)
chat_formatter = ChatFormatter(tokenizer)

Text="""
OneShot is a puzzle-adventure game developed by indie studio Future Cat 
and published by Degica. Based on a free version released online on June 30,
2014, it was released for Windows on December 8, 2016. A console adaptation,
OneShot: World Machine Edition, was released for Nintendo Switch, 
PlayStation 4, and Xbox One on September 22, 2022. OneShot's gameplay and 
plot break the fourth wall and involve metafictional elements. Many puzzles involve 
interacting with the computer's operating system outside of the game. Narratively, 
the player is separate from the protagonist, Niko. The latter arrives in a world without 
sunlight and aims to restore it by replacing its sun, a lightbulb, at the top of a tower.
OneShot was developed in RPG Maker XP. The game received positive reviews from critics,
who praised the story, art, and metafictional aspects of gameplay, including the relationship 
between the player and Niko. In 2017, the game was nominated for the PC Game of the Year category 
at the Golden Joystick Awards. (Full article...)
Recently featured: Nihonium"Well he would, wouldn't he?"
"""
encode=tokenizer.encode(Text)
print(encode)
print(tokenizer.decode(tokenizer.encode(Text)))