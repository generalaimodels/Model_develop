import sys
from pathlib import Path
file=Path(__file__).resolve()
sys.path.append(str(file.parents[1]))
import time
from typing import List, Optional, Dict, Any
from FAST_ANALYSIS import AiModelForHemanth, AdvancedPreProcessForHemanth
import torch
from torch.nn import functional as F
import logging
from typing import Optional, Union, List
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from transformers.generation.utils import GenerationMode
from transformers.generation.logits_process import (
    LogitsProcessorList,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
from transformers.generation.stopping_criteria import StoppingCriteriaList, MaxLengthCriteria

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedTextGenerator:
    def __init__(self, model_name: str):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.info(f"Initialized AdvancedTextGenerator with model: {model_name}")

    def generate_text(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        num_return_sequences: int = 1,
        do_sample: bool = True,
        num_beams: Optional[int] = None
    ) -> List[str]:
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            
            generation_config = GenerationConfig(
                max_length=max_length,
                num_return_sequences=num_return_sequences,
                do_sample=do_sample,
                num_beams=num_beams or 1,
            )

            logits_processor = LogitsProcessorList()
            if temperature != 1.0:
                logits_processor.append(TemperatureLogitsWarper(temperature))
            if top_k is not None:
                logits_processor.append(TopKLogitsWarper(top_k=top_k))
            if top_p is not None:
                logits_processor.append(TopPLogitsWarper(top_p=top_p))

            stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])

            outputs = self.model.generate(
                **inputs,
                generation_config=generation_config,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
            )

            generated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            logger.info(f"Generated {len(generated_texts)} text sequences")
            return generated_texts

        except Exception as e:
            logger.error(f"Error during text generation: {str(e)}")
            return []


class LLMGenerator:
    def __init__(
        self,
        max_seq_len: int = 2048,
        max_batch_size: int = 32,
        tokenizer: AdvancedPreProcessForHemanth= None,
        model:AiModelForHemanth= None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ) -> None:
        """
        Initialize the LLMGenerator with a specified model.

        Args:
            model_name (str): The name or path of the pre-trained model.
            max_seq_len (int): Maximum sequence length for input text.
            max_batch_size (int): Maximum batch size for inference.
            device (str): The device to run the model on ("cuda" or "cpu").

        Raises:
            ValueError: If the model or tokenizer cannot be loaded.
        """
        try:
            self.tokenizer = tokenizer
            self.model = model
        except Exception as e:
            raise ValueError(f"Failed to load model or tokenizer: {str(e)}")

        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size
        self.device = device

    def generate(
        self,
        prompt_tokens: List[List[int]],
        max_gen_len: int,
        temperature: float = 0.6,
        top_p: float = 0.3,
        logprobs: bool = False,
        echo: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate text based on input tokens.

        Args:
            prompt_tokens (List[List[int]]): List of tokenized prompts.
            max_gen_len (int): Maximum length of generated text.
            temperature (float): Temperature for sampling.
            top_p (float): Top-p value for nucleus sampling.
            logprobs (bool): Whether to return log probabilities.
            echo (bool): Whether to include input prompt in the output.

        Returns:
            Dict[str, Any]: A dictionary containing generated tokens and optionally log probabilities.

        Raises:
            ValueError: If input parameters are invalid.
        """
        if not prompt_tokens:
            raise ValueError("prompt_tokens cannot be empty")
        if max_gen_len <= 0:
            raise ValueError("max_gen_len must be positive")
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        if not 0 < top_p <= 1:
            raise ValueError("top_p must be between 0 and 1")

        batch_size = len(prompt_tokens)
        if batch_size > self.max_batch_size:
            raise ValueError(f"Batch size {batch_size} exceeds maximum {self.max_batch_size}")

        max_prompt_len = max(len(t) for t in prompt_tokens)
        if max_prompt_len > self.max_seq_len:
            raise ValueError(f"Prompt length {max_prompt_len} exceeds maximum sequence length {self.max_seq_len}")

        total_len = min(self.max_seq_len, max_gen_len + max_prompt_len)

        # Prepare input tensors
        input_ids = torch.full((batch_size, total_len), self.tokenizer.pad_token_id, dtype=torch.long, device=self.device)
        for i, tokens in enumerate(prompt_tokens):
            input_ids[i, :len(tokens)] = torch.tensor(tokens, dtype=torch.long, device=self.device)
        start_time = time.time()
        try:
          with torch.no_grad():
              for i in range(max_prompt_len, total_len):
                  outputs = self.model(input_ids[:, :i])
                  next_token_logits = outputs.logits[:, -1, :]
                  
                  print(f"Step {i}: Logits shape: {next_token_logits.shape}")
                  print(f"Step {i}: Logits min: {next_token_logits.min().item()}, max: {next_token_logits.max().item()}")
                  
                  next_token_logits = next_token_logits / temperature
                  filtered_logits = top_p_filtering(next_token_logits, top_p=top_p)
                  
                  probs = F.softmax(filtered_logits, dim=-1)
                  print(f"Step {i}: Probs min: {probs.min().item()}, max: {probs.max().item()}")
                  
                  next_token = torch.multinomial(probs, num_samples=1)
                  for batch_idx, token in enumerate(next_token):
                      print(f"Step {i}, Batch {batch_idx}: Next token: {token.item()}, Token text: '{self.tokenizer.decode([token.item()])}'")
                #   print(f"Step {i}: Next token: {next_token.item()}, Token text: '{self.tokenizer.decode([next_token.item()])}'")
                #   print(f"Step {i}: Next token: {next_token.item()}, Token text: '{self.tokenizer.decode([next_token.item()])}'")
                  
                  input_ids[:, i] = next_token.squeeze()
  
                  if self.tokenizer.eos_token_id in input_ids[:, i]:
                      print(f"EOS token generated at step {i}")
                      break

        except Exception as e:
            print(f"Error in generate: {str(e)}")
            print(f"Last logits: {next_token_logits}")
            print(f"Last probs: {probs}")
            raise

        generation_time = time.time() - start_time

        # Process output
        generated_sequences = []
        for seq in input_ids:
            eos_idx = (seq == self.tokenizer.eos_token_id).nonzero()
            if eos_idx.numel() > 0:
                seq = seq[:eos_idx[0].item()]
            generated_sequences.append(seq.tolist())

        output = {
            "generated_tokens": generated_sequences,
            "generation_time": generation_time,
        }
        try:
           if logprobs:
               logits = outputs.logits[:, :-1, :]
               input_ids_slice = input_ids[:, 1:logits.size(1) + 1]
               log_probs = logits.log_softmax(-1).gather(-1, input_ids_slice.unsqueeze(-1)).squeeze(-1)
               output["log_probs"] = log_probs.tolist()
        except Exception as e:
            print(f"Error calculating log probabilities: {str(e)}")
            print(f"Logits shape: {outputs.logits.shape}")
            print(f"Input IDs shape: {input_ids.shape}")
            raise

        # if logprobs:
        #     # Compute log probabilities (simplified version)
        #     output["log_probs"] = [outputs.logits[:, :-1, :].log_softmax(-1).gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1).tolist()]

        if not echo:
            output["generated_tokens"] = [seq[len(prompt):] for seq, prompt in zip(generated_sequences, prompt_tokens)]

        return output

    def text_completion(
        self,
        prompts: List[str],
        max_gen_len: Optional[int] = None,
        temperature: float = 0.6,
        top_p: float = 0.4,
        logprobs: bool = False,
        echo: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Perform text completion for a list of prompts.

        Args:
            prompts (List[str]): List of text prompts.
            max_gen_len (Optional[int]): Maximum length of generated text.
            temperature (float): Temperature for sampling.
            top_p (float): Top-p value for nucleus sampling.
            logprobs (bool): Whether to return log probabilities.
            echo (bool): Whether to include input prompt in the output.

        Returns:
            List[Dict[str, Any]]: List of dictionaries containing generated text and metadata.

        Raises:
            ValueError: If input parameters are invalid.
        """
        if not prompts:
            raise ValueError("prompts cannot be empty")

        if max_gen_len is None:
            max_gen_len = self.max_seq_len - 1

        try:
           prompt_tokens = [self.tokenizer.encode(prompt, add_special_tokens=True) for prompt in prompts]
           print("Encoded prompts:")
           for i, tokens in enumerate(prompt_tokens):
               print(f"Prompt {i + 1}: {tokens}")
           
           generation_output = self.generate(
               prompt_tokens=prompt_tokens,
               max_gen_len=max_gen_len,
               temperature=temperature,
               top_p=top_p,
               logprobs=logprobs,
               echo=echo,
           )
   
           completions = []
           for i, tokens in enumerate(generation_output["generated_tokens"]):
               decoded_text = self.tokenizer.decode(tokens)
               completion = {
                   "generated_text": decoded_text,
                   "num_generated_tokens": len(tokens),
                   "raw_tokens": tokens,
               }
               if not decoded_text.strip():
                   print(f"Warning: Empty generated text for prompt {i + 1}")
                   print(f"Tokens: {tokens}")
               if logprobs:
                   completion["token_logprobs"] = generation_output["log_probs"][i]
               completions.append(completion)
   
           return completions

        except Exception as e:
           print(f"Error during text completion: {str(e)}")
           import traceback
           traceback.print_exc()


def top_p_filtering(logits: torch.Tensor, top_p: float = 0.9) -> torch.Tensor:
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    logits[:, indices_to_remove] = float('-inf')
    
    # Check for NaN or Inf values
    if torch.isnan(logits).any() or torch.isinf(logits).any():
        print("Warning: NaN or Inf values in logits after top_p_filtering")
        print(f"Logits min: {logits.min().item()}, max: {logits.max().item()}")
    
    return logits

import logging
from typing import Optional, Union, List
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from transformers.generation.utils import GenerationMode
from transformers.generation.logits_process import (
    LogitsProcessorList,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
from transformers.generation.stopping_criteria import StoppingCriteriaList, MaxLengthCriteria

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedTextGenerator:
    def __init__(self, model_name: str):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.info(f"Initialized AdvancedTextGenerator with model: {model_name}")

    def generate_text(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        num_return_sequences: int = 1,
        do_sample: bool = True,
        num_beams: Optional[int] = None
    ) -> List[str]:
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            
            # Set up generation config
            generation_config = GenerationConfig(
                max_length=max_length,
                num_return_sequences=num_return_sequences,
                do_sample=do_sample,
                num_beams=num_beams or 1,
            )

            # Set up logits processors
            logits_processor = LogitsProcessorList()
            if temperature != 1.0:
                logits_processor.append(TemperatureLogitsWarper(temperature))
            if top_k is not None:
                logits_processor.append(TopKLogitsWarper(top_k))
            if top_p is not None:
                logits_processor.append(TopPLogitsWarper(top_p))

            # Set up stopping criteria
            stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])

            # Generate text
            outputs = self.model.generate(
                **inputs,
                generation_config=generation_config,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
            )

            generated_texts = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
            logger.info(f"Generated {len(generated_texts)} text sequences")
            return generated_texts

        except Exception as e:
            logger.error(f"Error during text generation: {str(e)}")
            return []

def main():
    model_name = input("Enter the model name: ")
    generator = AdvancedTextGenerator(model_name)

    prompt = input("Enter your prompt: ")
    max_length = int(input("Enter max length (default 100): ") or 100)
    temperature = float(input("Enter temperature (default 1.0): ") or 1.0)
    top_k = int(input("Enter top_k (optional): ") or 0) or None
    top_p = float(input("Enter top_p (optional): ") or 0) or None
    num_return_sequences = int(input("Enter number of sequences to generate (default 1): ") or 1)
    do_sample = input("Use sampling? (y/n, default y): ").lower() != 'n'
    num_beams = int(input("Enter number of beams for beam search (optional): ") or 0) or None

    generated_texts = generator.generate_text(
        prompt=prompt,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        num_return_sequences=num_return_sequences,
        do_sample=do_sample,
        num_beams=num_beams
    )

    print("\nGenerated Text:")
    for i, text in enumerate(generated_texts, 1):
        print(f"\nSequence {i}:")
        print(text)

if __name__ == "__main__":
    main()
def main():
    model_name = input("Enter the model name: ")
    generator = AdvancedTextGenerator(model_name)

    prompt = input("Enter your prompt: ")
    max_length = int(input("Enter max length (default 100): ") or 100)
    
    temperature = float(input("Enter temperature (default 1.0): ") or 1.0)
    top_k = int(input("Enter top_k (optional): ") or 0) or None
    top_p = float(input("Enter top_p (optional): ") or 0) or None
    
    num_return_sequences = int(input("Enter number of sequences to generate (default 1): ") or 1)
    
    do_sample = input("Use sampling? (y/n, default y): ").lower() != 'n'
    num_beams = int(input("Enter number of beams for beam search (optional): ") or 0) or None

    generated_texts = generator.generate_text(
        prompt=prompt,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        num_return_sequences=num_return_sequences,
        do_sample=do_sample,
        num_beams=num_beams
    )

    print("\nGenerated Text(s):")
    for i, text in enumerate(generated_texts, 1):
        print(f"\n{i}. {text}")

if __name__ == "__main__":
    main()


# def top_p_filtering(logits: torch.Tensor, top_p: float = 0.9) -> torch.Tensor:
#     """
#     Filter a distribution of logits using nucleus (top-p) sampling.

#     Args:
#         logits (torch.Tensor): Logits distribution shape (batch size, vocabulary size).
#         top_p (float): Keep the top tokens with cumulative probability >= top_p.

#     Returns:
#         torch.Tensor: Filtered logits.
#     """
#     sorted_logits, sorted_indices = torch.sort(logits, descending=True)
#     cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

#     # Remove tokens with cumulative probability above the threshold
#     sorted_indices_to_remove = cumulative_probs > top_p
#     # Shift the indices to the right to keep also the first token above the threshold
#     sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
#     sorted_indices_to_remove[..., 0] = 0

#     indices_to_remove = sorted_indices[sorted_indices_to_remove]
#     logits[:, indices_to_remove] = float('-inf')
#     return logits
