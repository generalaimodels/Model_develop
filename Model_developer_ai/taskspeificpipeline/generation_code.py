import sys
from pathlib import Path
file=Path(__file__).resolve()
sys.path.append(str(file.parents[1]))
import logging
from typing import List, Optional, Union
import torch

from FAST_ANALYSIS import AiModelForHemanth, AdvancedPreProcessForHemanth
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CodeGenerator_test:
    def __init__(
        self,
        tokenizer:AiModelForHemanth,
        model: AdvancedPreProcessForHemanth,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        try:
            self.tokenizer = tokenizer
            self.model = model
            self.model.to(self.device)
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def generate_code(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 50,
        num_return_sequences: int = 1
    ) -> List[str]:
        try:
            messages = [{"role": "user", "content": prompt}]
            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    num_return_sequences=num_return_sequences,
                    eos_token_id=self.tokenizer.eos_token_id
                )

            generated_texts = [
                self.tokenizer.decode(output[len(inputs[0]):], skip_special_tokens=True)
                for output in outputs
            ]
            return generated_texts
        except Exception as e:
            logger.error(f"Error generating code: {e}")
            return []

class CodeGenerator:
    def __init__(self,
                 model: AiModelForHemanth,
                 tokenizer: AdvancedPreProcessForHemanth,
                 device: Optional[str] = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

        try:
            self.tokenizer = tokenizer
            self.model = model
            self.model.to(self.device)
            logger.info(f"Model  loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def generate_code(self, messages: List[dict], max_new_tokens: int = 512,
                      temperature: float = 0.7, top_p: float = 0.95,
                      top_k: int = 50) -> str:
        try:
            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    num_return_sequences=1,
                    eos_token_id=self.tokenizer.eos_token_id
                )

            generated_text = self.tokenizer.decode(
                outputs[0][len(inputs[0]):],
                skip_special_tokens=True
            )
            logger.info("Code generated successfully")
            return generated_text.strip()
        except Exception as e:
            logger.error(f"Error during code generation: {str(e)}")
            raise