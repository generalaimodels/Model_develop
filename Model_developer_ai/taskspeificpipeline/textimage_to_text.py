#!/usr/bin/env python3
"""
Image Processing Pipeline with Language Model Integration.

This script processes an input image using a specified language model,
generating a response based on a given prompt.
"""

import argparse
import logging
from pathlib import Path
from typing import Optional, Union

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer, AutoProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_model(model_name: str, cache_dir: Optional[str], device: str) -> AutoModel:
    """
    Load the specified model.

    Args:
        model_name (str): Name of the model to load.
        cache_dir (Optional[str]): Directory to cache the model.
        device (str): Device to load the model on ('cpu' or 'cuda').

    Returns:
        AutoModel: Loaded model.
    """
    try:
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            cache_dir=cache_dir
        )
        model = model.to(device=device)
        model.eval()
        logger.info(f"Model {model_name} loaded successfully on {device}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def load_tokenizer(model_name: str, cache_dir: Optional[str]) -> AutoTokenizer:
    """
    Load the tokenizer for the specified model.

    Args:
        model_name (str): Name of the model to load the tokenizer for.
        cache_dir (Optional[str]): Directory to cache the tokenizer.

    Returns:
        AutoTokenizer: Loaded tokenizer.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir=cache_dir)
        logger.info(f"Tokenizer for {model_name} loaded successfully")
        return tokenizer
    except Exception as e:
        logger.error(f"Error loading tokenizer: {e}")
        raise

def load_image(image_path: Union[str, Path]) -> Image.Image:
    """
    Load an image from the specified path.

    Args:
        image_path (Union[str, Path]): Path to the image file.

    Returns:
        Image.Image: Loaded image.
    """
    try:
        image = Image.open(image_path).convert('RGB')
        logger.info(f"Image loaded successfully from {image_path}")
        return image
    except Exception as e:
        logger.error(f"Error loading image: {e}")
        raise

def process_image(
    image: Image.Image,
    prompt: str,
    model: AutoModel,
    tokenizer: AutoTokenizer,
    temperature: float = 0.7,
    max_new_tokens: int = 100
) -> str:
    """
    Process the image using the loaded model and generate a response.

    Args:
        image (Image.Image): Input image.
        prompt (str): Text prompt for the model.
        model (AutoModel): Loaded language model.
        tokenizer (AutoTokenizer): Loaded tokenizer.
        temperature (float): Sampling temperature.
        max_new_tokens (int): Maximum number of new tokens to generate.

    Returns:
        str: Generated response.
    """
    try:
        msgs = [{'role': 'user', 'content': prompt}]
        
        res = model.chat(
            image=image,
            msgs=msgs,
            tokenizer=tokenizer,
            sampling=True,
            temperature=temperature,
            max_new_tokens=max_new_tokens
        )
        logger.info("Image processed successfully")
        return res
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise

def main(
    image_path: str,
    prompt: str,
    model_name: str,
    cache_dir: Optional[str],
    device: str
) -> None:
    """
    Main function to run the image processing pipeline.

    Args:
        image_path (str): Path to the input image.
        prompt (str): Text prompt for the model.
        model_name (str): Name of the model to use.
        cache_dir (Optional[str]): Directory to cache the model and tokenizer.
        device (str): Device to run the model on ('cpu' or 'cuda').
    """
    try:
        model = load_model(model_name, cache_dir, device)
        tokenizer = load_tokenizer(model_name, cache_dir)
        image = load_image(image_path)
        
        response = process_image(image, prompt, model, tokenizer)
        print(f"Generated response: {response}")
        
    except Exception as e:
        logger.error(f"An error occurred in the main pipeline: {e}")
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Image-based chat model pipeline.

This script provides a pipeline for generating responses based on
input images and prompts using a specified model.
"""

import logging
from typing import Optional, List, Dict, Union
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ImageChatPipeline:
    """A pipeline for image-based chat using a specified model."""

    def __init__(
        self,
        model_name: str,
        cache_dir: Optional[Union[str, Path]] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize the ImageChatPipeline.

        Args:
            model_name (str): The name of the model to use.
            cache_dir (Optional[Union[str, Path]]): Directory to cache the model.
            device (str): The device to run the model on (default: "cuda" if available, else "cpu").
        """
        self.device = device
        self.cache_dir = Path(cache_dir) if cache_dir else None

        logger.info(f"Initializing model: {model_name}")
        try:
            self.model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                cache_dir=self.cache_dir,
            ).to(device=self.device)

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                cache_dir=self.cache_dir,
            )
            self.model.eval()
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise

        logger.info("Model initialized successfully")

    def process_image(self, image_path: Union[str, Path]) -> Image.Image:
        """
        Process and load the input image.

        Args:
            image_path (Union[str, Path]): Path to the input image.

        Returns:
            Image.Image: The processed PIL Image object.
        """
        try:
            image = Image.open(image_path).convert('RGB')
            logger.info(f"Image loaded successfully: {image_path}")
            return image
        except Exception as e:
            logger.error(f"Failed to load image: {e}")
            raise

    def generate_response(
        self,
        image: Image.Image,
        prompt: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        stream: bool = False,
    ) -> Union[str, Generator[str, None, None]]:
        """
        Generate a response based on the input image and prompt.

        Args:
            image (Image.Image): The input image.
            prompt (str): The input prompt.
            messages (List[Dict[str, str]]): List of previous messages.
            temperature (float): Sampling temperature (default: 0.7).
            stream (bool): Whether to stream the output (default: False).

        Returns:
            Union[str, Generator[str, None, None]]: The generated response or a generator for streaming.
        """
        messages.append({"role": "user", "content": prompt})

        try:
            response = self.model.chat(
                image=image,
                msgs=messages,
                tokenizer=self.tokenizer,
                sampling=True,
                temperature=temperature,
                stream=stream,
            )

            if stream:
                return self._stream_response(response)
            return response

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise

    def _stream_response(self, response_generator):
        """
        Stream the response from the model.

        Args:
            response_generator: Generator returned by the model.

        Yields:
            str: Chunks of the generated response.
        """
        generated_text = ""
        for new_text in response_generator:
            generated_text += new_text
            yield new_text


def main():
    """Main function to demonstrate the usage of ImageChatPipeline."""
    model_name = "openbmb/MiniCPM-Llama3-V-2_5"
    cache_dir = "./model_cache"
    image_path = "path/to/your/image.jpg"
    prompt = "What is in the image?"

    try:
        pipeline = ImageChatPipeline(model_name, cache_dir)
        image = pipeline.process_image(image_path)
        messages = []

        logger.info("Generating response...")
        response = pipeline.generate_response(image, prompt, messages)
        logger.info(f"Generated response: {response}")

        # Demonstrate streaming
        logger.info("Generating streamed response...")
        for chunk in pipeline.generate_response(image, prompt, messages, stream=True):
            print(chunk, end='', flush=True)
        print()  # New line after streaming

    except Exception as e:
        logger.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Processing Pipeline")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for the model")
    parser.add_argument("--model_name", type=str, default="openbmb/MiniCPM-Llama3-V-2_5", help="Name of the model to use")
    parser.add_argument("--cache_dir", type=str, help="Directory to cache the model and tokenizer")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run the model on ('cpu' or 'cuda')")
    
    args = parser.parse_args()
    
    main(args.image_path, args.prompt, args.model_name, args.cache_dir, args.device)