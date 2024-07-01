import logging
from typing import Optional, Tuple, Union
import requests
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_image(url: str) -> Image.Image:
    """
    Load an image from a given URL.

    Args:
        url (str): URL of the image.

    Returns:
        Image.Image: Loaded image.
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        return Image.open(response.raw)
    except requests.RequestException as e:
        logger.error(f"Error loading image from URL: {e}")
        raise

def process_image(
    image: Image.Image,
    prompt: str,
    model_name: str = "microsoft/Florence-2-large",
    cache_dir: Optional[str] = None,
    token: Optional[str] = None,
    device: str = "cpu"
) -> Tuple[dict, str]:
    """
    Process an image using the Florence-2 model.

    Args:
        image (Image.Image): Input image.
        prompt (str): Prompt for the model.
        model_name (str): Name of the model to use.
        cache_dir (Optional[str]): Directory to cache the model.
        token (Optional[str]): Authentication token.
        device (str): Device to run the model on.

    Returns:
        Tuple[dict, str]: Parsed answer and generated text.
    """
    try:
        logger.info(f"Loading model: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=cache_dir,
            token=token
        ).to(device)
        
        processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=cache_dir,
            token=token
        )

        logger.info("Processing input")
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        logger.info("Generating output")
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            num_beams=3,
            do_sample=False,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.0
        )

        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = processor.post_process_generation(
            generated_text,
            task=prompt,
            image_size=(image.width, image.height)
        )

        return parsed_answer, generated_text

    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise





def load_image_from_url(url: str) -> Image.Image:
    """
    Load an image from a given URL.
    
    Args:
        url (str): The URL of the image.
    
    Returns:
        Image.Image: The loaded image.
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        return Image.open(response.raw)
    except requests.RequestException as e:
        logger.error(f"Error loading image from URL: {e}")
        raise

def load_model_and_processor(
    model_name: str,
    cache_dir: Optional[str] = None,
    device: str = "cpu"
) -> Tuple[AutoModelForCausalLM, AutoProcessor]:
    """
    Load the model and processor.
    
    Args:
        model_name (str): The name of the model to load.
        cache_dir (Optional[str]): The cache directory for the model.
        device (str): The device to load the model on.
    
    Returns:
        Tuple[AutoModelForCausalLM, AutoProcessor]: The loaded model and processor.
    """
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=cache_dir
        ).to(device)
        processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=cache_dir
        )
        logger.info(f"Model and processor loaded successfully: {model_name}")
        return model, processor
    except Exception as e:
        logger.error(f"Error loading model and processor: {e}")
        raise

def generate_image_description(
    model: AutoModelForCausalLM,
    processor: AutoProcessor,
    image: Image.Image,
    prompt: str,
    max_new_tokens: int = 1024,
    num_beams: int = 3,
    do_sample: bool = False
) -> Union[dict, str]:
    """
    Generate a description for the given image using the loaded model.
    
    Args:
        model (AutoModelForCausalLM): The loaded model.
        processor (AutoProcessor): The loaded processor.
        image (Image.Image): The input image.
        prompt (str): The prompt for the task.
        max_new_tokens (int): Maximum number of new tokens to generate.
        num_beams (int): Number of beams for beam search.
        do_sample (bool): Whether to use sampling for generation.
    
    Returns:
        Union[dict, str]: The generated description or parsed answer.
    def main():
    model_name = "microsoft/Florence-2-large"
    image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true"
    prompt = "<OD>"
    cache_dir = None
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        image = load_image_from_url(image_url)
        model, processor = load_model_and_processor(model_name, cache_dir, device)
        result = generate_image_description(model, processor, image, prompt)
        print(result)
    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
    """
    try:
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=do_sample
        )
        
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = processor.post_process_generation(
            generated_text,
            task=prompt,
            image_size=(image.width, image.height)
        )
        
        logger.info("Image description generated successfully")
        return parsed_answer
    except Exception as e:
        logger.error(f"Error generating image description: {e}")
        raise

