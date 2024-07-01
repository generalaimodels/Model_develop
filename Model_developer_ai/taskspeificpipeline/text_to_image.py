import logging
from typing import Union, Dict, Any, Optional
import torch
from diffusers import StableDiffusion3Pipeline, SD3Transformer2DModel, FlashFlowMatchEulerDiscreteScheduler
from peft import PeftModel
import plotly.graph_objects as go
from PIL import Image
import requests
from io import BytesIO

class AdvancedStableDiffusion:
    """
    An advanced class for Stable Diffusion 3 image generation.
    """

    def __init__(
        self,
        model_path: str,
        lora_path: Optional[str] = None,
        device: str = "cuda",
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize the AdvancedStableDiffusion class.

        Args:
            model_path (str): Path to the pre-trained model.
            lora_path (Optional[str]): Path to the LoRA model.
            device (str): Device to run the model on (default: "cuda").
            cache_dir (Optional[str]): Directory to cache model files.
        """
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.device = device
        self.cache_dir = cache_dir

        self.logger.info(f"Initializing model from {model_path}")
        self._initialize_pipeline(model_path, lora_path)

    def _initialize_pipeline(self, model_path: str, lora_path: Optional[str]):
        """
        Initialize the Stable Diffusion pipeline.

        Args:
            model_path (str): Path to the pre-trained model.
            lora_path (Optional[str]): Path to the LoRA model.
        """
        try:
            if lora_path:
                transformer = SD3Transformer2DModel.from_pretrained(
                    model_path,
                    subfolder="transformer",
                    torch_dtype=torch.float16,
                    cache_dir=self.cache_dir,
                )
                transformer = PeftModel.from_pretrained(transformer, lora_path)
                self.pipe = StableDiffusion3Pipeline.from_pretrained(
                    model_path,
                    transformer=transformer,
                    torch_dtype=torch.float16,
                    text_encoder_3=None,
                    tokenizer_3=None,
                    cache_dir=self.cache_dir,
                )
            else:
                self.pipe = StableDiffusion3Pipeline.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    cache_dir=self.cache_dir,
                )

            self.pipe.scheduler = FlashFlowMatchEulerDiscreteScheduler.from_pretrained(
                model_path,
                subfolder="scheduler",
                cache_dir=self.cache_dir,
            )
            self.pipe.to(self.device)
            self.logger.info("Pipeline initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing pipeline: {str(e)}")
            raise

    def generate_image(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 28,
        guidance_scale: float = 7.0,
    ) -> Dict[str, Any]:
        """
        Generate an image based on the given prompt.

        Args:
            prompt (str): The input prompt for image generation.
            negative_prompt (str): Negative prompt for image generation.
            num_inference_steps (int): Number of inference steps.
            guidance_scale (float): Guidance scale for image generation.

        Returns:
            Dict[str, Any]: A dictionary containing the generated image and metadata.
        """
        try:
            self.logger.info(f"Generating image for prompt: {prompt}")
            result = self.pipe(
                prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            )
            image = result.images[0]
            score = result.nsfw_content_detected[0]

            output = {
                "image": image,
                "score": score,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
            }
            self.logger.info("Image generated successfully")
            return output
        except Exception as e:
            self.logger.error(f"Error generating image: {str(e)}")
            raise

    def save_output(self, output: Dict[str, Any], filename: str):
        """
        Save the output as an HTML file using Plotly.

        Args:
            output (Dict[str, Any]): The output dictionary from generate_image.
            filename (str): The filename to save the HTML output.
        """
        try:
            fig = go.Figure()

            fig.add_trace(
                go.Image(z=output["image"])
            )

            fig.update_layout(
                title=output["prompt"],
                annotations=[
                    dict(
                        text=f"Score: {output['score']:.2f}<br>"
                             f"Negative Prompt: {output['negative_prompt']}<br>"
                             f"Steps: {output['num_inference_steps']}<br>"
                             f"Guidance Scale: {output['guidance_scale']}",
                        showarrow=False,
                        xref="paper",
                        yref="paper",
                        x=0,
                        y=-0.1,
                    )
                ]
            )

            fig.write_html(filename)
            self.logger.info(f"Output saved to {filename}")
        except Exception as e:
            self.logger.error(f"Error saving output: {str(e)}")
            raise

    @staticmethod
    def load_image(image_path: Union[str, Image.Image]) -> Image.Image:
        """
        Load an image from a file path or URL.

        Args:
            image_path (Union[str, Image.Image]): The path to the image file or a URL.

        Returns:
            Image.Image: The loaded image.
        """
        if isinstance(image_path, Image.Image):
            return image_path

        if image_path.startswith(('http://', 'https://')):
            response = requests.get(image_path)
            return Image.open(BytesIO(response.content))
        else:
            return Image.open(image_path)

import logging
import os
from typing import Union, Dict, Any
import torch
from diffusers import StableDiffusion3Pipeline, SD3Transformer2DModel
from peft import PeftModel
from plotly import graph_objects as go
from PIL import Image
import requests
from io import BytesIO

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedImageGenerator:
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        cache_dir: str = None,
        lora_path: str = None
    ):
        self.device = device
        self.cache_dir = cache_dir

        logger.info(f"Initializing model: {model_name}")
        try:
            self.transformer = SD3Transformer2DModel.from_pretrained(
                model_name,
                subfolder="transformer",
                torch_dtype=torch.float16,
                cache_dir=cache_dir
            )

            if lora_path:
                logger.info(f"Loading LoRA from: {lora_path}")
                self.transformer = PeftModel.from_pretrained(self.transformer, lora_path)

            self.pipe = StableDiffusion3Pipeline.from_pretrained(
                model_name,
                transformer=self.transformer,
                torch_dtype=torch.float16,
                text_encoder_3=None,
                tokenizer_3=None,
                cache_dir=cache_dir
            )
            self.pipe.to(device)
            logger.info("Model initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            raise

    def generate_image(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 28,
        guidance_scale: float = 7.0
    ) -> Dict[str, Any]:
        logger.info(f"Generating image for prompt: {prompt}")
        try:
            image = self.pipe(
                prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            ).images[0]
            
            score = self._calculate_score(image)
            
            output = {
                "image": image,
                "score": score,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale
            }
            
            self._save_output_html(output)
            
            logger.info("Image generated successfully")
            return output
        except Exception as e:
            logger.error(f"Error generating image: {str(e)}")
            raise

    def _calculate_score(self, image: Image.Image) -> float:
        # Implement your scoring logic here
        # This is a placeholder implementation
        return 0.5

    def _save_output_html(self, output: Dict[str, Any]) -> None:
        logger.info("Saving output as HTML")
        try:
            fig = go.Figure(data=[go.Image(z=output["image"])])
            fig.update_layout(
                title=f"Generated Image (Score: {output['score']:.2f})",
                xaxis_title="Prompt: " + output["prompt"],
                yaxis_title="Negative Prompt: " + output["negative_prompt"]
            )
            fig.write_html("output.html")
            logger.info("Output saved as HTML")
        except Exception as e:
            logger.warning(f"Error saving output as HTML: {str(e)}")

    @staticmethod
    def load_image(image_path: Union[str, os.PathLike]) -> Image.Image:
        logger.info(f"Loading image from: {image_path}")
        try:
            if image_path.startswith(('http://', 'https://')):
                response = requests.get(image_path)
                image = Image.open(BytesIO(response.content))
            else:
                image = Image.open(image_path)
            return image
        except Exception as e:
            logger.error(f"Error loading image: {str(e)}")
            raise

if __name__ == "__main__":
    generator = AdvancedImageGenerator(
        model_name="stabilityai/stable-diffusion-3-medium-diffusers",
        device="cuda",
        cache_dir="./model_cache",
        lora_path="jasperai/flash-sd3"
    )

    result = generator.generate_image(
        prompt="A raccoon trapped inside a glass jar full of colorful candies, the background is steamy with vivid colors.",
        num_inference_steps=4,
        guidance_scale=0
    )

    print(f"Image generated with score: {result['score']}")
    print("Output saved as 'output.html'")