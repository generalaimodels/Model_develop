import logging
from typing import Union, Dict, Any
import os
import requests
from PIL import Image
import torch
import numpy as np
import plotly.graph_objects as go
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from pathlib import Path

# 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DepthEstimator:
    def __init__(
        self,
        model_name: str = "LiheYoung/depth-anything-small-hf",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        cache_dir: str = None,
    ):
        """
        Initialize the DepthEstimator class.

        Args:
            model_name (str): Name of the pre-trained model to use.
            device (str): Device to run the model on ('cuda' or 'cpu').
            cache_dir (str, optional): Directory to cache the model.
        # Example usage
              estimator = DepthEstimator()
              image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
              output_file = "depth_visualization.html"
              
              try:
                  results = estimator.process_image(image_url, output_file)
                  print(f"Results: {results}")
              except Exception as e:
                  print(f"An error occurred: {str(e)}")
        """
        self.logger = self._setup_logger()
        self.device = device
        self.cache_dir = cache_dir

        self.logger.info(f"Initializing DepthEstimator with model: {model_name}")
        self.image_processor = AutoImageProcessor.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_name, cache_dir=cache_dir).to(device)

    def _setup_logger(self) -> logging.Logger:
        """Set up and return a logger."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def load_image(self, image_path: str) -> Image.Image:
        """
        Load an image from a local path or URL.

        Args:
            image_path (str): Local file path or URL of the image.

        Returns:
            Image.Image: Loaded PIL Image object.
        """
        try:
            if image_path.startswith(('http://', 'https://')):
                self.logger.info(f"Downloading image from URL: {image_path}")
                response = requests.get(image_path, stream=True)
                response.raise_for_status()
                image = Image.open(response.raw)
            else:
                self.logger.info(f"Loading image from local path: {image_path}")
                image = Image.open(image_path)
            return image
        except Exception as e:
            self.logger.error(f"Error loading image: {str(e)}")
            raise

    def estimate_depth(self, image: Image.Image) -> Dict[str, Any]:
        """
        Estimate depth for the given image.

        Args:
            image (Image.Image): Input image.

        Returns:
            Dict[str, Any]: Dictionary containing depth estimation results.
        """
        self.logger.info("Processing image for depth estimation")
        inputs = self.image_processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth

        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image.size[::-1],
            mode="bicubic",
            align_corners=False,
        )

        depth_map = prediction.squeeze().cpu().numpy()
        normalized_depth = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())

        return {
            "depth_map": depth_map,
            "normalized_depth": normalized_depth,
            "min_depth": float(depth_map.min()),
            "max_depth": float(depth_map.max()),
            "mean_depth": float(depth_map.mean()),
        }

    def visualize_depth(self, depth_map: np.ndarray, output_path: str) -> None:
        """
        Visualize the depth map and save it as an interactive HTML plot.

        Args:
            depth_map (np.ndarray): The depth map to visualize.
            output_path (str): Path to save the output HTML file.
        """
        self.logger.info(f"Generating depth visualization and saving to {output_path}")
        fig = go.Figure(data=go.Heatmap(z=depth_map, colorscale='Viridis'))
        fig.update_layout(title='Depth Map Visualization')
        fig.write_html(output_path)

    def process_image(self, image_path: str, output_path: str) -> Dict[str, Any]:
        """
        Process an image for depth estimation and visualization.

        Args:
            image_path (str): Path or URL of the input image.
            output_path (str): Path to save the output HTML visualization.

        Returns:
            Dict[str, Any]: Dictionary containing all results and metadata.
        """
        try:
            image = self.load_image(image_path)
            results = self.estimate_depth(image)
            self.visualize_depth(results['normalized_depth'], output_path)

            results.update({
                "input_image_path": image_path,
                "output_visualization_path": output_path,
                "image_size": image.size,
            })

            self.logger.info("Image processing completed successfully")
            return results
        except Exception as e:
            self.logger.error(f"Error processing image: {str(e)}")
            raise



class DepthEstimator_test:
    """A class for depth estimation using pre-trained models."""

    def __init__(
        self,
        model_name: str = "LiheYoung/depth-anything-small-hf",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        cache_dir: Union[str, Path] = None,
    ):
        """
        Initialize the DepthEstimator.

        Args:
            model_name (str): The name of the pre-trained model to use.
            device (str): The device to use for computations.
            cache_dir (Union[str, Path]): The directory to cache model files.
            estimator = DepthEstimator_test()
            url = "http://images.cocodataset.org/val2017/000000039769.jpg"
            output_file = "depth_estimation_result.html"
            
            try:
                results = estimator.process_image(url, output_file)
                logger.info(f"Depth estimation results: {results}")
            except Exception as e:
        logger.error(f"Failed to process image: {e}")
        """
        self.device = device
        self.cache_dir = Path(cache_dir) if cache_dir else None

        logger.info(f"Initializing DepthEstimator with model: {model_name}")
        self.image_processor = AutoImageProcessor.from_pretrained(
            model_name, cache_dir=self.cache_dir
        )
        self.model = AutoModelForDepthEstimation.from_pretrained(
            model_name, cache_dir=self.cache_dir
        ).to(self.device)

    def load_image(self, image_source: Union[str, Path]) -> Image.Image:
        """
        Load an image from a file path or URL.

        Args:
            image_source (Union[str, Path]): The path or URL of the image.

        Returns:
            Image.Image: The loaded image.
        """
        if isinstance(image_source, str) and image_source.startswith(("http://", "https://")):
            logger.info(f"Loading image from URL: {image_source}")
            return Image.open(requests.get(image_source, stream=True).raw)
        else:
            logger.info(f"Loading image from file: {image_source}")
            return Image.open(image_source)

    def estimate_depth(self, image: Image.Image) -> Dict[str, Any]:
        """
        Estimate depth for the given image.

        Args:
            image (Image.Image): The input image.

        Returns:
            Dict[str, Any]: A dictionary containing depth estimation results.
        """
        logger.info("Preparing image for depth estimation")
        inputs = self.image_processor(images=image, return_tensors="pt").to(self.device)

        logger.info("Performing depth estimation")
        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth

        logger.info("Interpolating depth map to original image size")
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image.size[::-1],
            mode="bicubic",
            align_corners=False,
        )

        depth_map = prediction.squeeze().cpu().numpy()
        normalized_depth = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())

        return {
            "depth_map": depth_map,
            "normalized_depth": normalized_depth,
            "min_depth": float(depth_map.min()),
            "max_depth": float(depth_map.max()),
            "mean_depth": float(depth_map.mean()),
        }

    def visualize_depth(self, depth_map: np.ndarray, output_path: Union[str, Path]) -> None:
        """
        Visualize the depth map and save it as an HTML file.

        Args:
            depth_map (np.ndarray): The depth map to visualize.
            output_path (Union[str, Path]): The path to save the HTML file.
        """
        logger.info(f"Generating depth visualization and saving to {output_path}")
        fig = go.Figure(data=go.Heatmap(z=depth_map, colorscale="Viridis"))
        fig.update_layout(title="Depth Estimation")
        fig.write_html(output_path)

    def process_image(
        self, image_source: Union[str, Path], output_path: Union[str, Path]
    ) -> Dict[str, Any]:
        """
        Process an image for depth estimation and visualization.

        Args:
            image_source (Union[str, Path]): The path or URL of the input image.
            output_path (Union[str, Path]): The path to save the visualization.

        Returns:
            Dict[str, Any]: A dictionary containing depth estimation results and metadata.
        """
        try:
            image = self.load_image(image_source)
            results = self.estimate_depth(image)
            self.visualize_depth(results["depth_map"], output_path)

            results["image_source"] = str(image_source)
            results["output_path"] = str(output_path)
            results["image_size"] = image.size

            logger.info("Image processing completed successfully")
            return results
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            raise

